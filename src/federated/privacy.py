"""Opacus wrappers and privacy accounting for federated training."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant
    from opacus.grad_sample.utils import register_grad_sampler
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available. Privacy features will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define OpacusCompatibleWrapper at module level so grad sampler registration persists
class OpacusCompatibleWrapper(nn.Module):
    """Wrapper that returns single tensor instead of tuple for Opacus."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        # Support both attribute naming conventions:
        # FedRouteFMTL used num_poi_categories / num_music_genres
        # BehaviorFLModel uses num_poi_classes / num_music_classes
        self.num_poi_categories = (
            getattr(base_model, "num_poi_categories", None) or
            getattr(base_model, "num_poi_classes", 10)
        )
        self.num_music_genres = (
            getattr(base_model, "num_music_genres", None) or
            getattr(base_model, "num_music_classes", 10)
        )
        self._last_poi_logits = None
        self._last_music_logits = None
        self._last_poi_labels = None
        self._last_music_labels = None

    def forward(self, x: torch.Tensor, tier_mask: "Optional[torch.Tensor]" = None) -> torch.Tensor:
        """Return concatenated task logits for Opacus.

        Forward tier_mask to the base model so missing-feature filling remains
        active. Labels and loss computation stay in the training loop. The
        output has shape [batch_size, num_poi_categories + num_music_genres].
        """
        # Get logits from base model, forwarding tier_mask for learned imputation
        poi_logits, music_logits = self.base_model(x, tier_mask)

        # Store logits for later access
        self._last_poi_logits = poi_logits
        self._last_music_logits = music_logits

        # Opacus's functorch backend expects logits (raw outputs), not losses
        # It will compute the loss internally. We return concatenated logits.
        # Concatenate logits along feature dimension: [batch_size, num_poi + num_music]
        concatenated_logits = torch.cat([poi_logits, music_logits], dim=1)

        return concatenated_logits


# Register grad samplers to prevent functorch fallback
# Even when using "hooks" mode, Opacus may fall back to functorch for modules
# without registered grad samplers. We register samplers for container modules
# to ensure hooks mode is used consistently.

if OPACUS_AVAILABLE:
    # Register grad sampler for OpacusCompatibleWrapper — no direct parameters in wrapper
    @register_grad_sampler(OpacusCompatibleWrapper)
    def compute_opacus_wrapper_grad_sample(
        module: OpacusCompatibleWrapper,
        activations: list,
        backprops: torch.Tensor,
    ) -> dict:
        return {}

    # Register grad sampler for nn.Sequential — container with no direct parameters
    @register_grad_sampler(nn.Sequential)
    def compute_sequential_grad_sample(
        module: nn.Sequential,
        activations: list,
        backprops: torch.Tensor,
    ) -> dict:
        return {}

    # Grad samplers for BehaviorEncoderMLP and BehaviorFLModel are registered
    # lazily via _register_behaviorfl_grad_samplers() inside attach_to_model(),
    # guaranteeing correct sys.path in Ray workers.


def _register_behaviorfl_grad_samplers():
    """Register Opacus grad samplers for BehaviorFL container modules.

    Called right before make_private() to ensure sys.path is set in Ray workers.
    Safe to call multiple times — registration is idempotent.
    """
    if not OPACUS_AVAILABLE:
        return
    try:
        from models.behavior_encoder import BehaviorEncoderMLP

        @register_grad_sampler(BehaviorEncoderMLP)
        def _encoder_sampler(module, activations, backprops):
            # Container module — no direct parameters; sub-modules handled natively.
            return {}

    except (ImportError, Exception):
        pass

    try:
        from models.behaviorfl_model import BehaviorFLModel

        @register_grad_sampler(BehaviorFLModel)
        def _model_sampler(module, activations, backprops):
            # Container module — for DP use_uncertainty_weighting must be False
            # so there are no direct log_sigma_* parameters.
            return {}

    except (ImportError, Exception):
        pass


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    sample_rate: float = 0.01  # Fraction of dataset per batch


class DifferentialPrivacy:
    """
    Differential Privacy implementation using Opacus.

    Uses Rényi Differential Privacy (RDP) accountant for accurate
    privacy budget tracking.
    """

    def __init__(self, config: PrivacyConfig):
        """
        Initialize DP mechanism.

        Args:
            config: Privacy configuration
        """
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required for differential privacy")

        self.config = config
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.rdp_accountant: Optional[RDPAccountant] = None

        logger.info(f"Initialized DifferentialPrivacy: ε={config.epsilon}, δ={config.delta}")

    def _replace_dropout_with_identity(self, model: nn.Module) -> nn.Module:
        """
        Replace all Dropout layers with Identity layers for Opacus vmap compatibility.

        Opacus's functorch backend uses vmap for per-sample gradients, which doesn't
        support random operations like Dropout. We replace Dropout with Identity
        (no-op) to maintain model structure while avoiding vmap errors.

        Args:
            model: PyTorch model

        Returns:
            Model with Dropout replaced by Identity
        """
        def replace_dropout_recursive(module):
            """Recursively replace Dropout layers."""
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Dropout):
                    # Replace Dropout with Identity (no-op)
                    setattr(module, name, nn.Identity())
                    logger.debug(f"Replaced Dropout at {name} with Identity")
                else:
                    # Recursively process child modules
                    replace_dropout_recursive(child)

        replace_dropout_recursive(model)
        return model

    def _wrap_model_for_opacus(self, model: nn.Module) -> nn.Module:
        """
        Wrap model to return single tensor for Opacus compatibility.

        Opacus's functorch backend doesn't support models that return tuples.
        This wrapper returns a single combined loss value that Opacus can process.
        The actual logits are stored for later access.

        The wrapper stores labels passed via forward() and computes
        the combined loss from logits. This allows Opacus to compute per-sample
        gradients correctly.

        Args:
            model: PyTorch model that returns (poi_logits, music_logits) tuple

        Returns:
            Wrapped model that returns single tensor [batch_size] (per-sample losses)

        """
        # OpacusCompatibleWrapper is defined at module level with grad sampler already registered
        return OpacusCompatibleWrapper(model)

    def attach_to_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        Attach privacy engine to model and optimizer.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (will be recreated if model is fixed)
            data_loader: DataLoader for training

        Returns:
            (model, optimizer, privacy_engine) tuple
        """
        # Replace Dropout layers FIRST (before wrapping) for Opacus vmap compatibility
        # Opacus's functorch backend (vmap) doesn't support random operations like Dropout
        # Must be done before wrapping so the base model has no Dropout layers
        model = self._replace_dropout_with_identity(model)
        logger.info("Replaced Dropout layers with Identity for Opacus vmap compatibility")

        # Fix model for Opacus compatibility (handles multi-head models)
        # This may modify the model structure, so we need to recreate optimizer
        try:
            model = ModuleValidator.fix(model)
            logger.info("Model fixed for Opacus compatibility")
        except Exception as e:
            logger.warning(f"ModuleValidator.fix() failed: {e}. Model may not be fully compatible.")

        # Wrap model to return single tensor for Opacus functorch compatibility
        # Opacus's functorch backend doesn't support models that return tuples
        # Must be done AFTER Dropout replacement so wrapper's base_model has no Dropout
        model = self._wrap_model_for_opacus(model)
        logger.info("Model wrapped for Opacus functorch compatibility (tuple -> single tensor)")


        # Lazily register Opacus grad samplers for BehaviorFL container modules.
        # Done here (not at import time) to guarantee correct sys.path in Ray workers.
        _register_behaviorfl_grad_samplers()

        # Recreate optimizer with fixed model's parameters.
        # Preserve lr AND weight_decay from the original optimizer so that
        # DP and non-DP runs train under identical regularisation settings.
        lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.01
        wd = optimizer.param_groups[0].get('weight_decay', 0.0) if optimizer.param_groups else 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        logger.info(f"Optimizer recreated with fixed model parameters (lr={lr}, weight_decay={wd})")

        # Create privacy engine
        self.privacy_engine = PrivacyEngine(accountant="rdp")

        # Attach to model and optimizer using make_private
        # Opacus API: make_private returns (model, optimizer, data_loader)
        # Parameters are passed to make_private, not to PrivacyEngine constructor
        # Disable functorch backend for multi-task models
        # functorch backend doesn't support multi-task models with concatenated logits
        # Use 'hooks' backend instead which is more compatible with multi-task models
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.config.noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
            # Use Poisson subsampling so the reported RDP accountant value matches
            # the standard Opacus privacy analysis assumptions.
            poisson_sampling=True,
            grad_sample_mode="hooks"  # Use hooks backend instead of functorch for multi-task compatibility
        )

        # Get RDP accountant from privacy engine
        # The accountant is accessible via privacy_engine.accountant
        self.rdp_accountant = self.privacy_engine.accountant

        logger.info("Privacy engine attached to model")

        return model, optimizer, data_loader

    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get current privacy budget consumption.

        Returns:
            Dictionary with 'epsilon' and 'delta' values
        """
        if self.rdp_accountant is None or self.privacy_engine is None:
            return {'epsilon': 0.0, 'delta': self.config.delta}

        try:
            # Get epsilon from privacy engine's accountant
            # Opacus API: get_privacy_spent returns (epsilon, delta)
            epsilon, delta = self.privacy_engine.accountant.get_privacy_spent(
                delta=self.config.delta
            )
            return {'epsilon': float(epsilon), 'delta': float(delta)}
        except Exception as e:
            logger.warning(f"Error getting privacy spent: {e}")
            return {'epsilon': 0.0, 'delta': self.config.delta}

    def step(self):
        """Step the privacy accountant (called after each training step)."""
        if self.privacy_engine is not None:
            # Privacy accountant is automatically updated by Opacus
            pass

    def reinitialize_hooks(self, model: nn.Module):
        """
        Re-initialize Opacus hooks after model parameters are updated.

        This is needed when parameters are updated via load_state_dict in federated learning,
        as the hooks may lose track of the parameter objects.

        Initialize all required Opacus attributes on parameters.
        load_state_dict with strict=False can create new parameter objects without
        Opacus attributes (grad_sample, _forward_counter, _current_grad_sample),
        causing AttributeError during backward pass.

        This method:
        1. Initializes grad_sample to None if missing
        2. Initializes _forward_counter to 0 if missing
        3. Deletes None _current_grad_sample attributes (Opacus expects it to either
           not exist or be a tensor)
        4. Re-enables hooks if they were disabled

        Args:
            model: The model with updated parameters

        """
        if not OPACUS_AVAILABLE or self.privacy_engine is None:
            return model

        # Check if model is already wrapped by GradSampleModule
        from opacus.grad_sample.grad_sample_module import GradSampleModule
        if isinstance(model, GradSampleModule) and self.privacy_engine is not None:
            # Initialize all required Opacus attributes on parameters
            # load_state_dict with strict=False can create new parameter objects without
            # Opacus attributes, causing AttributeError during backward pass
            initialized_count = 0
            deleted_count = 0

            # First, ensure hooks are enabled
            if hasattr(model, 'enable_hooks'):
                model.enable_hooks()

            # Then initialize all required attributes on each parameter
            for param in model.parameters():
                if param.requires_grad:
                    # Initialize grad_sample if missing (required by DPOptimizer)
                    if not hasattr(param, 'grad_sample'):
                        param.grad_sample = None
                        initialized_count += 1

                    # Initialize _forward_counter if missing (tracks gradient accumulation)
                    # Only initialize if missing - don't reset existing value
                    # Forward hooks will increment it during forward pass
                    if not hasattr(param, '_forward_counter'):
                        param._forward_counter = 0
                        initialized_count += 1
                    # DO NOT reset _forward_counter if it exists - it might have been
                    # incremented by forward hooks in a previous batch

                    # Delete _current_grad_sample if it's None or not a tensor
                    # DO NOT pre-initialize _current_grad_sample
                    # Opacus creates this during backward pass via capture_backprops_hook
                    # It must either not exist or be a valid tensor
                    if hasattr(param, '_current_grad_sample'):
                        if param._current_grad_sample is None:
                            delattr(param, '_current_grad_sample')
                            deleted_count += 1
                        elif not isinstance(param._current_grad_sample, torch.Tensor):
                            # If it's not a tensor, delete it so Opacus can reinitialize
                            delattr(param, '_current_grad_sample')
                            deleted_count += 1

            # Re-register Opacus hooks after load_state_dict
            # The logs show _forward_hooks: 0 and autograd_grad_sample_hooks missing/empty,
            # meaning hooks are not registered. Opacus hooks are registered during make_private
            # via add_hooks(), but load_state_dict can cause hooks to be lost.
            # We need to call add_hooks() again to re-register them.
            try:
                from opacus.grad_sample.grad_sample_module import GradSampleModule
                if isinstance(model, GradSampleModule):
                    # After load_state_dict, hooks may be registered on old module objects.
                    # Even if hooks exist, they might not work because they're attached to the wrong objects.
                    # We must remove old hooks and re-register them on the new module objects.

                    # Check if hooks exist before removal
                    hooks_before_remove = 0
                    if hasattr(model, '_module') and hasattr(model._module, 'autograd_grad_sample_hooks'):
                        hooks_before_remove = len(model._module.autograd_grad_sample_hooks) if model._module.autograd_grad_sample_hooks else 0

                    # Remove existing hooks (they may be on old module objects)
                    if hooks_before_remove > 0:
                        model.remove_hooks()
                        logger.debug(f"Removed {hooks_before_remove} old Opacus hooks before re-registration")

                    # Re-register hooks on the new module objects with same parameters as make_private
                    # Defaults: loss_reduction="mean", batch_first=True, force_functorch=False
                    # These match what we use in make_private (grad_sample_mode="hooks")

                    # Clear any stale activations from previous forward passes
                    # Activations are stored on modules during forward pass and used during backward
                    # After load_state_dict, old activations might be stale
                    # Only clear activations when hooks are being re-registered (after load_state_dict),
                    # NOT during training (when reinitialize_hooks is called before backward pass)
                    # We detect this by checking if hooks already exist - if they do, we're re-registering
                    # and should clear activations. If they don't, we're in training and should NOT clear.
                    if hooks_before_remove > 0:  # Only clear if we're re-registering hooks
                        if hasattr(model, '_module'):
                            # iterate_submodules is not a public Opacus API — use standard modules()
                            for module in model._module.modules():
                                if hasattr(module, 'activations'):
                                    # Clear stale activations - they will be stored fresh during forward pass
                                    module.activations = []
                                if hasattr(module, 'max_batch_len'):
                                    delattr(module, 'max_batch_len')

                    model.add_hooks(
                        loss_reduction="mean",
                        batch_first=True,
                        force_functorch=False
                    )
                    # add_hooks() calls enable_hooks() automatically, but ensure it's enabled
                    model.enable_hooks()
                    # Ensure model is in training mode - forward hooks only store activations if module.training is True
                    model.train()
                    logger.debug("Re-registered Opacus hooks via remove_hooks() + add_hooks() after load_state_dict")

            except Exception as e:
                logger.warning(f"Could not re-register Opacus hooks: {e}")

            if initialized_count > 0 or deleted_count > 0:
                logger.debug(
                    f"Reinitialized Opacus hooks: {initialized_count} attributes initialized, "
                    f"{deleted_count} problematic attributes cleaned up"
                )

        return model


class NoPrivacy:
    """
    No-op privacy mechanism (for ablation studies).

    Returns gradients unchanged.
    """

    def __init__(self, config: PrivacyConfig = None):
        """Initialize no-privacy mechanism."""
        self.config = config or PrivacyConfig()
        logger.info("Initialized NoPrivacy (no privacy protection)")

    def attach_to_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        No-op: return model and optimizer unchanged.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            data_loader: DataLoader

        Returns:
            (model, optimizer, data_loader) unchanged
        """
        return model, optimizer, data_loader

    def get_privacy_spent(self) -> Dict[str, float]:
        """Return zero privacy consumption."""
        return {'epsilon': 0.0, 'delta': 0.0}

    def step(self):
        """No-op."""
        pass


class PrivacyBudgetTracker:
    """
    Tracks cumulative privacy budget consumption across FL rounds.

    Uses RDP accountant for accurate composition (not simple summation).

    RDP composition requires composing at the RDP level, not the (ε, δ)-DP level.
    Simple epsilon summation is incorrect and overestimates privacy loss.

    """

    def __init__(self, initial_epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize privacy budget tracker with persistent RDP accountant.

        Args:
            initial_epsilon: Initial privacy budget
            delta: Privacy parameter
        """
        self.initial_epsilon = initial_epsilon
        self.delta = delta
        self.round_epsilons: List[float] = []

        # Use persistent RDP accountant for proper composition
        if OPACUS_AVAILABLE:
            from opacus.accountants import RDPAccountant
            self.rdp_accountant = RDPAccountant()
        else:
            self.rdp_accountant = None
            logger.warning("Opacus not available. PrivacyBudgetTracker will use simple summation (not recommended).")

        logger.info(f"Initialized PrivacyBudgetTracker: ε={initial_epsilon}, δ={delta}")

    def update(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int
    ):
        """
        Update privacy budget consumption using proper RDP composition.

        Args:
            noise_multiplier: Noise multiplier used in this round
            sample_rate: Sample rate (batch_size / dataset_size) used in this round
            steps: Number of training steps in this round
        """
        if self.rdp_accountant is None:
            # Fallback to simple summation if Opacus not available (not recommended)
            logger.warning("RDP accountant not available. Using simple summation (incorrect for RDP).")
            # Estimate epsilon (rough approximation, not accurate)
            estimated_epsilon = noise_multiplier * sample_rate * steps * 0.1  # Rough estimate
            self.round_epsilons.append(estimated_epsilon)
            return

        # Add all steps from this round to the persistent accountant
        # This properly composes RDP values across rounds
        for _ in range(steps):
            self.rdp_accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )

        # Get cumulative epsilon after this round (properly composed)
        epsilon, _ = self.rdp_accountant.get_privacy_spent(delta=self.delta)
        self.round_epsilons.append(float(epsilon))

        logger.debug(
            f"Privacy budget updated: round_epsilon={epsilon:.4f}, "
            f"noise_multiplier={noise_multiplier}, sample_rate={sample_rate}, steps={steps}"
        )

    def get_remaining_budget(self) -> float:
        """
        Get remaining privacy budget.

        Returns:
            Remaining epsilon (initial - total consumed)
        """
        total_consumed = self.get_total_consumed()
        return max(0.0, self.initial_epsilon - total_consumed)

    def get_total_consumed(self) -> float:
        """
        Get total privacy budget consumed (properly composed via RDP).

        Returns:
            Total epsilon consumed (from RDP accountant)
        """
        if self.rdp_accountant is None:
            # Fallback: sum round epsilons (not accurate)
            return sum(self.round_epsilons) if self.round_epsilons else 0.0

        # Get cumulative epsilon from RDP accountant (properly composed)
        epsilon, _ = self.rdp_accountant.get_privacy_spent(delta=self.delta)
        return float(epsilon)

    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Get current privacy budget consumption (for compatibility with DifferentialPrivacy API).

        Returns:
            Dictionary with 'epsilon' and 'delta' values
        """
        epsilon = self.get_total_consumed()
        return {'epsilon': epsilon, 'delta': self.delta}

    def state_dict(self) -> Dict:
        """
        Get state dictionary for persistence.

        Returns:
            State dictionary containing accountant history
        """
        if self.rdp_accountant is None:
            return {
                'round_epsilons': self.round_epsilons,
                'initial_epsilon': self.initial_epsilon,
                'delta': self.delta
            }

        return {
            'rdp_accountant': self.rdp_accountant.state_dict(),
            'round_epsilons': self.round_epsilons,
            'initial_epsilon': self.initial_epsilon,
            'delta': self.delta
        }

    def load_state_dict(self, state_dict: Dict):
        """
        Load state dictionary.

        Args:
            state_dict: State dictionary to load
        """
        self.round_epsilons = state_dict.get('round_epsilons', [])
        self.initial_epsilon = state_dict.get('initial_epsilon', 1.0)
        self.delta = state_dict.get('delta', 1e-5)

        if self.rdp_accountant is not None and 'rdp_accountant' in state_dict:
            self.rdp_accountant.load_state_dict(state_dict['rdp_accountant'])


def create_privacy(
    privacy_type: str,
    config: PrivacyConfig
) -> object:
    """
    Factory function to create privacy mechanism.

    Args:
        privacy_type: 'dp' for differential privacy, 'none' for no privacy
        config: Privacy configuration

    Returns:
        Privacy mechanism instance
    """
    if privacy_type == 'dp' or privacy_type == 'differential_privacy':
        if not OPACUS_AVAILABLE:
            logger.warning("Opacus not available. Falling back to NoPrivacy.")
            return NoPrivacy(config)
        return DifferentialPrivacy(config)
    elif privacy_type == 'none' or privacy_type == 'no_privacy':
        return NoPrivacy(config)
    else:
        raise ValueError(f"Unknown privacy type: {privacy_type}")
