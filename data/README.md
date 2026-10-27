# Data

The release includes `processed/behaviorfl_dataset.pt`, the feature tensor used
for the experiments (`behaviorfl_multitask_v4`). It contains 22-dimensional trip
prefix features, sensor masks, destination and state labels, supervision masks,
and vehicle-disjoint client splits. Raw GPS and telemetry files are not bundled.

| Source | Retained trips | State labels |
|---|---:|---|
| VED | 30,709 | Telemetry-derived proxy |
| GeoLife | 844 | None |
| T-Drive | 71,290 | None |

There are 5,617 training clients (188 with state supervision) and 2,871 held-out
clients. Destination labels are relative cells in each source's own 4 x 4 grid.
State labels come only from VED and do not represent diagnostic fault codes.

## Get the original datasets

Download the sources below if you want to rebuild the feature tensor. The
bundled tensor is sufficient for training and result reproduction.

| Dataset | Original source | Files used |
|---|---|---|
| VED | [Oh, LeBlanc and Peng's repository](https://github.com/gsoh/VED) | Both dynamic-data archives |
| GeoLife | [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=52367) | GeoLife Trajectories 1.3, including transportation labels |
| T-Drive | [Microsoft Research dataset page](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) | The one-week sample of 10,357 taxis |

Use this layout under the repository root. Dataset names are lowercase; the
folders inside them keep the names used by the original releases.

```text
data/
  raw/
    ved/
      Data/
        VED_DynamicData_Part1/*.csv
        VED_DynamicData_Part2/*.csv
    geolife/
      Data/
        000/
          Trajectory/*.plt
          labels.txt
        ...
    tdrive/
      release/
        taxi_log_2008_by_id/*.txt
  processed/
    behaviorfl_dataset.pt
```

### VED

Clone the original repository into its data directory:

```bash
mkdir -p data/raw
git clone https://github.com/gsoh/VED.git data/raw/ved
```

Extract both `.7z` archives with [7-Zip](https://www.7-zip.org/). With its `7z`
command installed:

```bash
7z x data/raw/ved/Data/VED_DynamicData_Part1.7z -odata/raw/ved/Data/VED_DynamicData_Part1
7z x data/raw/ved/Data/VED_DynamicData_Part2.7z -odata/raw/ved/Data/VED_DynamicData_Part2
```

The extractor searches each part recursively, so an extra folder created by the
archive is supported. The static vehicle spreadsheets are not used.

### GeoLife

Open the Microsoft download page, select **Download**, and extract the GeoLife
Trajectories 1.3 ZIP. Place its `Data` directory at `data/raw/geolife/Data`.
Keep the user directories, `Trajectory` directories and `labels.txt` files
together. The extractor selects the intervals labeled `car` or `taxi`; users
without transportation labels do not contribute to this study. The original
[user guide](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)
describes the formats.

### T-Drive

On the Microsoft Research page, follow **Download the Trajectory Data** to the
publisher's hosted files. Download and extract the complete sample. Place the
`taxi_log_2008_by_id` directory at
`data/raw/tdrive/release/taxi_log_2008_by_id`. Keep all vehicle `.txt` files.
Each record contains a taxi ID, timestamp, longitude and latitude.

## Check and preprocess

After setting up the environment from the root README, check the raw layout:

```bash
scripts/reproduce.sh data-check
scripts/reproduce.sh data-build
scripts/reproduce.sh check
```

`data-check` reports the VED CSV counts, labeled GeoLife users, and T-Drive
vehicle files. It catches missing inputs without writing processed data.
`data-build` reruns extraction and replaces the processed tensor and feature
caches. Save a copy of the bundled tensor first if you need to retain it.

To keep raw data on another drive, use the same three dataset subdirectories
there and pass their parent directory:

```bash
RAW_DATA_DIR=/path/to/raw scripts/reproduce.sh data-check
RAW_DATA_DIR=/path/to/raw scripts/reproduce.sh data-build
```

The direct preprocessing command is:

```bash
python scripts/behavior/assemble_behaviorfl_dataset.py --raw-data-dir data/raw --rebuild-extractors
```

Feature extraction lives in `src/data/behavior_extractors/`; assembly, target
construction and splits are in `scripts/behavior/assemble_behaviorfl_dataset.py`.
Inputs use the first 25% of recorded samples, with source-specific minimums.
Destination targets use final endpoints. The VED state proxy uses full-trip
RPM, engine-load and MAF statistics; its normalization and threshold are fitted
on eligible training trips only. T-Drive traces are segmented at calendar-day
boundaries and gaps longer than 30 minutes, with at most 20 segments per taxi.
The assembler filters trips to the fixed study regions and creates the client
splits. Matching the saved results requires the same source releases and
processing settings.

## Attribution and terms

The original datasets were created by their respective authors, not by the
BehaviorFL-MT authors. Cite the dataset papers when using this release's derived
features. [CITATIONS.bib](CITATIONS.bib) contains the VED paper and all five
papers requested on the GeoLife and T-Drive download pages.

- VED: Geunseob (GS) Oh, David J. LeBlanc and Huei Peng. *Vehicle Energy Dataset
  (VED), A Large-scale Dataset for Vehicle Energy Consumption Research*.
  [DOI](https://doi.org/10.1109/TITS.2020.3035596).
- GeoLife: Microsoft Research Asia; Yu Zheng and collaborators. The download
  page requests the WWW 2009 location-mining paper, the UbiComp 2008 mobility
  paper, and the 2010 GeoLife overview.
- T-Drive: Microsoft Research; Jing Yuan, Yu Zheng and collaborators. The dataset
  page requests the KDD 2011 and SIGSPATIAL GIS 2010 papers.

VED's repository carries the Apache 2.0 license; a copy is in
[licenses/VED-LICENSE.txt](licenses/VED-LICENSE.txt). Keep the notices and terms
supplied with the Microsoft downloads as well. This repository grants no
additional rights over the source datasets or derived data. Raw downloads and
extraction caches are excluded from Git and Docker builds.
