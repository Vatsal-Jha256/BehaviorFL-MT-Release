FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-cpu.txt

COPY . .

ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTEST_ADDOPTS="-o cache_dir=/tmp/behaviorfl-mt-pytest-cache"

RUN chmod +x scripts/reproduce.sh

CMD ["bash", "scripts/reproduce.sh", "all"]
