This project analyses the H->ZZ->4l (four-lepton) data from the ATLAS Open Data portal. The aim is to reproduce the invariant-mass distribution of the four-lepton system and related histograms, using both single-node and distributed (cluster) execution. The code base contains scripts and notebooks that read ATLAS xAOD/DAOD files, convert the events into pandas/Dask data frames and then compute histograms for the Higgs boson candidate events. The assignment demonstrates how to package the analysis into a container, automate the environment setup, and scale the computation across multiple cores or nodes using a Python distributed computing framework.

## Prerequisites

The project has been tested on Linux and on Windows using **WSL2**. You will need:

- A Unix-like environment:
  - **Linux** or **macOS**, or
  - **Windows 10/11 with WSL2** enabled and an Ubuntu (or similar) distro.
- **Python 3.10+** (3.11/3.12 recommended).
- Either:
  - **conda / mamba**, or  
  - the built-in `venv` module for virtual environments.
- **Docker** (optional) if you want to run the analysis inside a container.
- **Git** (optional) for cloning the repository.

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-URL> CloudTech
   cd CloudTech
   ```

2. **Create and activate a Python environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   (If you prefer conda/mamba, create and activate a conda env instead and skip the `venv` command.)

3. **Install dependencies**

   Dependencies are listed in `requirements.txt`. Install them with:

   ```bash
   pip install -r requirements.txt
   ```

4. **Build the Docker image (optional)**

   A `Dockerfile` is provided that installs all required system packages (build tools, etc.) and copies the analysis scripts into the image. Build the image using:

   ```bash
   docker build -t cloudtech:latest .
   ```

## Data download

The data used in this assignment is the publicly available four-lepton (4lep) dataset from the ATLAS Open Data portal.

1. **Get the data**

   Visit:  
   https://opendata.cern.ch/record/15005

2. **Download the ZIP archive**

   Download the `4lep.zip` file from the record.

3. **Extract the data**

   Unzip the archive. It will produce a directory containing multiple ROOT files.

4. **Place the data**

   Create the data directories and move the extracted files:

   ```bash
   mkdir -p data/raw
   unzip /path/to/4lep.zip -d data/raw
   ```

   The final directory layout should look like:

   ```text
   CloudTech/
     ├── data/
     │   └── raw/
     │       ├── DAOD_HIGG4D1.00000.root
     │       ├── DAOD_HIGG4D1.00001.root
     │       └── ...
   ```

The analysis scripts look for the raw ROOT/xAOD files under `data/raw/` by default. If your data is stored elsewhere, use the `--data-dir` option (see below).

## Configuration

All analysis settings live in `config/config.yaml`. You can either edit `config/config.yaml` directly or override options (such as the number of workers or data directory) on the command line.

## Running the analysis

The main entry point is:

```bash
python -m src.run_analysis --config config/config.yaml --n-workers <N>
```

This script will:

1. Discover all ROOT files matching `data_dir / file_pattern`.
2. For each file:
   - Load the necessary branches (`lep_pt`, `lep_eta`, `lep_phi`, `lep_E`, `lep_charge`, …).
   - Apply basic lepton-level selection cuts.
   - Require at least 4 leptons and build four-vectors.
   - Reconstruct Z candidates and a four-lepton Higgs candidate.
   - Fill histograms for `m4l`, Z masses, lepton kinematics, etc.
3. Merge per-file histograms and save final plots and NumPy arrays in `outputs/`.

### A. Serial mode (single core)

To get a reference timing and check everything works, run in strictly serial mode using a single worker:

```bash
python -m src.run_analysis --config config/config.yaml --n-workers 1
```

Alternatively, you can set `n_workers: 1` inside `config/config.yaml` and omit the `--n-workers` flag.

### B. Parallel / “distributed” mode (multi-core, ProcessPoolExecutor)

To process input files in parallel using Python’s `ProcessPoolExecutor`:

```bash
python -m src.run_analysis --config config/config.yaml --n-workers <N>
```

Notes:

- The script will clamp the number of workers to the number of available CPU cores.
- You can also set `n_workers` in `config/config.yaml` and run:

  ```bash
  python -m src.run_analysis --config config/config.yaml
  ```

- Each worker processes a different subset of the ROOT files, so the scaling behaviour is straightforward to test (e.g. N = 1, 2, 4, 8).

### C. Distributed execution with Dask

If you want to run the same per-file analysis using Dask Distributed instead of the built-in process pool, use the helper utilities under `src/distributed/`.

A minimal example (Python shell or notebook):

```python
from glob import glob
from src.run_analysis import process_file, load_config
from src.distributed.executor import create_local_client, build_tasks

config = load_config("config/config.yaml")
files = sorted(glob("data/raw/*.root"))

# Start a local Dask cluster
client = create_local_client(n_workers=4, threads_per_worker=1)

# Create one delayed task per input file
tasks = build_tasks(files, process_file, config)

# Execute on the Dask cluster
results = client.gather(tasks)
```

- `create_local_client` starts a local Dask `Client` using multiple workers on your machine.
- For a real cluster, point `Client` at a remote scheduler instead of using `create_local_client`, keeping the same `process_file` and `config`.

This lets you re-use the exact same per-file logic as the serial/ProcessPool case, but with the scheduling handled by Dask.

## Running with Docker

This repository includes a `Dockerfile` configured to run the analysis in a self-contained container based on `python:3.12-slim`.

The Dockerfile:

- Installs basic build tools (`build-essential`, `gcc`).
- Installs all Python dependencies from `requirements.txt`.
- Copies `src/` and `config/` into `/app` inside the container.
- Sets `WORKDIR` to `/app`.
- Uses the default command:

  ```bash
  python -m src.run_analysis --config config/config.yaml
  ```

### 1. Build the Docker image

From the project root (where the `Dockerfile` lives):

```bash
docker build -t cloudtech:latest .
```

### 2. Prepare data and outputs on the host

On your host (or inside WSL in the project folder), make sure the data and output directories exist:

```bash
mkdir -p data/raw
mkdir -p outputs
```

Copy or unzip the ATLAS `4lep` ROOT files into `data/raw` as described in the _Data download_ section. The container will see these via a bind mount.

### 3. Run the analysis in Docker (default command)

The Dockerfile’s default `CMD` already runs the main analysis script with the standard configuration. To run it and mount your data and outputs:

```bash
docker run --rm   -v "$PWD/data:/app/data"   -v "$PWD/outputs:/app/outputs"   cloudtech:latest
```

- `/app/data` inside the container is your `data/` directory on the host.
- `/app/outputs` inside the container is your `outputs/` directory on the host (plots and NumPy arrays will appear there).

### 4. Run with explicit options (override the default command)

If you want to explicitly choose the number of workers or a different config file, you can override the default `CMD` by passing a command after the image name:

```bash
docker run --rm   -v "$PWD/data:/app/data"   -v "$PWD/outputs:/app/outputs"   cloudtech:latest   python -m src.run_analysis --config config/config.yaml --n-workers 4
```

This uses the same image, but replaces the default command with the one given at the end.

## Running the test suite

This project includes a full pytest test suite under the `tests/` directory.

To run all tests, first activate your environment and install the required dependencies:

```bash
pip install -r requirements.txt
pytest -v
```

To show printed output during tests (useful when debugging):

```bash
pytest -s
```

The tests check both the physics utilities (four-vector construction, invariant mass) and the I/O selection logic, providing a safety net when you modify the analysis or configuration.
