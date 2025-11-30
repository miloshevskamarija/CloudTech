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

4. **Build the Docker image**

   A `Dockerfile` is provided that installs all required system packages (ROOT, uproot, pandas, Dask, etc.) and copies the analysis scripts into the image. Build the image using:

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

### B. Parallel / “distributed” mode (multi-core)

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

### C. Dask-based distributed execution

The module `src/distributed/executor.py` contains helpers for running the same per-file analysis using Dask Distributed. This allows you to extend from local multi-processing to a Dask cluster (e.g. across multiple nodes) without changing the core analysis logic. For a true multi-node cluster, you can point the Dask client at a remote scheduler instead of creating a local cluster, keeping the same per-file processing function and configuration.

## Viewing logs and progress

The analysis prints progress and a final summary to standard output, including:

- Number of files processed.
- Number of selected events.
- Mean and RMS of the `m4l` distribution.
- Event yields in Z and Higgs mass windows.
- Total wall time and approximate processing rate in events/s.
- Location of the output directory.

To keep a log for later analysis, you can run:

```bash
mkdir -p logs
python -m src.run_analysis --config config/config.yaml --n-workers 4   2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
```

## Outputs

By default, outputs are written to `outputs/` (or the directory given by `output_dir` in `config/config.yaml`).

Typical artefacts include:

- **Plots (PNG):**
  - `outputs/m4l.png` – full `m4l` spectrum.
  - `outputs/m4l_zoom.png` – zoom around the Z and Higgs mass region.
  - `outputs/m4l_log.png` – `m4l` with a logarithmic y-axis.
  - `outputs/m12.png`, `outputs/m34.png` – dilepton invariant masses.
  - `outputs/m12_m34_2D.png` – 2D histogram of the two Z candidates.
  - `outputs/leading_pt.png`, `outputs/lepton_eta.png`, `outputs/deltaR_leading_leptons.png`, etc.
- **NumPy arrays:**
  - `outputs/m4l_edges.npy`
  - `outputs/m4l_counts.npy`

These let you re-plot the `m4l` distribution or combine results from different runs without reprocessing the ROOT files.

You can change the output directory by editing `output_dir` in `config/config.yaml`.

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
