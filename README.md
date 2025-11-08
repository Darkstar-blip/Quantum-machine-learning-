# Quantum-machine-learning-
Benchmark small quantum machine learning approaches (quantum kernels and QNNs) against strong classical baselines on small tasks, with an emphasis on reproducible experiments and pinned environments for fair, repeatable comparisons. The codebase appears primarily in Python with auxiliary components in C/C++/Tcl for performance or vendor tooling integration, reflecting a mixed-language workflow often used in scientific benchmarking pipelines.​

Highlights
Clear, minimal baselines for quantum kernels and variational quantum neural networks on small datasets to enable apples-to-apples comparisons with classical embeddings and models.​

Reproducible runs via version pinning and deterministic configs to support verifiable results across machines and time.​

Lightweight structure suitable for iterative experiments, result logging, and report-ready tables/figures for a comparative study deliverable.​

Repository structure
src/ … Library code for models, kernels, circuits, data loaders, metrics, and utilities (<create modules if not present>).​

experiments/ … Configs and scripts describing each experiment, seeds, and output paths (<add .yaml/.toml as preferred>).​

notebooks/ … Exploration, sanity checks, and result visualization notebooks suitable for a paper appendix.​

data/ … Small cached datasets or loaders; prefer automatic download/load to avoid committing large files.​

results/ … Saved metrics, tables, and plots; keep under version control if small or use artifact storage if large.​

scripts/ … CLI entry points for training, evaluation, sweeping, and plotting (<add run files if not present>).​

Languages
Python ~60.9% for experiment logic, models, and analysis.​

C ~28.9% and C++ ~0.8% for performance-critical or extension components as needed.​

Tcl ~5.0%, Roff ~2.7%, Makefile ~0.6%, and other ~1.1% for build, tooling, or ancillary tasks.​

Methods compared
Family	Representative approaches	Notes
Quantum kernels	Feature map circuits with kernel evaluation, SVM/GS processes ​	Strong on small-data regimes; careful regularization and shot-noise treatment required ​
QNNs (VQCs)	Shallow variational circuits optimized by gradient-based or gradient-free methods ​	Sensitive to barren plateaus and optimizer choice; compare across seeds ​
Classical baselines	PCA/Kernel methods/Neural embeddings + linear or tree models ​	Provide competitive, well-tuned baselines for fairness ​
Installation
Python only (virtualenv): create and activate a virtual environment, then install pinned requirements for deterministic runs.​

Conda/mamba: optionally use environment.yml with exact versions and build strings; prefer strict channel priority for stability.​

Optional extras: separate extras for dev (lint/test/format) and plotting to keep runtime slim.​

Example (pip + venv):

bash
python -m venv .venv
. .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt  # pinned versions recommended
Example (conda/mamba):

bash
mamba env create -f environment.yml
mamba activate qml-bench
Pinning workflow suggestions:

bash
# pip-tools
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt
pip-sync requirements.txt

# or uv
uv pip compile pyproject.toml -o requirements.txt
uv pip sync requirements.txt
Quickstart
Prepare data: ensure data loaders are configured to download or locate datasets automatically to avoid committing large files.​

Run a classical baseline: execute a baseline training/eval script or notebook to establish a reference metric on each dataset.​

Run quantum kernels: evaluate circuit-based kernels with identical splits and metrics to compare fairly with classical baselines.​

Run QNNs: train shallow VQCs with fixed seeds, logging metrics and wall-clock time comparable to classical runs.​

Example CLI pattern (adapt to your file names):

bash
python scripts/run_experiment.py \
  --method quantum_kernel \
  --dataset <name> \
  --seed 0 \
  --shots 2048 \
  --out results/<dataset>/kernel/seed0.json
Reproducibility
Seeds: set and record seeds for numpy/torch/random and any quantum sampler to stabilize comparisons across runs.​

Configs: store all hyperparameters and circuit settings in tracked config files (.yaml/.toml) committed to the repo for provenance.​

Logs and artifacts: write metrics to JSON/CSV and plots to PNG/SVG, using consistent filenames per dataset/method/seed for easy aggregation.​

Datasets and tasks
Small tabular or low-dimensional classification tasks recommended; document input dimension, train/val/test splits, and preprocessing steps.​

For each dataset, report accuracy/F1/AUC as appropriate, plus wall-clock, parameter count, and (for quantum) shot count or simulator backend details.​

Results reporting
Provide per-dataset tables comparing quantum kernels, QNNs, and classical baselines with mean ± std across seeds.​

Include scaling notes: sensitivity to circuit depth, shot noise, and optimizer choice; discuss compute footprint and stability.​

Development
Formatting and linting: add pre-commit with black, isort, flake8/ruff to keep diffs minimal and style unified across contributions.​

Testing: include unit tests for kernels, circuit builders, and metric computations to guard against regressions as methods evolve.​

Contributing
Open an issue describing the method, dataset, or optimization to be added, and align on evaluation protocol before a PR.​

Submit focused PRs with updated configs, pinned dependencies if needed, and a brief result snippet validating the change.​

Citation
If you use this repository in academic work, please cite the repository and any upstream libraries or datasets as appropriate; add a CITATION.cff for convenience..
