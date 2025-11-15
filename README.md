# Quantum-Assisted Machine Learning (QML) — Exploratory Study

Quantum-assisted ML combines quantum circuits with classical ML to address bottlenecks in optimization, sampling, and high‑dimensional representation learning, with near‑term traction coming from hybrid workflows on NISQ hardware and simulators rather than broad, universal speedups across ML tasks. [Surveys 2023–2025] 

## What QML means today
- Hybrid pipelines: quantum processors act as feature mappers, kernels, or optimization/sampling subroutines within classical loops for small-data classification, structure discovery, and constrained optimization.
- Three families dominate: variational/QNNs, quantum kernels/feature maps, and quantum-accelerated subroutines (linear algebra, clustering); all are constrained by noise, trainability, and data-loading overheads. 

## State of the art and timelines
- Near‑term (NISQ): incremental wins via hybrid inference/preprocessing at 100–200+ physical qubits; specialized advantages are most plausible on quantum or very low‑data distributions; robust proofs of general advantage remain rare. 
- Mid‑term (5–10 yrs): early fault‑tolerant systems with tens of logical qubits may enable deeper, reliable QML pipelines (chemistry/materials/physics simulation), while classical ML dominates mainstream tasks. 

## Algorithm families to explore
- Variational/QNNs: parameterized circuits trained classically; expressive, but sensitive to barren plateaus/noise; prioritize problem‑informed encodings and shallow ansätze. 
- Quantum kernels/feature maps: embed data with circuits and learn via SVM/KRR; examples include ZZFeatureMap with QSVC; promise hinges on encodability and kernel/sampler quality. 
- Quantumized classics: QSVM, qPCA, quantum k‑means, and kNN variants claim speedups under low‑rank/sparsity and special data access (e.g., QRAM), but practicality depends on I/O and hardware constraints. 

## Practical stacks and tooling
- Qiskit Machine Learning: QuantumKernel, QSVC, kernel training (alignment) utilities, and Sampler/Estimator primitives for realistic execution. Start from the Quantum Kernel tutorial. 
- PennyLane: end‑to‑end QML demos, device‑agnostic execution, hybrid autodiff, and QNN templates with PyTorch/TF integration. See the variational classifier demo. 
- TFQ/Cirq: tight TF integration and low-level circuit control; optional for teams standardizing on TF/Cirq stacks. 

## Emerging use cases
- Quantum‑enhanced featurization/kernels for small, structured datasets (chemistry, materials, selected finance niches) where entangling maps may capture correlations missed by standard kernels. 
- Hybrid optimization/sampling to accelerate RL/generative components or inner loops in simulation‑heavy workflows under HPC orchestration. 

## Limits and risks to manage
- Data loading/access: many speedup claims assume QRAM or favorable state preparation; account for encoding cost and sampler noise in wall‑clock. 
- Trainability and noise: barren plateaus, gradient noise, and device infidelity limit depth/width; prefer shallow/problem‑informed ansätze, layer‑wise training, and error mitigation. 

## Getting started resources
- Qiskit Quantum Kernel tutorial: defining kernels, precomputed Gram matrices, and QSVC pipelines. 
- QSVC API and Quantum Kernels docs: scikit‑learn‑style interfaces and evaluate-based Gram computation for parity checks. 
- PennyLane variational classifier demo: AngleEmbedding + Rot/CNOT layers for small‑N classification. 

---

## Pilot plan with go/no‑go criteria

### Week 1 — Repo, baselines, and smoke tests
- Set up repo structure (data/, src/, notebooks/, results/), create virtualenv, install/pin qiskit, qiskit-machine-learning, scikit-learn, pennylane, jupyterlab, matplotlib, seaborn; freeze requirements.txt. 
- Implement classical baselines (RBF SVM/KRR) with stratified K‑fold CV, tuned C/λ and γ, and fixed seeds; log fold metrics to results/. 
- Run Qiskit Quantum Kernel tutorial pattern on a toy dataset: compute precomputed Gram matrices via QuantumKernel.evaluate, train SVC(kernel='precomputed'), export K_train/K_test and fold metrics. 
- Deliverable: baseline metrics + kernel Gram matrices with plots; criteria: reproducible CV loop and artifact logging complete.

### Quantum kernels and diagnostics
- Extend QSVC experiments with small real or synthetic datasets; hyperparameter sweeps over reps, input scaling/bandwidth, and C; unify splits with classical baselines. 
- Add kernel-matrix diagnostics (heatmaps, condition numbers) and optional kernel PCA to relate geometry to separability.
- Deliverable: CV tables and visuals comparing classical RBF vs quantum kernels; criteria: matched splits, pinned versions, and consistent metrics across runs. 

### Projected quantum kernels
- Build projected features φ(x) by measuring simple observables (e.g., Z expectations per qubit) using Sampler quasi_dists; train outer RBF SVM on φ(x) with CV over C and γ. 
- Export φ(x), optional φ‑space Gram matrices, and fold metrics; compare to fidelity kernels and classical baselines. 
- Deliverable: projected‑kernel CV results and artifacts; criteria: stability under shots and small hyperparameter perturbations. 

### QNN and robustness
- Implement a shallow PennyLane variational classifier (AngleEmbedding + Rot/CNOT) on the same small dataset(s); log training curves and test accuracy.
- Robustness: swap statevector for Sampler-based kernel estimation; document shot noise impact, any transpilation settings, and runtime deltas.
- Deliverable: QNN results + noise study notes; criteria: parity of splits and preprocessing with kernels and baselines.
### Week 5 — Consolidation and optional alignment
- Consolidate metrics, best hyperparameters, Gram visuals, and selected learning curves into a concise report (reports/comparative_study.md). 
- Optional: add QuantumKernelTrainer (alignment) to tune feature-map parameters with SVC-style loss on a toy/small dataset; record alignment curves and downstream QSVC gains. 
- Deliverable: draft comparative study; criteria: end-to-end reproducibility (scripts + notebooks) and clear assumptions on encoding and I/O costs. 

### Go/No‑Go and next steps
- Go if any quantum approach matches or exceeds classical baselines under matched CV with meaningful geometric/robustness evidence and acceptable wall‑clock; else No‑Go with rationale and risks. 
- If Go, plan a single small backend run (IBM Runtime Sampler) with shots and transpilation recorded, and define a broader dataset sweep or domain‑specific case (e.g., featurization in chemistry). 
---

## Run commands and artifacts

### Scripts (headless)
- QSVC kernels with precomputed Gram:
  - python scripts/run_qsvc_cv.py → results/qsvc_cv/K_train_*.npy, K_test_*.npy, cv_results.json, best_cfg.json. 
- Projected kernel (φ + outer RBF):
  - python scripts/run_projected_cv.py → results/projected_kernel_cv/Phi_*.npy, Kphi_*.npy, cv_results.json, best_cfg.json. 
- Classical baselines:
  - python scripts/run_baselines.py → results/baselines/cv_results.json, best_cfg.json. 

### Notebooks (interactive)
- notebooks/01_qsvc_kernel.ipynb — QSVC + Gram export (precomputed parity). 
- notebooks/02_pennylane_qnn.ipynb — shallow QNN (AngleEmbedding + Rot/CNOT). 
- notebooks/03_classical_baselines.ipynb — RBF SVM/KRR baselines with matched CV. 
- notebooks/04_projected_kernel.ipynb — projected features + outer RBF SVM. 

### Reporting
- reports/comparative_study.md — datasets, preprocessing, models, CV protocol, metrics, kernel visuals, robustness notes, and recommendations; include links to artifacts in results/.

---

## References
- Qiskit Quantum Kernel tutorial; Quantum Kernels & QSVC docs (evaluate API, precomputed SVC interoperability).
- PennyLane variational classifier demo and QML tutorials hub. 
- Recent QML surveys and outlooks (state, risks, timelines, applications).
- Qiskit Sampler and quasi_dists for measurement-driven features in projected kernels. 
- ZZFeatureMap note: track deprecations and replacements in current Qiskit docs when updating stacks. 
