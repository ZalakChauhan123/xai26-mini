### Explainable Graph Neural Networks for Node Classification using AIFB Dataset ###
Group project for the Explainable AI course, Summer Semester 2026.

We build an end-to-end, reproducible machine learning pipeline to train a Graph Neural Network (GNN) on knowledge graph data and explain its node classification predictions using two complementary Explainable AI (XAI) methods: GNNExplainer (for local sub-graph structural explanations) and Captum via Integrated Gradients (for feature attribution and faithfulness benchmarking). All visual outputs and quantitative reports are automatically compiled into a centralized directory structure.

### Pipeline Overview ###
The workflow is designed for end-to-end execution, moving from graph feature processing to model training and multi-method explainability evaluation:

## Pipeline Overview

| Step | Script | Output / Artifacts | Description |
|---|---|---|---|
| **Step 1** | `src/feature_engineering.py` | `data/processed/` | Graph normalization & feature extraction |
| **Step 2** | `src/model_train.py` | `models/saved_models/` | Model weights & training performance curves |
| **Step 3** | `src/model_explainability.py` | `mega_folder/plots/` | GNNExplainer local sub-graph masks |
| **Step 4** | `src/model_explainability_Carpus.py` | `Evaluate_explaination.txt` | Captum Integrated Gradients attributions |


model.sh serves as the master orchestration script. It initializes the required output directories (such as mega_folder/plots/ and mega_folder/reports/) and sequentially executes the Python scripts in src/. The interactive Jupyter notebooks in notebooks/ mirror this exact pipeline for step-by-step visual debugging and experimentation.

### Requirements ###
- OS: Linux, macOS, or Windows (via WSL2 recommended for bash script execution)

- Python: Version 3.10 or later

- Package Manager: pip or uv (recommended for resolving Windows/CUDA environment dependencies)
