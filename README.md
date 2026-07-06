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
- OS: Linux, macOS, or Windows (via WSL2 recommended for bash script execution) / Use .Bat file if WSL2 doesn't work in Windows

- Python: Version 3.10 or later

- Package Manager: pip or uv (recommended for resolving Windows/CUDA environment dependencies)


## 🔄 Pipeline Architecture

Our automated workflow is orchestrated sequentially through four core stages:

1. **Graph Feature Engineering** (`src/feature_engineering.py`)
   * **Output:** `data/processed/`
   * **Description:** Ingests raw graph files, normalizes node attributes, extracts relational edge indices, and splits data for training and testing.

2. **Model Training & Validation** (`src/model_train.py`)
   * **Output:** `models/saved_models/`, `mega_folder/plots/`
   * **Description:** Trains the Relational Graph Convolutional Network (R-GCN), saves model checkpoints, and generates loss and performance curves.

3. **Local Structural Explainability** (`src/model_explainability.py`)
   * **Output:** `mega_folder/plots/`
   * **Description:** Applies GNNExplainer to identify critical sub-graphs and edge importance masks for individual node predictions.

4. **Attribution Scoring & Faithfulness** (`src/model_explainability_Carpus.py`)
   * **Output:** `Evaluate_explaination.txt`, `mega_folder/reports/`
   * **Description:** Leverages Captum via Integrated Gradients to compute feature-level attributions and quantitatively benchmark explanation fidelity and stability.
  

# 🛠️ Complete Setup, Configuration & Execution Guide

This document provides step-by-step instructions for setting up the environment, configuring dataset paths (with specific instructions for **Windows** users), and executing the full Explainable AI pipeline from start to finish.

---

## 📋 Table of Contents

1. [System Prerequisites](#1-system-prerequisites)
2. [Installation & Environment Setup](#2-installation--environment-setup)
   - [Linux / macOS Setup](#a-linux--macos-setup)
   - [Windows Setup (PowerShell / Command Prompt)](#b-windows-setup-native-powershell--cmd)
   - [Windows Setup (WSL2 / Git Bash Recommended)](#c-windows-setup-wsl2--git-bash-recommended)
3. [Configuring Dataset Paths for Windows](#3-configuring-dataset-paths-for-windows)
   - [Target Files That Require Path Updates](#target-files-that-require-path-updates)
   - [How to Fix Path Slashes for Windows](#how-to-fix-path-slashes-for-windows)
4. [Step-by-Step Execution Commands](#4-step-by-step-execution-commands)
   - [Option A: Automated Execution (Bash / WSL2)](#option-a-automated-execution-via-bash-script)
   - [Option B: Manual Sequential Execution (Windows Native)](#option-b-manual-sequential-execution-windows-native)
5. [Verifying Your Outputs](#5-verifying-your-outputs)

---

## 1. System Prerequisites

Before starting, ensure your operating system has the following installed:
* **Git:** Version control system to clone the repository ([Download Git](https://git-scm.com/downloads)).
* **Python:** Version **3.10** or higher ([Download Python](https://www.python.org/downloads/)).
* **Package Manager:** Standard `pip` (included with Python) or **`uv`** (highly recommended for Windows users to avoid PyTorch Geometric compilation errors).

---

## 2. Installation & Environment Setup

### 1. Clone the Repository
Open your terminal (or Command Prompt/PowerShell on Windows) and clone the project:
```bash
git clone [https://github.com/ZalakChauhan123/xai26-mini.git](https://github.com/ZalakChauhan123/xai26-mini.git)
cd xai26-mini



# Project Setup and Execution Guide

## 2. Create and Activate a Virtual Environment

### A. Linux / macOS Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### B. Windows Setup (Native PowerShell / CMD)

If you are running natively on Windows without WSL:

#### PowerShell

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Command Prompt (CMD)

```cmd
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\activate.bat

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **⚠️ Windows Troubleshooting Note**
>
> If `pip install -r requirements.txt` fails while building PyTorch Geometric wheels or CUDA extensions on Windows, install using **uv**:

```powershell
pip install uv
uv pip install -r requirements.txt
```

---

### C. Windows Setup (WSL2 / Git Bash Recommended)

Because the automated pipeline relies on a Bash script (`model.sh`), using **Windows Subsystem for Linux (WSL2)** or **Git Bash** is the easiest way to run the project identically to a Linux environment.

1. Open **Git Bash** or your **WSL2 Ubuntu terminal** inside the project directory.
2. Run the exact same commands described in the **Linux / macOS Setup** section.

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

### D. Windows Setup (Batch File Alternative)

If **WSL2** or **Git Bash** is unavailable or not functioning correctly on your system, you can execute the complete pipeline using the provided Windows batch file.

The batch file automates the execution of each stage of the pipeline in the correct order, eliminating the need to manually run each Python script.

#### Steps

1. Open **Command Prompt** or **PowerShell**.
2. Navigate to the project root directory.
3. Ensure that your Python virtual environment has been activated.
4. Run the batch file:

```powershell
.\model.bat
```

or from Command Prompt:

```cmd
model.bat
```

The batch file automatically performs the following tasks:

1. Executes graph feature engineering.
2. Trains the Relational GNN model.
3. Generates GNNExplainer visualizations.
4. Runs Captum explainability and faithfulness evaluation.

> **Note:** Before running `model.bat`, ensure that all required dependencies have been installed using:
>
> ```bash
> pip install -r requirements.txt
> ```
>
> Additionally, verify that any required dataset paths have been configured correctly as described in the **Configuring Dataset Paths for Windows** section.

# 3. Configuring Dataset Paths for Windows

When running natively on Windows, file system paths use backslashes (`\`) instead of forward slashes (`/`). If hardcoded Linux paths are present in the scripts, Windows may throw a `FileNotFoundError`.

## Target Files That Require Path Updates

Before running the pipeline on Windows, check and update the data loading and output saving paths in the following source scripts:

### 1. `src/feature_engineering.py`

Responsible for:

- Ingesting raw graph files from `data/raw/`
- Saving processed tensors to `data/processed/`

---

### 2. `src/model_train.py`

Responsible for:

- Loading processed tensors from `data/processed/`
- Saving trained weights to `models/saved_models/`
- Saving training plots to `mega_folder/plots/`

---

### 3. `src/model_explainability.py`

Responsible for:

- Loading trained model weights
- Loading processed datasets
- Saving GNNExplainer masks and visualizations into `mega_folder/plots/`

---

### 4. `src/model_explainability_Carpus.py`

Responsible for:

- Running Captum explainability methods
- Writing attribution logs to `Evaluate_explaination.txt`
- Saving reports into `mega_folder/reports/`

---

## How to Fix Path Slashes for Windows

Open each of the scripts listed above and locate where file paths are defined.

There are two recommended approaches.

---

### Option 1: Use Forward Slashes (Quick Fix)

Python supports forward slashes on Windows.

#### ❌ Bad

```python
data_path = "data\raw\dbpedia_subset.pt"
output_dir = "mega_folder\plots\"
```

#### ✅ Good

```python
data_path = "data/raw/dbpedia_subset.pt"
output_dir = "mega_folder/plots/"
```

---

### Option 2: Use `os.path` or `pathlib` (Recommended)

#### Using `os.path.join`

```python
import os

raw_data_path = os.path.join("data", "raw", "dbpedia_subset.pt")

saved_model_path = os.path.join(
    "models",
    "saved_models",
    "best_rgcn_model.pt"
)
```

#### Using `pathlib`

```python
from pathlib import Path

processed_dir = Path("data") / "processed"

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)
```

This approach is recommended because it automatically generates the correct file separator for Windows, Linux, and macOS.

---

# 4. Step-by-Step Execution Commands

Once your environment is active and dataset paths are configured, execute the pipeline from start to finish.

---

## Option A: Automated Execution via Bash Script

**(Linux, macOS, WSL2, or Git Bash)**

This single command automatically creates the required output folders and executes every stage of the pipeline sequentially.

### Step 1 — Give execution permission

```bash
chmod +x model.sh
```

### Step 2 — Run the complete pipeline

```bash
./model.sh
```

The script executes the following stages automatically:

1. Feature Engineering
2. Model Training
3. GNNExplainer Explainability
4. Captum Explainability and Evaluation

---

## Option B: Manual Sequential Execution (Windows Native)

If you are using **Windows PowerShell** or **Command Prompt**, execute each stage manually.

### Step 1 — Create output directories

PowerShell:

```powershell
mkdir models\saved_models
mkdir mega_folder\plots
mkdir mega_folder\reports
```

---

### Step 2 — Execute Feature Engineering

```powershell
python src/feature_engineering.py
```

---

### Step 3 — Train the Relational GNN

```powershell
python src/model_train.py
```

---

### Step 4 — Generate GNNExplainer Visualizations

```powershell
python src/model_explainability.py
```

---

### Step 5 — Run Captum Explainability & Faithfulness Evaluation

```powershell
python src/model_explainability_Carpus.py
```

---

# 5. Verifying Your Outputs

After the final script (`model_explainability_Carpus.py`) completes successfully, verify that the following artifacts have been generated.

| Output Directory / File | Expected Contents |
|-------------------------|------------------|
| `data/processed/` | Normalized feature tensors, PyTorch Geometric edge-index tensors, processed graph datasets, and train/validation/test split files. |
| `models/saved_models/` | Trained Relational Graph Convolutional Network (R-GCN) model checkpoint (e.g., `best_model.pt` or `.pth`). |
| `mega_folder/plots/` | High-resolution PNG figures including training loss curves, ROC-AUC curves, confusion matrices (if generated), and GNNExplainer subgraph visualizations. |
| `mega_folder/reports/` | CSV, Markdown, or text reports summarizing explainability metrics, benchmarking results, sparsity, fidelity, and evaluation statistics. |
| `Evaluate_explaination.txt` | Detailed numerical feature attribution scores, Captum Integrated Gradients results, execution runtimes, and quantitative faithfulness metrics. |

---

# Pipeline Summary

The complete workflow is illustrated below.

```text
Create Virtual Environment
            │
            ▼
Install Dependencies
            │
            ▼
Configure Dataset Paths
            │
            ▼
Feature Engineering
(src/feature_engineering.py)
            │
            ▼
Model Training
(src/model_train.py)
            │
            ▼
GNNExplainer Explainability
(src/model_explainability.py)
            │
            ▼
Captum Explainability
(src/model_explainability_Carpus.py)
            │
            ▼
Generated Outputs
├── data/processed/
├── models/saved_models/
├── mega_folder/plots/
├── mega_folder/reports/
└── Evaluate_explaination.txt
```

---

# Project Directory Structure

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── saved_models/
│
├── mega_folder/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── feature_engineering.py
│   ├── model_train.py
│   ├── model_explainability.py
│   └── model_explainability_Carpus.py
│
├── model.sh
├── requirements.txt
└── Evaluate_explaination.txt
```
