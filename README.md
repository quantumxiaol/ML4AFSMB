# ML4AFSMB
## This is the original data of article: Data-Driven Knowledge Discovery Reveals Quantitative Electrolyte Design Rules for Anode-Free Sodium Metal Batteries
### If you use this code in your research, please cite this article:https://doi.org/10.1021/jacs.6c05130  
## Project Structure

```
.
├── All.py                      # Main analysis script (run this)
├── requirements.txt            # Full Python dependencies (pip freeze)
├── environment.yml             # Conda environment specification
├── Electrolytes.xlsx           # Training dataset (molecular descriptors + LCE)
├── Validation_dataset.xlsx     # External validation dataset
├── Reference.xlsx              # Supplementary reference data
├── .gitignore
└── README.md
```

After running `All.py`, the following will be generated:

```
├── graph/                      # 31 PDF figures (see Output section)
├── data/                       # Intermediate data files (.pkl, .xlsx)
├── rf_model_LCE.pkl            # Trained Random Forest model
├── scaler_LCE.pkl              # Feature scaler
└── LCE_CV_Results.xlsx         # Cross-validation results
```

## Quick Start

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Windows / Linux / macOS

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/langercuccy-sudo/ML4AFSMB.git
cd ML4AFSMB

# Create and activate the conda environment
conda env create -f environment.yml
conda activate lce_prediction

# Run the analysis
python All.py
```

### Option 2: Manual pip Installation

```bash
# Create a Python 3.10 virtual environment
conda create -n lce_prediction python=3.10 -y
conda activate lce_prediction

# Install core dependencies manually
pip install numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 matplotlib scikit-learn xgboost lightgbm mlxtend shap statsmodels seaborn openpyxl joblib
```

> **Note**: If `minepy` fails to build, it's safe to skip — it is not used by `All.py`. Simply install the core dependencies manually:
> ```bash
> pip install numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 matplotlib==3.7.3 seaborn==0.13.2 scikit-learn==1.3.2 openpyxl==3.1.5 xgboost==2.1.1 lightgbm==4.5.0 shap==0.44.1 joblib==1.4.2 statsmodels==0.14.1 mlxtend==0.23.4
> ```

## Dependencies

Core dependencies used by `All.py`:

- **Python**: 3.10+
- **Data**: numpy, pandas, scipy
- **ML**: scikit-learn, xgboost, lightgbm, mlxtend
- **Interpretability**: shap, statsmodels
- **Visualization**: matplotlib, seaborn
- **I/O**: openpyxl, joblib

See `requirements.txt` for the full dependency list.

## Reproducibility

All random seeds are fixed (`random_state=42`). Running the script on the same data files will produce identical results.


