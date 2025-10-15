# FAIRNESS IN MACHINE LEARNING: SELDONIAN APPROACHES - CODE

**Author**: *[ELENI GEORGANTZI]* **Thesis Project – [2025]*

This repository contains the code developed during my thesis, focusing on fairness-aware regression using synthetic and real datasets. It includes implementations of OLS, soft constraints, NDLR (Non-Discriminatory Linear Regression), and QNDLR (Quasi NDLR) with various fairness metrics and optimization approaches

### **⬇️ Data Download Link ⬇️**

**[DOWNLOAD SYNTHETIC DATASET (all_synthetic_data2.csv)](https://drive.google.com/uc?export=download&id=1rdeuwwPi5p-QitlJmOlLiXefASxSKY2V)**

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `synthetic_data_creation.ipynb` | Generates the synthetic dataset (`all_synthetic_data2.csv`) based on the illustrative example. Find the link to download it above. |
| `OLS_training.ipynb` | Applies OLS to compute regression coefficients and discrimination statistics. Produces `discrimination_statistics_with_coefficients.csv`. |
| `soft_constraints_exp.ipynb` | Experiments with soft fairness constraints. Requires TensorFlow. |
| `ndlr.py` | Implementation of the NDLR algorithm in Python. |
| `ndlr_application.ipynb` | Applies NDLR to synthetic data. Requires `ndlr.py`. |
| `QNDLR.py` | Implementation of QNDLR for 1D features. |
| `QNDLR_application.ipynb` | Applies QNDLR in synthetic data and includes NDLR and QNDLR experiments. Requires both `QNDLR.py` and `ndlr.py`. |
| `QNDLR_MSE.py` | QNDLR for 2D features using MSE. |
| `qndlr_real_data.ipynb` | Applies QNDLR to real-world datasets using `QNDLR_MSE.py`. |
| `QNDLR_MAE.py` | QNDLR variant using MAE loss. |
| `QNDLR_MAE_DIFF.py` | QNDLR variant using MAE and `MAE_diff` discrimination metric. |
| `QNDLR_VS_GSR.ipynb` | Compares QNDLR methods to GSR. Requires `aif360` and calls both `QNDLR_MAE.py` and `QNDLR_MAE_DIFF.py`. |
| `demo_grid_search_reduction_regression_sklearn.ipynb` | Unmodified demo from AIF360’s GitHub. Used for reference. **Do not alter.** |

---

## ⚠️ Reproducibility Notice

> **NDLR, QNDLR, and their variants are stochastic.** Their outputs vary slightly across runs.
> To reproduce the **exact results** from the thesis, **do not re-run** the corresponding cells.
> Pre-generated output files are included.

---

## ⚙️ Requirements

- Python ≥ 3.6
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `aif360`

---

## 🧪 Usage Notes

1.  **Download Data:** First, download `all_synthetic_data2.csv` using the link above.
2.  **Dependencies:** Make sure `.py` dependencies (e.g., `ndlr.py`, `QNDLR.py`) are in the **same folder** as notebooks that import them.
3.  **Run Order:** Begin by installing requirements, then proceed through the notebooks in the order they appear in the project structure table.
4.  **Stochastic Models:** Avoid re-running stochastic models (NDLR, QNDLR, etc.) if exact thesis results are required.
