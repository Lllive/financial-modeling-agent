# Resume Project Entry — English Version

---

## Project Entry (Bullet-Point Style, for Resume / CV)

**Agent-Driven Financial Modeling System** | Python · LangChain · PyTorch · XGBoost  
*Personal Project · Feb 2026*

- Built an end-to-end automated machine learning pipeline powered by a **LangChain ReAct Agent**, encapsulating data exploration, preprocessing, feature engineering, model training, and evaluation as **6 composable LangChain Tools**
- Implemented **strict temporal train/test splitting** (80/20 by `trade_date`) with automated data-leakage auditing, ensuring zero future information leakage in a real-world financial time-series dataset (81,046 rows × 321 columns)
- Applied **variance filtering + mutual information** to reduce feature dimensionality from 300 → 100, improving signal-to-noise ratio and training efficiency
- Trained and compared 4 model families under class imbalance (1:6 positive/negative ratio): **Logistic Regression**, **XGBoost**, **LightGBM**, and a **3-layer PyTorch MLP** (BatchNorm + Dropout + Adam), each with `scale_pos_weight` correction
- Achieved AUC = **0.5586** (best model: Logistic Regression) on the held-out test set; generated comprehensive evaluation artifacts including **ROC curves**, **confusion matrix**, and **feature importance plots**

---

## Project Entry (Paragraph Style, for Project Description Section)

**Agent-Driven Financial Modeling System** | Python · LangChain · PyTorch · XGBoost · LightGBM  
*Feb 2026*

Designed and implemented an autonomous financial modeling agent using the **LangChain ReAct (Reasoning + Acting)** pattern. The agent orchestrates a complete ML pipeline without requiring an external LLM API: each pipeline stage—EDA, preprocessing, data-leakage detection, feature selection, multi-model training, and evaluation—is wrapped as a dedicated LangChain Tool sharing state through a global context dictionary. The dataset consists of 81,046 samples with 300 numeric features and 12 multi-class financial signal labels. A time-aware 80/20 split preserves temporal ordering to prevent look-ahead bias, and fit/transform isolation ensures no test-set information contaminates preprocessing. Four models (Logistic Regression, XGBoost, LightGBM, PyTorch MLP) are trained with explicit class-imbalance correction and benchmarked on AUC, Precision, Recall, and F1. Full visualization suite (ROC curves, confusion matrix, performance bar charts, MI-based feature importance) is auto-generated and saved.

---

## Tech Stack Keywords (for ATS / keyword matching)

`Python 3.12` · `LangChain` · `ReAct Agent` · `PyTorch` · `XGBoost` · `LightGBM` · `scikit-learn` · `pandas` · `numpy` · `Jupyter Notebook` · `Time-Series ML` · `Imbalanced Classification` · `Feature Engineering` · `Mutual Information` · `AUC/ROC` · `Financial Signal Prediction`
