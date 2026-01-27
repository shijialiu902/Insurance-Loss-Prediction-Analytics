# 🚗 Insurance Loss Analytics & Claim Prediction

**Predicting insurance risk, loss cost, and claim probability using explainable ML**

This project applies **machine learning and analytics** to a real-world auto insurance problem: identifying high-risk policyholders and estimating expected loss to support **risk-adjusted pricing and decision-making**.

---

## 🎯 Problem

Insurance data is **highly imbalanced** (~89% of policyholders file no claims), making traditional models ineffective. The goal was to accurately predict:

* **Loss Cost (LC)** – expected claim cost
* **Historical Loss Cost (HALC)** – frequency-adjusted loss
* **Claim Status (CS)** – probability of filing a claim

while maintaining **interpretability** required in regulated environments.

---

## 🧠 Approach

* Engineered behavioral, policy, driver, and vehicle features
* Used **Tweedie regression** to handle zero-inflated, right-skewed losses
* Applied **gradient boosting (LightGBM, XGBoost)** with 5-fold CV
* Validated robustness using **adversarial validation**
* Interpreted results with **SHAP** for business explainability

---

## 📈 Results

* **LC Prediction:** XGBoost + Tweedie
  → ~**60% MSE improvement** over baseline
* **HALC Prediction:** LightGBM + Tweedie
  → Best fit across low–moderate loss range
* **Claim Status:** LightGBM Classifier
  → **ROC AUC ≈ 0.77**

Key drivers across models:

* Policy cancellations & tenure
* Net premium amount
* Number of active policies
* Insurance & license years

---

## 💡 Business Impact

* Enables **risk-adjusted pricing** and early flagging of high-risk policyholders
* Shows **policy behavior > vehicle features** in predicting risk
* Balances **model accuracy with interpretability**, critical for insurance and fintech use cases

---

## 🛠 Tech Stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `XGBoost` · `SHAP`
