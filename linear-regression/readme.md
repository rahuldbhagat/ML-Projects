# 📊 Linear Regression — Key Concepts & Interpretation

This repository contains Jupyter notebooks and datasets for exploring **Linear Regression** concepts using Python (`statsmodels` and `scikit-learn`).  

As part of our learning sessions, we’re maintaining these notebooks collaboratively so that everyone can refer back to a **centralized source**. If you miss a session or want to revise key topics, this repo will serve as a single reference point.

---

## 🧠 Introduction

**Linear Regression** is one of the most fundamental techniques in statistics and machine learning.  
It models the relationship between a **target variable (Y)** and one or more **predictor variables (X)** by fitting a straight line (or plane, in multiple dimensions):

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \varepsilon
\]

Where:
- \( \beta_0 \) is the **intercept**  
- \( \beta_1 \ldots \beta_p \) are the **coefficients** for each predictor  
- \( \varepsilon \) is the error term (residual)

Interpreting regression output correctly is crucial for understanding relationships, validating assumptions, and building reliable predictive models.  

The table below summarizes the **key statistical terms** that commonly appear in regression summaries (e.g., `statsmodels.summary()` output) — along with their **intuition** and **purpose** 👇

---

## 📋 Key Regression Terms — Intuition & Purpose

| Term | Intuition (Simple English) | Purpose / Why It Matters |
|------|-----------------------------|----------------------------|
| **Intercept (β₀)** | Predicted value of Y when all X’s are 0. Baseline level. | Anchors the regression line/plane. |
| **Coefficient (β)** | How much Y changes for a 1-unit change in X, holding others constant. | Measures strength & direction of predictor impact. |
| **Standard Error** | How much the estimated coefficient might vary across samples. | Indicates estimate reliability; used in hypothesis testing. |
| **t-statistic** | Coefficient ÷ Standard Error. Shows how far the estimate is from 0. | Tests if each predictor’s effect is statistically significant. |
| **p-value** | Probability of observing the t-stat if the true coefficient is 0. Small p ⇒ significant. | Decides if predictor should be kept (hypothesis test). |
| **Confidence Interval [0.025, 0.975]** | Range within which the true coefficient likely lies with 95% confidence. | Captures uncertainty; if 0 not in interval ⇒ significant effect. |
| **R-squared (R²)** | % of variation in Y explained by the model. Higher = better fit. | Indicates overall model fit (but can be inflated by more predictors). |
| **Adjusted R-squared** | R² adjusted for number of predictors. Penalizes unnecessary variables. | More reliable for comparing models with different numbers of predictors. |
| **F-statistic** | Tests if at least one predictor is useful (non-zero). | Global test for overall model significance. |
| **F-test p-value** | Probability that all coefficients are zero (no relationship). | Checks if the model as a whole is meaningful. |
| **Residuals (ε)** | Difference between actual and predicted Y. | Used to validate model assumptions. |
| **Mean Squared Error (MSE)** | Average of squared residuals. Lower = better fit. | Common prediction error metric. |
| **Degrees of Freedom (DF)** | Roughly, number of observations minus estimated parameters. | Used in t/F distributions for testing. |
| **OLS (Ordinary Least Squares)** | Method that minimizes the sum of squared residuals. | Core algorithm behind linear regression. |

---

## 📌 How to Use This Repo

1. Clone the repo using SSH or HTTPS  
   ```bash
   git clone git@github.com:<your-username>/<repo-name>.git
