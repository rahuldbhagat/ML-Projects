# K-Prototypes Clustering — Intuitive Summary

---

## 🎯 Problem Statement

> **We have sales data spread across 100+ countries. Each country has both numbers (Sales, Profit) and categories (Region, Segment, Product Category). We want to automatically group countries that behave similarly — so we can make smarter, segment-aware business decisions instead of treating every market the same.**

The challenge: standard clustering (K-Means) only works with numbers. Throwing away the categorical columns loses real signal. One-hot encoding them inflates the feature space and distorts distances. **K-Prototypes is the right tool — it was designed for exactly this mixed-data problem.**

---

## 🧠 What Are We Trying to Achieve?

| Question | Answer we're building toward |
|---|---|
| Which countries behave similarly? | Cluster membership tells you |
| What makes them similar? | Cluster profile (Step 9) tells you |
| How many groups make sense? | Elbow method (Step 7) tells you |
| Can I see it visually? | PCA scatter (Step 10) shows you |
| What should I do with each group? | Business interpretation (Step 12) guides you |

The end deliverable is a **country → cluster mapping** with a clear profile of what each cluster represents commercially.

---

## 🔬 The Algorithm in Plain English

### K-Means (what everyone knows)
Groups data by minimising the average distance to a cluster centre. Works only on numbers. Centroid = mean.

### K-Modes (less known)
Like K-Means but for categories only. Distance = number of mismatches (Hamming). Centroid = most frequent value (mode).

### K-Prototypes (what we use)
Combines both:

```
Total Distance = Numerical Distance (Euclidean) + γ × Categorical Distance (Hamming)
```

`γ (gamma)` is the weight that balances how much the categorical columns influence clustering versus the numerical ones. We let the library auto-select it.

---

## 📓 Cell-by-Cell Walkthrough

---

### 🔧 Step 1 — Install Dependencies
**What:** Installs `kmodes` (the K-Prototypes library), `scikit-learn`, and plotting libraries.

**Why here:** Jupyter environments don't always have `kmodes` pre-installed. Running this once sets up the environment.

---

### 📦 Step 2 — Imports
**What:** Loads all Python libraries into memory.

**Why it matters:** We import `KPrototypes` from `kmodes`, `StandardScaler` + `PCA` from `sklearn`, and `pandas`/`numpy` for data handling. Each serves a distinct role later.

---

### 📥 Step 3 — Load Dataset
**What:** Tries to download the **Global Superstore 2016** dataset (51K orders, 147 countries) from a GitHub mirror. Falls back to a local CSV, and if that also fails, generates a realistic synthetic dataset so the notebook always runs.

**Why this dataset:** It has both numerical columns (Sales, Profit, Quantity, Discount) and categorical columns (Region, Segment, Category) across many real countries — exactly the mixed-data scenario K-Prototypes is built for.

**Key insight:** The raw data is at **order line level** — one row per product per order. We cannot cluster on this directly. We need to roll it up.

---

### 🔍 Step 4 — EDA (Exploratory Data Analysis)
**What:** Plots distributions of Sales, Profit, Quantity, Discount. Shows top 20 countries by total sales. Checks for nulls.

**Why before modelling:** 
- Sales is heavily right-skewed → confirms we must scale before clustering
- Helps spot data quality issues before they corrupt results
- Gives intuition for what the clusters might represent

Think of it as reading the room before giving a presentation.

---

### 🗺️ Step 5 — Aggregate to Country Level
**What:** Collapses 51K rows into **one row per country** using:

| Column | How | Why |
|---|---|---|
| Sales, Profit, Quantity | Sum | Total contribution of that country |
| Discount | Mean | Typical discounting behaviour |
| Region, Segment, Category | Mode (most frequent) | Dominant character of that market |
| Profit Margin | Derived (Profit/Sales) | Profitability efficiency metric |
| Avg Order Value | Derived (Sales/Orders) | Ticket size per transaction |

**Why this matters:** K-Prototypes clusters *entities* — in our case, countries. Each country must be one row with one set of features describing it.

---

### ⚙️ Step 6 — Preprocess for K-Prototypes
**What:** 
- Applies `StandardScaler` to numerical columns (zero mean, unit variance)
- Keeps categorical columns as plain strings
- Tracks which column indices are categorical (required by the library)

**Why scaling is critical:** Without it, `Total_Sales` (values in thousands) would completely overpower `Discount` (values 0–0.5), making the numerical distance meaningless.

**Why not one-hot encode categoricals:** That would turn 3 categorical columns into ~15 binary columns, inflate the space, and make Euclidean distance behave badly. K-Prototypes avoids this entirely.

---

### 📉 Step 7 — Elbow Method (Find Optimal K)
**What:** Runs K-Prototypes for K = 2, 3, 4 ... up to `min(10, n_countries//2)`. Records the total cost (inertia) at each K. Auto-detects the elbow using the **second derivative** (the point where cost stops dropping steeply).

**Why not just pick K=3:** There's no free lunch. The elbow method lets the data tell you how many natural groupings exist. Picking K arbitrarily risks either over-splitting (noise clusters) or under-splitting (losing real distinctions).

**The three fixes applied here vs the original:**
- `init='Huang'` instead of `Cao` — more robust with small datasets and low-cardinality categoricals
- K capped dynamically at `n_countries // 2` — prevents K from approaching sample size
- `try/except` per seed — skips failing K values cleanly instead of crashing

---

### 🏗️ Step 8 — Fit Final Model
**What:** Fits K-Prototypes with the optimal K discovered above. Runs 5 initialisations (`n_init=5`) and keeps the best result (lowest cost).

**Why multiple initialisations:** K-Prototypes is sensitive to random starting centroids. Running it 5 times and keeping the best avoids getting stuck in a poor local minimum.

---

### 🔥 Step 9 — Cluster Profiles (Heatmap)
**What:** Computes the mean of each numerical feature per cluster, and the dominant categorical value per cluster. Displays as a colour-coded heatmap.

**Why this is the most important step:** The cluster number (0, 1, 2...) is meaningless on its own. The profile is what gives it business meaning. This is where you read: *"Cluster 2 has high sales, high margin, Technology dominant, Corporate segment — this is our premium B2B market."*

---

### 🔵 Step 10 — PCA Visualisation
**What:** Uses PCA (Principal Component Analysis) to compress 6 numerical features into 2 dimensions, then plots countries as labelled dots coloured by cluster.

**Why PCA and not direct plotting:** We have 9 features — you can't plot 9 dimensions. PCA finds the 2 directions of maximum variance and projects everything onto them. It's a lossy compression, but good enough to see cluster separation.

**Honest caveat:** PCA only uses numerical features. Categorical features are not visible in this plot — which is why the heatmap in Step 9 is the more complete picture.

---

### 🗂️ Step 11 — Countries per Cluster
**What:** Lists every country in each cluster alongside the cluster's dominant region, segment, category, average sales, and margin. Shows boxplots of Sales and Profit Margin per cluster. Shows stacked bar charts for categorical mix.

**Why multiple views:** The scatter shows separation, the boxplot shows spread and outliers, the bar chart shows categorical composition. Together they give a 360° picture of each cluster.

---

### 💡 Step 12 — Business Interpretation
**What:** A reference table of typical cluster archetypes and what actions they suggest.

**Why it's the point of the whole exercise:** Clustering is not an end — it's a lens. The output is only useful when translated into a decision:
- High Sales + High Margin → protect and invest
- High Volume + Low Margin → pricing strategy review
- Low Sales → evaluate market entry ROI
- High Discount → discount discipline needed

The final CSV (`country_clusters.csv`) is the deliverable that feeds into a BI tool or strategy deck.

---

## 🔄 Data Flow at a Glance

```
Raw Orders (51K rows)
        │
        ▼
   Aggregate by Country
   (one row per country, mixed types)
        │
        ▼
   Scale Numericals + Keep Categoricals as Strings
        │
        ▼
   Elbow Method → Find Optimal K
        │
        ▼
   K-Prototypes.fit() → Cluster Labels
        │
        ├──▶ Profile Heatmap   (what each cluster looks like)
        ├──▶ PCA Scatter       (where clusters sit in space)
        ├──▶ Country Listings  (who is in each cluster)
        └──▶ country_clusters.csv (exportable deliverable)
```

---

## ⚠️ Honest Limitations

| Limitation | Impact |
|---|---|
| K-Prototypes is non-deterministic | Different runs can give slightly different clusters — use `random_state` for reproducibility |
| PCA plot is approximate | Only numerical features are projected — categorical influence is invisible here |
| Elbow auto-detection is heuristic | Always sanity-check the elbow plot with your eye — the second derivative can be fooled by noisy cost curves |
| Gamma is auto-selected | The balance between numerical and categorical distance is not tuned — for production use, treat gamma as a hyperparameter |
| Single aggregation per country | A country's "dominant segment" might mask a 51/49 split — cluster profiles are averages, not absolutes |
