# 🛍️ Online Retail Customer Analytics & Unsupervised ML Segmentation Platform

An executive-grade intelligence platform built with **Python**, **Streamlit**, **Plotly**, and **Scikit-Learn** designed to analyze macroeconomic retail trends, product revenue concentration, return impacts, and customer lifetime value via **RFM Analysis** and **Unsupervised Machine Learning Clustering**.

---

## To-Do
-[x] **Interactive 3D Globe**:
   - Orthographic 3D projection rendering global order densities and revenue contribution across international markets with cast country borders.
-[x] **Dynamic Market Scope & Country Filtering**:
   - Instant segmentation filters allowing deep-dives into **All Global Markets**, **United Kingdom Only**, or **Rest of World (International)** alongside specific country drill-downs.
-[x] **Exploratory Data Analysis (EDA)**:
   - Monthly revenue & Q4 holiday surges, 80/20 Pareto product concentration curves, hourly traffic/cadence heatmaps, and Average Order Value (AOV) distributions.
-[x] **Returns & Cancellation Impact Analysis**:
   - Dedicated tracking of cancelled transactions and quantified revenue loss per SKU.
-[x] **RFM Behavioral Segmentation & Business Matrix**:
   - Automated Recency, Frequency, and Monetary quantile scoring mapped to actionable business archetypes (*Champions, Loyal Customers, At-Risk / Churn, New Inflow, Potential Loyalists*).
-[x] **Multi-Model Unsupervised Machine Learning**:
   - **K-Means Clustering**: Evaluated with WCSS (Inertia) Elbow curves and Silhouette score optimization.
   - **Agglomerative Hierarchical Clustering**: Ward linkage with interactive dendrogram visualization.
   - **DBSCAN**: Density-based clustering with $k$-distance knee graphs and noise point handling (`-1`).

---

## 📂 Project Directory Structure

```text
CustSeg/
│
├── data/
│   └── OnlineRetail.csv          # Raw transaction dataset
├── assets/
│   └── style.css                 # Custom CSS styles (Inter & Fira Code fonts)
├── src/
│   ├── __init__.py
│   ├── transform.py              # Data cleaning, deduplication, and feature engineering
│   ├── eda.py                    # Exploratory Data Analysis & Plotly chart generators
│   ├── segmenter.py              # RFM scoring, business matrix mapping, and 3D scatters
│   └── models.py                 # Clustering models (K-Means, Hierarchical, DBSCAN) & diagnostics
│
├── main.py                       # Main Streamlit executive dashboard application
└── README.md                     # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have **Python 3.12+** and `pip` (or `uv`) installed.

### 1. Clone the Repository
Navigate to your project root folder:
```bash
git clone https://github.com/S9npai/SegRFM
cd CustSeg
```

### 2. Project setup
Create a virtual environment and install dependencies:
```bash
pip install requirements.txt
```

or with **uv**:

```bash
uv sync
```

### 3. Add Dataset
Ensure your dataset `OnlineRetail.csv` is placed inside the `data/` directory:
```text
data/OnlineRetail.csv
```

---

## 🖥️ Running the Application

Launch the Streamlit web application locally:
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501` to access the interactive dashboard.


