# Airline Passenger Satisfaction – Analytics & Modeling

A repo for descriptive analytics and supervised models on the airline passenger satisfaction dataset (UvA).

## Repository Structure (current)
```
airline-analytics/
├── data/raw/               # train.csv, test.csv (only raw data kept)
├── notebooks/              # modeling scripts and notebooks
│   ├── preprocessing.py
│   ├── logistic_regression_model.py
│   ├── decision_tree_model.py
│   ├── random_forest_model.py
│   ├── xgboost_model.py    # uses sklearn GradientBoosting
│   ├── descriptive_analytics.ipynb
│   └── GradientTree.ipynb
├── figures/                # all generated plots (rf_, lr_, dt_, xgb_)
└── README.md
```

## How to Run the Models
From the repo root:
```bash
cd notebooks
python3 logistic_regression_model.py
python3 decision_tree_model.py
python3 random_forest_model.py
python3 xgboost_model.py     # gradient boosting (sklearn)
```
Outputs (PNGs) are written to `../figures/`.

## Data
Place raw files in `data/raw/`:
- `train.csv`
- `test.csv`

All preprocessing (dropping id/Unnamed:0, encoding, scaling (only done for logistic regression), imputation) is handled inside the pipelines; no preprocessed files are required.

## Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
Jupyter (optional) for the notebooks:
```bash
pip install jupyter
```
