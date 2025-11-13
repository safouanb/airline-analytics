# Descriptive Analytics: Airline Passenger Satisfaction

A repository to perform comprehensive descriptive analytics on the airline passenger satisfaction dataset provided by the University of Amsterdam.

## Repository Structure

```
airline-analytics/
├── data/
│   └── raw/              # Raw datasets (train.csv, test.csv)
├── notebooks/            # Jupyter notebooks for analysis
│   ├── 01_business_understanding.md
│   └── descriptive_analytics.ipynb
├── reports/              # Project deliverables organized by type
│   ├── phase_reports/    # Individual CRISP-DM phase reports
│   ├── visualizations/   # Plots, charts, and figures
│   ├── models/           # Saved model files
│   └── final/            # Final report and presentation
├── results/              # Temporary output files
├── CRISP-DM_Status.md    # Project status tracker
└── README.md             # This file
```

## Getting Started

1. Navigate to the notebooks directory:
   ```bash
   cd notebooks
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `descriptive_analytics.ipynb` and run the cells.

## Data

The datasets are located in `data/raw/`:
- `train.csv` - Training dataset
- `test.csv` - Test dataset

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn jupyter
```
