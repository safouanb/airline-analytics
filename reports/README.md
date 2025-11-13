# Reports Directory

This directory contains all project deliverables organized by CRISP-DM phase and output type.

## Structure

```
reports/
├── phase_reports/     # Individual phase reports (Business Understanding, Data Understanding, etc.)
├── visualizations/    # Saved plots, charts, and figures
├── models/            # Saved model files and model documentation
└── final/             # Final report and presentation materials
```

---

## Phase Reports (`phase_reports/`)

**Purpose:** Individual reports for each CRISP-DM phase

**Contents:**
- `01_business_understanding.md` - Business objectives and research questions
- `02_data_understanding.md` - Data exploration findings and insights
- `03_data_preparation.md` - Data cleaning and preprocessing documentation
- `04_modeling.md` - Model selection, training, and comparison
- `05_evaluation.md` - Model evaluation and performance metrics
- `06_deployment.md` - Final recommendations and deployment plan

---

## Visualizations (`visualizations/`)

**Purpose:** All plots, charts, and figures generated during analysis

**Organization:**
- `exploratory/` - EDA visualizations
- `modeling/` - Model-related plots (feature importance, ROC curves, etc.)
- `final/` - Visualizations for final report

**Formats:** PNG, PDF, SVG

---

## Models (`models/`)

**Purpose:** Saved model files and model documentation

**Contents:**
- Trained model files (`.pkl`, `.joblib`, etc.)
- `model_comparison.md` - Comparison of different models
- `model_performance.md` - Detailed performance metrics
- `feature_importance/` - Feature importance analysis

---

## Final (`final/`)

**Purpose:** Final project deliverables

**Contents:**
- `final_report.md` or `final_report.pdf` - Complete CRISP-DM report
- `presentation/` - Presentation slides (if applicable)
- `executive_summary.md` - Executive summary for stakeholders

---

## Usage

### Saving Phase Reports
```python
# Example: Save data understanding insights
with open('reports/phase_reports/02_data_understanding.md', 'w') as f:
    f.write("# Data Understanding Report\n\n...")
```

### Saving Visualizations
```python
# Example: Save a plot
plt.figure(figsize=(10, 6))
# ... create plot ...
plt.savefig('reports/visualizations/exploratory/satisfaction_distribution.png', 
            dpi=300, bbox_inches='tight')
```

### Saving Models
```python
# Example: Save a trained model
import joblib
joblib.dump(model, 'reports/models/logistic_regression_model.pkl')
```

---

## Best Practices

1. **Naming Convention:**
   - Use descriptive, consistent names
   - Include dates or version numbers if needed
   - Example: `satisfaction_by_class_2024.png`

2. **Documentation:**
   - Always include context in report files
   - Document assumptions and methodology
   - Explain key findings

3. **Version Control:**
   - Keep reports in markdown for easy version control
   - Use Git LFS for large model files if needed

4. **Organization:**
   - Keep related files together
   - Use subdirectories for complex projects
   - Maintain a clear folder structure


