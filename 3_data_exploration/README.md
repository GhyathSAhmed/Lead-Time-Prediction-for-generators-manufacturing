# Data Exploration

This folder contains Python script used to **explore and
understand** the prepared dataset. Exploration at this stage is meant to be
visual, descriptive, and curiosity-driven — without running any inferential
statistics or machine learning models.

All scripts in this folder follow:

1. **Read only from prepared datasets in `1_datasets/`.**  
   These dataset are the output of the data-preparation stage.

2. **Explore the dataset without modifying it.**  
   Exploration includes:
   - Visualizations (histograms, bar charts, heatmaps, etc.)
   - Descriptive statistics
   - Structural inspection (column types, missing values, distributions)

3. **Never modify or overwrite datasets in `1_datasets/`.**  
   All outputs produced during exploration must be saved in this folder (or a
   subfolder of it), ensuring reproducibility.

4. **Avoid any inferential statistics or machine learning during exploration.**

---

## Scripts in This Folder

### `explore_lead_time_dataset.py`

**Purpose:**  
Perform a full exploratory data analysis (EDA) of the cleaned generator dataset
`ELO2_lead_time_prepared.csv`.

**Reads from:**  

- `1_datasets/ELO2_lead_time_prepared.csv`

**Does NOT modify this file.**

**Creates:**  
A fully organized set of exploration outputs under:

```bash
data_exploration_outputs/
    plots/
    summaries/
```

---

## What the Script Does

The EDA script includes the following exploration components:

### **1. Dataset Structure & Metadata**

- Saves dataset shape, column types, and `DataFrame.info()` to:
  - `summaries/dataset_info.txt`

### **2. Descriptive Statistics**

- Numeric column descriptions → `numeric_describe.csv`
- Missing value counts (cleaned and realistic) → `missing_values.csv`
- Top value counts for categorical columns → `categorical_value_counts.txt`

### **3. Visual Explorations**

All plots are saved under `data_exploration_outputs/plots/`:

- **Histograms**  
  - `hist_target_lead_time_to_finish_days.png`  
  - `hist_lead_time_to_ship.png`  
  - `hist_lead_time_from_ckd.png`

- **Categorical Distributions**  
  - `bar_engine_type.png`  
  - `bar_engine_model_top20.png`  
  - `bar_alternator_type.png`  
  - `bar_genset_size.png`  
  - `bar_voltage.png`

- **Correlation Heatmaps**  
  - `corr_heatmap_numeric.png` (numeric-only)  
  - `corr_heatmap_full.png` (all columns, with categorical encoded)

### **4. Correlation Matrices**

Saved under:

- `summaries/correlation_matrix_numeric.csv`
- `summaries/correlation_matrix_full.csv`

---

## How to Run

From the **project root directory**, run:

```bash
python 2_data_exploration/explore_lead_time_dataset.py
```

The script will automatically create the folder:

`
2_data_exploration/data_exploration_outputs/
`

and fill it with:

Plots (PNG images)

Summary tables and text files (CSV/TXT)
