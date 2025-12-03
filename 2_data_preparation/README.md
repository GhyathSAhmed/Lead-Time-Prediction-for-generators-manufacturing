# Data Preparation

This folder contains Python script used to clean and prepare dataset before
they are used for analysis or modeling. All scripts in this folder:

1. Read datasets from `1_datasets/`
2. Clean, reformat, or otherwise process the datasets
3. Save processed versions back into `1_datasets/` using a **new filename**

**Important:**  
Do **not** modify or overwrite any existing dataset in `1_datasets/`.  
Always write cleaned data to a new file so the entire workflow can be reproduced

---

## Scripts in This Folder

### `prepare_lead_time_dataset.py`

**Purpose:**  
Prepare the generator production dataset for **lead-time prediction**.

**Reads:**  

- `1_datasets/ELO2_Raw_Data.csv`

**Creates:**  

- `1_datasets/ELO2_lead_time_prepared.csv`  
  (Fully cleaned and ready for modeling)

---

## Processing Steps Performed

The script performs the following transformations:

### 1. Load Raw Dataset

Reads the original CSV file from the `1_datasets/` directory without modifying it.

### 2. Standardize Column Names

- Converts all column names to lowercase `snake_case`
- Removes special characters and repeated underscores  
This ensures consistent naming across the project.

### 3. Remove Serial-Number / ID Columns (Leakage Prevention)

The script automatically detects and removes:

- Serial number fields  
- Unit numbers  
- Columns containing “id”, “unit”, “serial”, “no.”, etc.  
- Columns with extremely high uniqueness (≥ 98%)

These columns can cause **data leakage** and are removed before modeling.

### 4. Parse Date Columns with Mixed Formats

The dataset contains multiple date formats:

- `order_date` → **DD/MM/YYYY**
- `receiving_ckd_date`, `finishing_date`, `shipping_date` → **MM/DD/YYYY**

The script parses each column using the correct format and converts them into
clean, consistent `datetime` objects.

### 5. Create the Target Column

From the raw column **`LEAD TIME TO FINISH`**, the script creates:

- `target_lead_time_to_finish_days` (numeric, cleaned)

This is the prediction target used in modeling.

### 6. Drop Rows with Missing Target

Any row that does not contain the required target value is removed.

### 7. Sort by Order Date

For cleaner time-based analysis, the dataset is sorted chronologically.

### 8. Save the Processed Dataset

The final cleaned DataFrame is saved as:
