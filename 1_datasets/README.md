# Datasets

## Dataset 1: AQT Generators Manufacturing Data

### Overview

This dataset contains manufacturing and lead time data for power generators
produced at AQT Generators. The data was
**self-collected through direct observation** and integrated into the company's
workflow for the first time. This dataset captures the complete manufacturing
journey from order placement to shipping.

### Data Collection Method

- **Type**: Primary Data (Observational)
- **Collection**: Direct observation integrated into AQT Generators' workflow
- **Source**: Company manufacturing records and operational logs
- **Time Period**: Historical manufacturing orders

### Dataset Characteristics

- **Format**: Structured tabular data (.csv or .xlsx)
- **Purpose**: Predict lead times for generator manufacturing orders
- **Size**: [246X22]
- **Records**: [246]

### Data Structure & Columns

#### Order & Process Tracking

- `PO no.` - Purchase Order number (identifier)
- `PI no.` - Production/Internal order number
- `Generator Serial no.` - Unique generator identifier

#### Date Columns (Temporal Data)

- `Order Date` - Date order was placed
- `Receiving CKD Date` - Date CKD (Completely Knocked Down) components received
- `Finishing Date` - Date manufacturing/assembly completed
- `Shipping Date` - Date product shipped to customer

#### Lead Time Metrics (Target Variables)

- `LEAD TIME TO FINISH` - Days from order to manufacturing completion
- `LEAD TIME TO SHIP` - Days from order to shipping
- `LEAD TIME FROM CKD` - Days from CKD receipt to completion

#### Generator Specifications (Features)

- `Canopy Size` - Physical enclosure size of generator
- `Freq 50? 60?` - Frequency rating (50Hz or 60Hz)
- `Voltage` - Electrical voltage specification
- `Genset Size` - Generator power output capacity
- `Engine Type` - Type of engine used
- `Engine Model` - Specific engine model
- `Engine Serial no.` - Unique engine identifier
- `Alternator Type` - Type of alternator
- `Alternator Model` - Specific alternator model
- `Alternator S/No.` - Unique alternator identifier
- `Controller Model` - Control system model
- `Controller S/No.` - Unique controller identifier

### Data Quality & Limitations

#### Known Issues

- **Missing Date Values**: Some date fields have missing values that can be
obtained later through company records
- **Status**: These gaps will be addressed in the data preparation phase

#### Data Type Notes

- Dates should be converted to datetime format before analysis
- Categorical variables: Frequency, Engine Type, Alternator Type, etc.
- Numerical variables: Lead Time metrics
- Identifiers: Serial numbers, PO/PI numbers

### Relevance to Research Question

This dataset directly supports the lead time prediction model by capturing:

1. **Actual historical timelines** across the entire manufacturing process
2. **Product specifications** that may impact manufacturing duration
3. **Multi-stage lead times** (CKD receipt, finishing, shipping)

### Data Preparation Notes

- Date columns will require feature engineering (extracting year, month, day)
- Categorical variables will need encoding (one-hot encoding or label encoding)
- Serial numbers should be handled as identifiers (not used in model)
- Missing dates will be handled during data cleaning phase

### Usage

This is **raw observational data** and should not be modified directly. When
cleaning and processing, create new files with descriptive names
(e.g., `generator_data_cleaned.csv`, `generator_data_engineered.csv`).

---

**Last Updated:** December 2, 2025
