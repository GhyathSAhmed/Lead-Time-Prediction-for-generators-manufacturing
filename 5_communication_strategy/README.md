# 📣 Lead-Time Prediction Demo App (Streamlit)

This folder contains the **communication & presentation interface** for the
project  
**“Lead-Time Prediction for Generators Manufacturing.”**

The purpose of this app is to **demonstrate the trained machine-learning model**
to
management, factory engineers, and the ERP/Odoo team before full integration.

This is **not** the production deployment — it is a **showcase interface** for early
adoption and stakeholder communication.

---

## 🎯 Purpose of the Streamlit Demo

- Provide a simple, interactive UI for entering generator order details  
- Apply the **trained PyCaret regression model** to predict:
  **→ Lead Time to Finish (days)**  
- Show results instantly for decision-makers
- Build confidence in the model before deploying it inside **Odoo ERP**

This demo is your **communication artefact** — a bridge between:

- Technical ML research  
- Business planning  
- ERP system integration  

---

## 🧠 High-Level Explanation of How the App Works

1- **Loads the trained ML model**  
   From the folder:

```bash
4_data_analysis/models/run_20251206_145317/
```

The model includes:

- Preprocessing pipeline (encoding, date features, imputers)
- Final regressor (e.g., Gradient Boosting)

2- **Builds an input form**  
The app reads the prepared dataset to extract unique values for dropdowns:

- engine model  
- alternator model  
- genset size  
- canopy size  
- controller  
- frequency  
- engine type / alternator type  

Users fill in **only the raw features** — preprocessing is handled by PyCaret.

3- **Generates a single-row DataFrame**  
Matching the exact structure used during training.

4- **Sends it to PyCaret for prediction**

```python
preds = predict_model(model, data=input_df)
```

- PyCaret automatically:

- Encodes categories

- Extracts date features

- Imputes values

- Applies transformers

- Runs the regression model

5- **Extracts the correct prediction column**

Through a custom helper function that handles:

- `prediction_label` (PyCaret 3.x)

- `Label` (older versions)

- Fallback to last numeric column

6- **Displays the predicted lead time**

```python
st.metric("Estimated Lead Time (days)", predicted_value)
```

7- **Optionally shows raw prediction output**
Useful for transparency, debugging, and stakeholder confidence.

## 🧩 System Diagram (High-Level)

```pgsql
                     ┌────────────────────────┐
                     │     Factory Manager     │
                     │     General Manager     │
                     │   System Admin (Odoo)   │
                     └─────────────┬───────────┘
                                   │
                                   │ Inputs (order details)
                                   ▼
                     ┌────────────────────────┐
                     │   Streamlit Demo App    │
                     │ (User Interface Layer)  │
                     └───────┬────────────────┘
                             │ Build single-row
                             │ DataFrame
                             ▼
                     ┌────────────────────────┐
                     │     PyCaret Pipeline    │
                     │ - encoders, imputers    │
                     │ - date feature extract  │
                     │ - regressor model       │
                     └───────────┬────────────┘
                                 │ Prediction
                                 ▼
                     ┌────────────────────────┐
                     │  Lead-Time Estimation   │
                     │  (days to finish unit)  │
                     └────────────────────────┘
```

## ▶️ How to Start the Demo App

This assumes you already setup your
Python version, virtual environment, and requirements installation.

From the project root directory:

1- Activate your virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```powershell
(.venv) PS C:\...\Lead-Time-Prediction-for-generators-manufacturing>
```

2- Run the Streamlit app

```powershell
streamlit run 5_communication_strategy/demo_lead_time_app.py
```

Streamlit will open your browser at:

```arduino
<http://localhost:8501>
```

3- Stop the app

Press:

```objectivec
CTRL + C
```

in the terminal.

## 📌 Notes for Deployment Planning

- This demo is not the production version

- After approval from management, the model will be integrated into Odoo ERP

- Odoo can call the model through:

  - A FastAPI microservice

  - A Python RPC call

  - A scheduled worker

- The Streamlit app acts as a visual communication layer only

## ✅ Intended Audience

MIT emerging talent team

Factory Manager

General Manager

System Administrator (Odoo)

Engineering & Planning teams
