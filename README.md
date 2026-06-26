# Customer Churn Prediction

This repository contains code and resources for building a customer churn prediction system using machine learning. The project is organized as follows:

- **data/**: Raw and processed datasets.
- **notebooks/**: Jupyter notebooks for exploration and modeling.
- **src/**: Source code for data preprocessing, model training, prediction, and utilities.
- **models/**: Saved machine learning models (Pickle/Joblib).
- **api/**: Flask or FastAPI service for model inference.
- **streamlit_app/**: Optional Streamlit UI for interactive exploration.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API service:
   ```bash
   cd api
   python app.py
   ```
3. (Optional) Launch the Streamlit app:
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

## Project Structure

```
customer-churn-prediction/
├── data/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── models/
├── api/
│   └── app.py
├── streamlit_app/
├── requirements.txt
├── README.md
└── .gitignore
```
