import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# Load the new dataset
DATA_PATH = '../data/synthetic_churn_2817x21.csv'  # Adjust path if needed
MODEL_PATH = 'xgboost_model_20250602_140516.pkl'   # Update if using a different model

df = pd.read_csv(DATA_PATH)

# Load the trained XGBoost pipeline
model_file = Path(MODEL_PATH)
if not model_file.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

from models.xgboost import ProductionXGBoostPipeline
pipeline = ProductionXGBoostPipeline.load_model(str(model_file))

# Run predictions
preds, probs = pipeline.predict(df)

# If the target is not numeric, map to 0/1 for metrics
if 'Churn' in df.columns:
    y_true = df['Churn']
    if y_true.dtype == object or y_true.dtype.name == 'category':
        # Map 'No' to 0, 'Yes' to 1 (or use the model's target_encoder if available)
        if hasattr(pipeline, 'model_metadata') and 'target_encoder' in pipeline.model_metadata and pipeline.model_metadata['target_encoder'] is not None:
            y_true = pipeline.model_metadata['target_encoder'].transform(y_true)
        else:
            y_true = y_true.map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
else:
    y_true = None

# Ensure preds are also numeric for metrics
if y_true is not None:
    if hasattr(pipeline, 'model_metadata') and 'target_encoder' in pipeline.model_metadata and pipeline.model_metadata['target_encoder'] is not None:
        if hasattr(pipeline.model_metadata['target_encoder'], 'transform'):
            # If preds are string labels, convert to numeric using the same encoder
            if isinstance(preds[0], str):
                preds_numeric = pipeline.model_metadata['target_encoder'].transform(preds)
            else:
                preds_numeric = preds
        else:
            preds_numeric = preds
    else:
        # Fallback: map 'No'/'Yes' to 0/1 if needed
        if isinstance(preds[0], str):
            preds_numeric = pd.Series(preds).map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
        else:
            preds_numeric = preds
else:
    preds_numeric = preds

# Visualize metrics
f1 = f1_score(y_true, preds_numeric) if y_true is not None else None
acc = accuracy_score(y_true, preds_numeric) if y_true is not None else None
cm = confusion_matrix(y_true, preds_numeric) if y_true is not None else None
report = classification_report(y_true, preds_numeric, output_dict=True) if y_true is not None else None

if f1 is not None and acc is not None:
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

if cm is not None:
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if report is not None:
    print("Classification Report:")
    print(classification_report(y_true, preds_numeric))
    # Optional: visualize as heatmap
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8,4))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
    plt.title('Classification Report')
    plt.show()

# Show a sample of predictions
print("Sample predictions:")
print(pd.DataFrame({
    'prediction': preds,
    'probability': probs
}).head(20))

# Optionally, save predictions to CSV
output = df.copy()
output['prediction'] = preds
output['probability'] = probs
output.to_csv('../data/synthetic_churn_predictions.csv', index=False)
print('Predictions saved to ../data/synthetic_churn_predictions.csv')

def log_confusion_details(df, y_true, preds_numeric, id_col=None, label_map=None, output_path=None):
    """
    Logs which rows are TP, TN, FP, FN for binary classification.
    Optionally saves a CSV with the results.
    Args:
        df: Original dataframe
        y_true: Ground truth (numeric 0/1)
        preds_numeric: Model predictions (numeric 0/1)
        id_col: Optional column name to use as unique ID
        label_map: Optional dict to map 0/1 back to original labels
        output_path: Optional path to save the CSV
    Returns:
        DataFrame with confusion details
    """
    results = df.copy()
    results['y_true'] = y_true
    results['y_pred'] = preds_numeric
    if label_map:
        results['y_true_label'] = results['y_true'].map(label_map)
        results['y_pred_label'] = results['y_pred'].map(label_map)
    # Assign confusion type
    def confusion_type(row):
        if row['y_true'] == 1 and row['y_pred'] == 1:
            return 'TP'
        elif row['y_true'] == 0 and row['y_pred'] == 0:
            return 'TN'
        elif row['y_true'] == 0 and row['y_pred'] == 1:
            return 'FP'
        elif row['y_true'] == 1 and row['y_pred'] == 0:
            return 'FN'
        else:
            return 'Unknown'
    results['confusion_type'] = results.apply(confusion_type, axis=1)
    # Print summary
    print(results['confusion_type'].value_counts())
    # Optionally save
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Confusion details saved to {output_path}")
    return results

# After metrics and visualizations, log confusion details
if y_true is not None:
    label_map = None
    if hasattr(pipeline, 'model_metadata') and 'target_encoder' in pipeline.model_metadata and pipeline.model_metadata['target_encoder'] is not None:
        label_map = {i: l for i, l in enumerate(pipeline.model_metadata['target_encoder'].classes_)}
    else:
        label_map = {0: 'No', 1: 'Yes'}
    confusion_details = log_confusion_details(df, y_true, preds_numeric, label_map=label_map, output_path='../data/synthetic_churn_confusion_details.csv')
