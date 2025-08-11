import pandas as pd
from sklearn.metrics import roc_auc_score
import os

def perform_fairness_analysis(mimic_dir, predictions_path):
    """
    Loads predictions and demographic data to analyze model performance
    across different subgroups.
    """
    print("--- Starting Fairness and Bias Analysis ---")

    # 1. Load required files
    try:
        preds_df = pd.read_parquet(predictions_path)
        admissions_df = pd.read_csv(os.path.join(mimic_dir, 'hosp', 'admissions.csv'))
        patients_df = pd.read_csv(os.path.join(mimic_dir, 'hosp', 'patients.csv'))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure MIMIC-IV CSVs are in the correct directory and test_predictions.parquet exists.")
        return

    # 2. Merge dataframes to get demographics for each prediction
    # Merge admissions to get subject_id and demographics
    analysis_df = pd.merge(preds_df, admissions_df[['hadm_id', 'subject_id', 'insurance', 'language', 'ethnicity']], on='hadm_id')
    # Merge patients to get gender
    analysis_df = pd.merge(analysis_df, patients_df[['subject_id', 'gender']], on='subject_id')

    # Define the demographic columns we want to analyze
    subgroup_columns = ['ethnicity', 'insurance', 'language', 'gender']

    # 3. Loop through each column, group by subgroup, and calculate metrics
    for column in subgroup_columns:
        print(f"\n--- Performance by {column.upper()} ---")
        
        # Group by the unique values in the column
        grouped = analysis_df.groupby(column)
        
        results = []
        for name, group in grouped:
            # Skip groups that are too small or have only one class
            if len(group) < 20 or len(group['true_label'].unique()) < 2:
                continue
            
            auc_score = roc_auc_score(group['true_label'], group['predicted_prob'])
            results.append({'Subgroup': name, 'AUC': auc_score, 'Count': len(group)})
        
        # Display results in a table
        if results:
            result_df = pd.DataFrame(results).sort_values(by='AUC', ascending=True)
            print(result_df.to_string(index=False))

if __name__ == "__main__":
    # IMPORTANT: Update this path to the root directory of your MIMIC-IV dataset
    MIMIC_DATA_DIR = "/path/to/your/mimic-iv-2.2/"
    PREDICTIONS_FILE = "test_predictions.parquet"
    
    perform_fairness_analysis(MIMIC_DATA_DIR, PREDICTIONS_FILE)