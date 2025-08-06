# Predicting Hospital Readmission using Clinical Notes

A machine learning project to predict 30-day hospital readmission risk, helping hospitals identify high-risk patients for targeted interventions. This project leverages NLP techniques on unstructured EHR data.

## 1. Problem Statement

Unplanned hospital readmissions are costly for healthcare systems and often indicate poor patient outcomes. By proactively identifying patients at high risk of readmission, hospitals can implement post-discharge plans to improve care and reduce costs. This project builds a binary classification model to predict which patients will be readmitted within 30 days of discharge.

## 2. Dataset

This project uses the publicly available **MIMIC-III** dataset. Specifically, it utilizes the `NOTEEVENTS.csv` file containing de-identified clinical notes.

* **Preprocessing:** The text data was cleaned by removing stop words, punctuation, and converting to lowercase.
* **Feature Engineering:** TF-IDF was used to vectorize the clinical notes into a numerical format suitable for machine learning.

## 3. Methodology

A **Logistic Regression** model was chosen as a strong, interpretable baseline. The pipeline consists of:
1.  Loading clinical notes.
2.  Text cleaning and preprocessing.
3.  Vectorization using `TfidfVectorizer`.
4.  Training the Logistic Regression classifier.

## 4. Results

The model's performance was evaluated on a held-out test set. Given the class imbalance inherent in readmission prediction, metrics beyond accuracy are crucial.

| Metric         | Score |
| -------------- | ----- |
| **AUC-ROC** | 0.78  |
| **F1-Score** | 0.65  |
| **Precision** | 0.72  |
| **Recall** | 0.59  |


## 5. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/shivaheidari/Readmission.git](https://github.com/shivaheidari/Readmission.git)
    cd Readmission
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Open and run the `Readmission_Analysis.ipynb` notebook.
6. Future Work
## 6. Future Work
* **Advanced Models:** Advancing model with multi-modal predictions models.
* **Deployment:** Containerize the prediction pipeline with **Docker** and deploy it as a REST API using **Flask/FastAPI**.
* **Feature Engineering:** Incorporate structured data (lab values, demographics) alongside the text data.