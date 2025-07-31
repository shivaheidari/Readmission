# Readmission
## AI for Clinical Decision Support: Predicting 30-Day Hospital Readmission
This project develops a state-of-the-art machine learning model to predict the **risk of 30-day hospital readmission** using unstructured clinical notes from the MIMIC-IV dataset. The core of this work is a fine-tuned Bio_ClinicalBERT model, enhanced with Explainable AI (XAI) techniques using SHAP to provide transparent, interpretable predictions for clinical decision support.


### Key Features
- State-of-the-Art NLP Model: Utilizes a fine-tuned transformer model (Bio_ClinicalBERT) pre-trained on clinical text for superior contextual understanding.

- Explainable AI (XAI): Integrates the SHAP library to provide clear, word-level explanations for each prediction, answering why a patient is flagged as high-risk.

- High-Performance Workflow: Engineered to run on High-Performance Computing (HPC) clusters using the Slurm scheduler for efficient, large-scale training.

- Reproducible Environment: Uses a Conda environment.yml file to ensure a fully reproducible setup.

### Tech Stack
- Core Libraries: PyTorch, Hugging Face Transformers, SHAP, Scikit-learn, Pandas

- Environment: Conda

- Compute: HPC Cluster with Slurm Workload Manager & NVIDIA GPUs
### License
This project is licensed under the MIT and Hotchkiss Brain Institute License. See the LICENSE file for details.
### Acknowledgments
This project uses the MIMIC-IV dataset, which is made available by the MIT Laboratory for Computational Physiology.