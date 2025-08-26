# config.py
import os
from dotenv import load_dotenv
# File Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data/train_dataset.parquet')
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models/readmission_model.joblib')

# Model Parameters
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
load_dotenv()
API_KEY = os.getenv("API_KEY")

