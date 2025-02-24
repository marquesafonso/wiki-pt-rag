import os
import datasets
from dotenv import load_dotenv

def create_hf_dataset(dataset: datasets.Dataset):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("USER")
    TARGET_DATASET_NAME = os.getenv("TARGET_DATASET_NAME")
    TARGET_DATASET_SPLIT = os.getenv("TARGET_DATASET_SPLIT")
    dataset.push_to_hub(f"{HF_USER}/{TARGET_DATASET_NAME}", split=TARGET_DATASET_SPLIT, token=HF_TOKEN)

def check_hf_dataset_exists():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("USER")
    TARGET_DATASET_NAME = os.getenv("TARGET_DATASET_NAME")
    TARGET_DATASET_SPLIT = os.getenv("TARGET_DATASET_SPLIT")
    try:
        dataset = datasets.load_dataset(f"{HF_USER}/{TARGET_DATASET_NAME}", split=TARGET_DATASET_SPLIT, token=HF_TOKEN).to_polars()
        return {"created": True, "dataset": dataset}
    except Exception as e:
        return {"created": False, "exception": e}