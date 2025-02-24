import os, argparse
import datasets
from dotenv import load_dotenv

def create_hf_dataset(dataset: datasets.Dataset):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("USER")
    DATASET_NAME = os.getenv("DATASET_NAME")
    dataset.push_to_hub(f"{HF_USER}/{DATASET_NAME}", token=HF_TOKEN)

def check_hf_dataset_exists():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("USER")
    DATASET_NAME = os.getenv("DATASET_NAME")
    try:
        dataset = datasets.load_dataset(f"{HF_USER}/{DATASET_NAME}", split=f"train", token=HF_TOKEN).to_polars()
        return {"created": True, "dataset": dataset}
    except Exception as e:
        return {"created": False, "exception": e}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creating a HF dataset')
    parser.add_argument('--config_file', required=True, type=str,
                        help='DB configs file path')
    args = parser.parse_args()
    create_hf_dataset(config_file=args.config_file)