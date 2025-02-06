import os, argparse
from dotenv import load_dotenv

def create_hf_dataset(dataset):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USER = os.getenv("USER")
    DATASET_NAME = os.getenv("DATASET_NAME")
    dataset.push_to_hub(f"{HF_USER}/{DATASET_NAME}", token=HF_TOKEN, private=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creating a HF dataset')
    parser.add_argument('--config_file', required=True, type=str,
                        help='DB configs file path')
    args = parser.parse_args()
    create_hf_dataset(config_file=args.config_file)