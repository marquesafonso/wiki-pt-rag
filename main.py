import logging, os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import duckdb
from src.create_hf_dataset import create_hf_dataset, check_hf_dataset_exists
from src.chunk_and_embed_dataset import chunk_and_embed_dataset
from src.duckdb_utils import setup_db, create_fts_index, full_text_search
from src.timer import timer
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer

@timer
def main():
    load_dotenv()
    logging.basicConfig(filename=f'./logs/main.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    dataset_info = check_hf_dataset_exists()
    if not dataset_info["created"]:
        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train").to_polars()
        model_name = 'sentence-transformers/static-similarity-mrl-multilingual-v1'
        chunker = SemanticChunker(
            embedding_model=model_name,  # Default model
            threshold=0.5,                               # Similarity threshold (0-1) or (1-100) or "auto"
            chunk_size=512,                              # Maximum tokens per chunk
            min_characters_per_sentence=50,
            min_sentences=1,                             # Initial sentences per chunk
        )
        embedder = SentenceTransformer(model_name, truncate_dim=256)
        embeddings_ds = chunk_and_embed_dataset(ds, chunker, embedder)
        create_hf_dataset(Dataset.from_polars(embeddings_ds)) # 8500s or 2.3h
    
    dataset = dataset_info["dataset"]

    with duckdb.connect() as con:
        setup_db(con=con, dataset=dataset)
        create_fts_index(con=con) ## é necessário criar uma chave única que aja com id único (e.g., file_chunk_id) para o FTS do duckdb
        query = "Qual é o astro mais interessante do espaço?"
        res = full_text_search(con=con, query=query, top_k=10)
        print(res)

        ### TODO: Converter o código acima num módulo funcional e passá-lo para o folder src.
        ### TODO: usar vicinity para criar uma vector store em ficheiro.
        ### TODO: Gradio app que permita receber queries e devolva as respostas
        ### TODO: procurar perceber se este método é passível de se escalar a partir de datasets existentes e definições do utilizador. 


if __name__ == "__main__":
    main()