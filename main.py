import logging
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import polars as pl
from src.create_hf_dataset import create_hf_dataset, check_hf_dataset_exists
from src.chunk_and_embed_dataset import chunk_and_embed_dataset
from src.duckdb_utils import get_fts_results
from src.vicinity_vss import get_vss_results
from src.timer import timer
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer
from vicinity import Vicinity, Backend, Metric
import numpy as np


@timer
def main():
    load_dotenv()
    logging.basicConfig(filename=f'./logs/main.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    
    query = "Qual é a ciência que estuda o espaço?"
    top_k = 10
    model_name = 'sentence-transformers/static-similarity-mrl-multilingual-v1'
    embedder = SentenceTransformer(model_name, truncate_dim=256)
    query_vector = embedder.encode(query.strip()).tolist()
    dataset_info = check_hf_dataset_exists()
    if not dataset_info["created"]:
        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train").to_polars()
        chunker = SemanticChunker(
            embedding_model=model_name,  # Default model
            threshold=0.5,                               # Similarity threshold (0-1) or (1-100) or "auto"
            chunk_size=512,                              # Maximum tokens per chunk
            min_characters_per_sentence=50,
            min_sentences=1,                             # Initial sentences per chunk
        )
        embeddings_ds = chunk_and_embed_dataset(ds, chunker, embedder)
        create_hf_dataset(Dataset.from_polars(embeddings_ds)) # 8500s or 2.3h
    
    dataset = dataset_info["dataset"]
    dataset = dataset.with_columns((pl.col("id") + ":" + pl.col("chunk_number").cast(str)).alias("chunk_id")).drop(["id","chunk_number"])
    fts_res = get_fts_results(dataset=dataset, query=query, top_k=top_k)
    vss_res = get_vss_results(dataset=dataset, query_vector=query_vector, top_k=top_k)
    combined_res = fts_res.join(vss_res, on='chunk_id', how='outer')
    print(combined_res)
    ### TODO: Módulo funcional mas é necessário simplificar: 
        # reordenar colunas do df; 
        # abstrair variáveis nos queries de SQL para melhor configurabilidade;
    ### TODO: usar vicinity para criar uma vector store.
        # Solução funcional mas bootstrapped, precisa de limpeza e abstração
        # ponderar diferentes backends / kwargs (estudar as opções)
        # utilizar vector store em disco
    ### TODO: Fazer combinação linear entre os resultados consoante definição do utilizador.
    ### TODO: Gradio app que permita receber queries e devolva as respostas
    ### TODO: procurar perceber se este método é passível de se escalar a partir de datasets existentes e definições do utilizador. 


if __name__ == "__main__":
    main()