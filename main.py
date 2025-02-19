import logging
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import duckdb
import numpy as np
import polars as pl
from src.create_hf_dataset import create_hf_dataset, check_hf_dataset_exists
from src.chunk_and_embed_dataset import chunk_and_embed_dataset
from src.duckdb_utils import setup_db, create_fts_index, full_text_search
from src.timer import timer
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer
from vicinity import Vicinity, Backend, Metric


@timer
def main():
    load_dotenv()
    logging.basicConfig(filename=f'./logs/main.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    
    query = "Qual é a ciência que estuda o espaço?"
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
    dataset = dataset.with_columns((pl.col("id") + ":" + pl.col("chunk_number").cast(str)).alias("chunk_id")).drop(["id","chunk_number"])
    with duckdb.connect() as con:
        setup_db(con=con, dataset=dataset)
        create_fts_index(con=con)
        res = full_text_search(con=con, query=query, top_k=10)
        print(res)
    vicinity = Vicinity.from_vectors_and_items(
        vectors=np.array(dataset["embeddings"].to_list()).astype(np.float32),
        items=list(zip(dataset["chunk_id"].to_list(), dataset["chunk"].to_list())),
        backend_type=Backend.USEARCH,
        metric=Metric.COSINE
    )
    model_name = 'sentence-transformers/static-similarity-mrl-multilingual-v1'
    embedder = SentenceTransformer(model_name, truncate_dim=256)
    query_vector = embedder.encode(query.strip()).tolist()
    results = vicinity.query(query_vector, k=3)
    print(results)
        ### TODO: Módulo funcional mas é necessário simplificar: 
            # reordenar colunas do df; 
            # abstrair variáveis nos queries de SQL para melhor configurabilidade;
            # mover módulo para src
        ### TODO: usar vicinity para criar uma vector store.
            # Solução funcional mas bootstrapped, precisa de limpeza e abstração
            # necessário avaliar se passar zip para preservar os chunk_ids é a solução mais adequada. Provavelmente será melhor criar uma lista de dicts com uma list comprehension.
            # ponderar diferentes backends / kwargs (estudar as opções)
            # utilizar vector store em disco
        ### TODO: Fazer combinação linear entre os resultados consoante definição do utilizador.
        ### TODO: Gradio app que permita receber queries e devolva as respostas
        ### TODO: procurar perceber se este método é passível de se escalar a partir de datasets existentes e definições do utilizador. 


if __name__ == "__main__":
    main()