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
    
    query = "Qual é a ciência que estuda o espaço, os astros e as estrelas?"
    alpha = 0.5
    top_k = 5000 # O valor do top_k parece alterar pouco o tempo de execução. duplicando de 2500 para 5000 até se passou de 40.9s para 32.9s. O valor parece estabilizar perto de 33s
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
    dataset:pl.DataFrame = dataset.with_columns((pl.col("id") + ":" + pl.col("chunk_number").cast(str)).alias("chunk_id")).drop(["id","chunk_number"])
    fts_res = get_fts_results(dataset=dataset, query=query,top_k=top_k)
    vss_res = get_vss_results(dataset=dataset, query_vector=query_vector, top_k=top_k)
    ftext_df = dataset.join(fts_res, on='chunk_id', how='right').fill_null(0.0)
    combined_res = ftext_df.join(vss_res, on='chunk_id', how='full').fill_null(0.0)
    max_fts, max_vss = combined_res["fts_score"].max(), combined_res["vss_score"].max()
    inf_sem_score, inf_lex_score = -1.0, 0.0
    df_norm_scores = combined_res.with_columns(
                                        ((pl.col("fts_score") - inf_lex_score) / (max_fts - inf_lex_score)).alias("normalized_fts_sim"),
                                        ((pl.col("vss_score") - inf_sem_score) / (max_vss - inf_sem_score)).alias("normalized_vss_sim"))
    df_final = df_norm_scores.with_columns(((pl.col("normalized_fts_sim") * alpha + pl.col("normalized_vss_sim") * (1 - alpha))).alias("convex_score")).sort("convex_score", descending=True)
    print(df_final.head(10))
    ### TODO: Módulo FTS funcional mas é necessário testar: 
        # abstrair variáveis nos queries de SQL para melhor configurabilidade;
        # avaliar melhor maneira de guardar índice fts em disco.
    ### TODO: usar vicinity para criar uma vector store.
        # ponderar diferentes backends / kwargs (estudar as opções)
        # utilizar vector store em disco
    ### TODO: Gradio app que permita receber queries e devolva as respostas
    ### TODO: procurar perceber se este método é passível de se escalar a partir de datasets existentes e definições do utilizador. 


if __name__ == "__main__":
    main()