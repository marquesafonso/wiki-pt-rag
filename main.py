import logging, os
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


@timer
def main():
    load_dotenv()
    logging.basicConfig(filename=f'./logs/main.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    VSS_INDEX_PATH = os.getenv("VSS_INDEX_PATH")
    FTS_INDEX_PATH = os.getenv("FTS_INDEX_PATH")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    ALPHA = float(os.getenv("ALPHA"))
    TOP_K = int(os.getenv("TOP_K"))
    query = "Qual é a ciência que estuda o espaço, os astros e as estrelas?"
    embedder = SentenceTransformer(EMBEDDING_MODEL, truncate_dim=256)
    query_vector = embedder.encode(query.strip()).tolist()
    dataset_info = check_hf_dataset_exists()
    if not dataset_info["created"]:
        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train").to_polars()
        chunker = SemanticChunker(
            embedding_model=EMBEDDING_MODEL,
            threshold=0.5,
            chunk_size=512,
            min_characters_per_sentence=50,
            min_sentences=1,
        )
        embeddings_ds = chunk_and_embed_dataset(ds, chunker, embedder)
        create_hf_dataset(Dataset.from_polars(embeddings_ds))
    
    dataset = dataset_info["dataset"]
    dataset:pl.DataFrame = dataset.with_columns((pl.col("id") + ":" + pl.col("chunk_number").cast(str)).alias("chunk_id")).drop(["id","chunk_number"])
    fts_res = get_fts_results(dataset=dataset, path=FTS_INDEX_PATH, query=query,top_k=TOP_K)
    vss_res = get_vss_results(dataset=dataset, path=VSS_INDEX_PATH, query_vector=query_vector, top_k=TOP_K)
    ftext_df = dataset.join(fts_res, on='chunk_id', how='right').fill_null(0.0)
    combined_res = ftext_df.join(vss_res, on='chunk_id', how='full').fill_null(0.0)
    max_fts, max_vss = combined_res["fts_score"].max(), combined_res["vss_score"].max()
    inf_sem_score, inf_lex_score = -1.0, 0.0
    df_norm_scores = combined_res.with_columns(
        ((pl.col("fts_score") - inf_lex_score) / (max_fts - inf_lex_score)).alias("normalized_fts_sim"),
        ((pl.col("vss_score") - inf_sem_score) / (max_vss - inf_sem_score)).alias("normalized_vss_sim")
    )
    df_final = df_norm_scores.with_columns(((pl.col("normalized_fts_sim") * ALPHA + pl.col("normalized_vss_sim") * (1 - ALPHA))).alias("convex_score")).sort("convex_score", descending=True)
    print(df_final.head(10))


if __name__ == "__main__":
    main()