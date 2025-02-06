import logging
from typing import List
import polars as pl
import pandas as pd
from datasets import load_dataset, Dataset
from src.create_hf_dataset import create_hf_dataset
from src.timer import timer
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer

def process_group(
    df: pl.DataFrame,
    chunker: SemanticChunker,
    embedder: SentenceTransformer
    ) -> pl.DataFrame:
    schema={"id": pl.String, "chunk": pl.String, "chunk_number": pl.Int64, "embeddings": pl.List(pl.Float64)}
    results = pl.DataFrame(schema=schema)
    for row in df.iter_rows(named=True):
        chunks = chunker.chunk(row["text"])
        chunk_df = pl.from_dicts([
            {
                "id": row["id"],
                "chunk": chunk.text.strip(),
                "chunk_number": i,
                "embeddings": embedder.encode(chunk.text.strip()).tolist()
            }
            for i, chunk in enumerate(chunks)
        ], schema=schema)
        results.vstack(chunk_df, in_place=True)
    final_df = df.join(results, on="id", how="left")
    final_df.drop_in_place("text")
    return final_df

@timer
def main():
    logging.basicConfig(filename=f'./logs/main.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
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
    embeddings_ds = process_group(ds, chunker, embedder)
    create_hf_dataset(Dataset.from_polars(embeddings_ds))
    # 8500s or 2.3h

if __name__ == "__main__":
    main()