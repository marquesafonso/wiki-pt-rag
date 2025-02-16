import polars as pl
from sentence_transformers import SentenceTransformer
from chonkie import SemanticChunker

def chunk_and_embed_dataset(
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