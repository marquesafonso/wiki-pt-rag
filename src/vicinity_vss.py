from vicinity import Vicinity, Backend, Metric
import numpy as np
import polars as pl

def get_vss_results(dataset:pl.DataFrame, query_vector:list, top_k:int):
    vector_store = Vicinity.from_vectors_and_items(
        vectors=np.array(dataset["embeddings"].to_list()).astype(np.float32),
        items=dataset["chunk_id"].to_list(),
        backend_type=Backend.USEARCH,
        metric=Metric.COSINE,
        store_vectors=True,
    )
    schema = pl.Schema(schema={"chunk_id":pl.String(), "vss_score": pl.Float32})
    results = vector_store.query(query_vector, k=top_k)[0]
    results_df = pl.DataFrame(results, schema=schema, orient="row")
    return results_df