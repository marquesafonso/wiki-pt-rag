from datasets.exceptions import DatasetNotFoundError
from vicinity import Vicinity, Backend, Metric
import numpy as np
import polars as pl

def create_vector_store(dataset:pl.DataFrame, repo_id:str, token:str):
    vector_store = Vicinity.from_vectors_and_items(
        vectors=np.array(dataset["embeddings"].to_list()).astype(np.float32),
        items=dataset["chunk_id"].to_list(),
        backend_type=Backend.USEARCH,
        metric=Metric.COSINE
    )
    vector_store.push_to_hub(repo_id=repo_id, token=token, private=True)
    return vector_store
    
def get_vss_results(dataset:pl.DataFrame, repo_id:str, token:str, query_vector:list, top_k:int):
    try:
        vector_store = Vicinity.load_from_hub(repo_id=repo_id, token=token)
    except DatasetNotFoundError:
        vector_store = create_vector_store(dataset=dataset, repo_id=repo_id, token=token)
    schema = pl.Schema(schema={"chunk_id":pl.String(), "vss_score": pl.Float32})
    results = vector_store.query(query_vector, k=top_k)[0]
    results_df = pl.DataFrame(results, schema=schema, orient="row")
    return results_df