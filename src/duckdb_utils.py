import duckdb
import polars as pl
from datasets import load_dataset

def setup_db(con:duckdb.DuckDBPyConnection, dataset: pl.DataFrame):
        con.execute(f"""CREATE TABLE IF NOT EXISTS documents (
                        url VARCHAR,
                        title VARCHAR,
                        chunk TEXT,
                        embeddings FLOAT[256],
                        chunk_id VARCHAR
                    )""")
        con.sql("INSERT INTO documents SELECT * FROM dataset")

def create_fts_index(con: duckdb.DuckDBPyConnection):
        con.install_extension("fts")
        con.load_extension("fts")
        try:
            con.execute(f"""PRAGMA drop_fts_index('documents')""")
        except duckdb.CatalogException:
            pass
        query = """
        PRAGMA create_fts_index('documents', 'chunk_id', 'chunk',  
        stemmer = 'portuguese', ignore = '(\\.|[^a-z])+', strip_accents = 1, lower = 1, overwrite = 0)
        """
        con.execute(query)


def full_text_search(con:duckdb.DuckDBPyConnection,
                    query:str,
                    top_k:int|None = 10) -> pl.DataFrame:
    db_query = f"""
    SELECT url, chunk, chunk_id, score
    FROM (
        SELECT *, fts_main_documents.match_bm25(chunk_id, '{query}') AS score
        FROM documents
        )
    WHERE score IS NOT NULL
    ORDER BY score DESC
    LIMIT {top_k}
    """
    results = con.execute(db_query).df()
    return results
