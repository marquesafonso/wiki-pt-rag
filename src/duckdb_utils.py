import os
import duckdb
import polars as pl

def export_db(con:duckdb.DuckDBPyConnection, path:str):
    con.sql(f"""EXPORT DATABASE '{path}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD);""")

def import_db(path:str):
    with duckdb.connect(path) as con:
        con.sql(f"PRAGMA import_database('{path}');")
    return con

def setup_db(con:duckdb.DuckDBPyConnection, dataset: pl.DataFrame):
    con.execute(f"""CREATE TABLE IF NOT EXISTS documents (
                    chunk_id VARCHAR,
                    chunk TEXT
                )""")
    con.sql("INSERT INTO documents SELECT chunk_id, chunk FROM dataset")
    con.commit()

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
    con.commit()


def full_text_search(con:duckdb.DuckDBPyConnection,
                    query:str,
                    top_k:int) -> pl.DataFrame:
    db_query = f"""
    SELECT chunk_id, fts_score
    FROM (
        SELECT *, fts_main_documents.match_bm25(chunk_id, '{query}') AS fts_score
        FROM documents
        )
    WHERE fts_score IS NOT NULL
    LIMIT {top_k}
    """
    results = con.execute(db_query).pl()
    return results

def get_fts_results(dataset:pl.DataFrame, path:str, query:str, top_k:int):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        with duckdb.connect(path) as con:
            setup_db(con=con, dataset=dataset)
            create_fts_index(con=con)
            # export_db(con=con, path=path)
    with duckdb.connect(path) as con:
        # con = import_db(con=con, path=path)
        res = full_text_search(con=con, query=query, top_k=top_k)
    return res
