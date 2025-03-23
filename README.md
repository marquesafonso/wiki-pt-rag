## Criação do dataset

+ Chunking semântico utilizando o modelo estático: sentence-transformers/static-similarity-mrl-multilingual-v1
+ Demorou aproximadamente 8500s or 2.3h

<image src="example.png"/>

## Análise de sensibilidade à escrita em disco da vector store (vicinity:usearch)

+ Query: "Qual é a ciência que estuda o espaço, os astros e as estrelas?"
+ Modelo: sentence-transformers/static-similarity-mrl-multilingual-v1
+ Backend: USEARCH
+ Alpha: 0.5
+ Top k: 5000
+ Unidade: segundos

| Tempo (s) / Linhas | 10.000 | 100.000 | 400.000 | 1.000.000 | 2.052.058 |
| :------- | :---: | :---: | :---: | :---: | :---: |
| Sem índice | 8.61 | 35 | 76.5 | 185.7 | 547.5 |
| Com índice | 6.94 | 24 | 37.6 | 62.7 | 244.3 |
    
## Análise de sensibilidade à escrita em disco da base de dados com índice (duckdb:fts)

+ Query: "Qual é a ciência que estuda o espaço, os astros e as estrelas?"
+ Vector store com 2.052.058 linhas mantida constante
+ FTS params: [stemmer = 'portuguese', ignore = '(\\.|[^a-z])+', strip_accents = 1, lower = 1, overwrite = 0]
+ Alpha: 0.5
+ Top k: 5000
+ Unidade: segundos

| Tempo (s) / Linhas | 10.000 | 100.000 | 400.000 | 1.000.000 | 2.052.058 |
| :------- | :---: | :---: | :---: | :---: | :---: |
| Sem índice | 8.62 | 32.9 | 43.7 | 63.27 | 141.8 |
| Com índice | 5.7 | 5.1 | 6.08 | 7.45 | 17.2 |

## Análise de sensibilidade ao parâmetro Top K

+ Query: "Qual é a ciência que estuda o espaço, os astros e as estrelas?"
+ Vector store completa linhas mantida constante
+ Base de dados completa mantida constante
+ Alpha: 0.5
+ Unidade: segundos

| Tempo (s) / Top k | 10 | 100 | 1000 | 5.000 | 10.000 | 50.000 | 100.000 |
| :------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sem índice | 10.36 | 10.25 | 10.73 | 11.20 | 10.58 | 11.04 | 12.32 |

## Análise de compressão da base de dados duckdb com indice fts

Situação base:
```
def get_fts_results(dataset:pl.DataFrame, path:str, query:str, top_k:int):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
        with duckdb.connect(path) as con:
            setup_db(con=con, dataset=dataset)
            create_fts_index(con=con)
    with duckdb.connect(path) as con:
        res = full_text_search(con=con, query=query, top_k=top_k)
    return res
```

Tentativa de compressão da base de dados:
```
def export_db(con:duckdb.DuckDBPyConnection, path:str):
    con.sql(f"""EXPORT DATABASE '{path}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD);""")

def import_db(con:duckdb.DuckDBPyConnection, path:str):
    con.sql(f"PRAGMA import_database('{path}');")
    return con

def get_fts_results(dataset:pl.DataFrame, path:str, query:str, top_k:int):
    if not os.path.exists(path):
        with duckdb.connect() as con:
            setup_db(con=con, dataset=dataset)
            create_fts_index(con=con)
            export_db(con=con, path=path)
    with duckdb.connect() as con:
        con = import_db(con=con, path=path)
        res = full_text_search(con=con, query=query, top_k=top_k)
    return res
```

Para o mesmo query das experiências anteriores, quando comparado com a utilização do metódo duckdb.connect() numa base de dados em disco, temos o seguinte resultados:

| Query / Método | Com compressão (~1GB) | Sem compressão (~5GB) |
| :------- | :---: | :---: |
| Qual é a ciência que estuda o espaço, os astros e as estrelas? |  24 s | 12 s |

# Experiência de quantização dos embeddings (int8)

```
#src.vicinity_vss.py
def create_vector_store(dataset:pl.DataFrame, repo_id:str, token:str):
    vector_store = Vicinity.from_vectors_and_items(
        vectors=sentence_transformers.quantize_embeddings(
            embeddings=np.array(dataset["embeddings"].to_list()).astype(np.float32),
            precision="int8"
        ),
        items=dataset["chunk_id"].to_list(),
        backend_type=Backend.USEARCH,
        metric=Metric.COSINE
    )
    vector_store.push_to_hub(repo_id=repo_id, token=token, private=True)
    return vector_store

#main.py
...
query_vector = sentence_transformers.quantize_embeddings(embeddings=embedder.encode(query.strip()), precision="float16")
...
```

Perda substantiva de qualidade da representação vectorial e respostas obtidas.

Poder-se-á testar uma outra configuração com embeddings binários + reranking.
