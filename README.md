## Criação do dataset

+ Chunking semântico utilizando o modelo estático: sentence-transformers/static-similarity-mrl-multilingual-v1
+ Demorou aproximadamente 8500s or 2.3h

## Análise de sensibilidade à escrita em disco da vector store (vicinity:usearch)

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

+ Vector store completa linhas mantida constante
+ Base de dados completa mantida constante
+ Alpha: 0.5
+ Unidade: segundos

| Tempo (s) / Top k | 10 | 100 | 1000 | 5.000 | 10.000 | 50.000 | 100.000 |
| :------- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sem índice | 10.36 | 10.25 | 10.73 | 11.20 | 10.58 |  | 12.32 |