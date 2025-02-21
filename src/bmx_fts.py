from baguetter import BMXSparseIndex

def bmx_fts(doc_ids, docs):
    index = BMXSparseIndex(index_name="wiki-pt-index", normalize_scores=True, n_workers=8)
    index.add_many(doc_ids, docs, show_progress=True)
    index.save("indices/wiki-pt-index")