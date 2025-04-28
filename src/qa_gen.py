import re
from gradio_client import Client
import polars as pl
import lmstudio as lms
from typing import Dict, List, Any

def get_llm_answer(input_text:str):
    llm = lms.llm("PleIAs/Pleias-RAG-1B-gguf")
    prediction = llm.respond_stream(input_text)
    for token in prediction:
        print(token.content, end="", flush=True)

def format_prompt(query: str, sources: List[Dict[str, Any]]) -> str:
    """
    Format the query and sources into a prompt with special tokens.

    The prompt follows a specific format with special tokens to guide the model:
    - <|query_start|>...<|query_end|> for the user's question
    - <|source_start|><|source_id|>N ...<|source_end|> for each source
    - <|language_start|> to indicate the beginning of generation
    Args:
        query: The user's question
        sources: List of source documents with their metadata. Format is list of dictionaries,
                    each with a "text" key and optional "metadata" key.
                    The metadata is not used in the prompt but can be useful for later processing.
                    Example: [{"text": "Document text", "metadata": {"source_id": 1, "source_name": "Doc1"}}]
    Returns:
        Formatted prompt string
    """
    prompt = f"<|query_start|>{query}<|query_end|>\n"

    # Add each source with its ID
    for idx, source in enumerate(sources, 1):
        source_text = source.get("text", "")
        prompt += f"<|source_start|><|source_id|>{idx} {source_text}<|source_end|>\n"

    # Add the source analysis start token
    prompt += "<|language_start|>\n"

    return prompt

def main(query:str):
    client = Client("http://127.0.0.1:7860/")
    result = client.predict(
            query=query,
            alpha=0.4,
            num_results=4,
            api_name="/predict"
    )
    schema={'url':pl.String, 'title': pl.String, 'chunk': pl.String, 'embeddings': pl.List(pl.Float64),
            'chunk_id': pl.String, 'fts_score': pl.Float64, 'chunk_id_right': pl.String, 'vss_score':pl.Float64,
            'normalized_fts_sim':pl.Float64, 'normalized_vss_sim':pl.Float64, 'convex_score':pl.Float64}
    df = pl.DataFrame(data = result["data"], schema=schema, orient='row')
    source_texts = [{
                    "text" : r["chunk"],
                    "metadata": {
                        "url" : r["url"],
                        "title": r["title"]
                        }
                    }
                    for r in df.iter_rows(named=True)
                    ]
    input_text = format_prompt(query, source_texts)
    get_llm_answer(input_text=input_text)

if __name__ == '__main__':
    main(query="Qual é a ciência que estuda o espaço, os astros e as estrelas?")