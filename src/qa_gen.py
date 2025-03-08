import re
from gradio_client import Client
import polars as pl
import lmstudio as lms
from transformers import AutoModelForCausalLM, AutoTokenizer

def transformers_answer(input_text):
    model_id = "PleIAs/Pleias-Pico"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0].split("<|answer_start|>"), skip_special_tokens=True))

def get_llm_answer(input_text:str):
    llm = lms.llm("PleIAs/Pleias-Pico-GGUF")
    prediction = llm.respond_stream(input_text)
    for token in prediction:
        print(token.content, end="", flush=True)


def chat_template(query:str, results:pl.DataFrame):
    query_str = f"""<|query_start|>{query}<|query_end|>"""
    source_str = [f"<|source_start|><|source_id_start|>{r['url']}<|source_id_end|>{r['chunk']}<|source_end|>" for r in results.iter_rows(named=True)]
    start_token = "<|source_analysis_start|>"
    intermediate_str = query_str + "".join(source_str) + start_token
    chat_str = re.sub(r'\s+', ' ', intermediate_str)
    return chat_str

def main(query:str):
    client = Client("http://127.0.0.1:7860/")
    result = client.predict(
            query=query,
            alpha=0.5,
            num_results=3,
            api_name="/search"
    )
    schema={'url':pl.String, 'title': pl.String, 'chunk': pl.String, 'embeddings': pl.List(pl.Float64),
            'chunk_id': pl.String, 'fts_score': pl.Float64, 'chunk_id_right': pl.String, 'vss_score':pl.Float64,
            'normalized_fts_sim':pl.Float64, 'normalized_vss_sim':pl.Float64, 'convex_score':pl.Float64}
    df = pl.DataFrame(data = result["data"], schema=schema, orient='row')
    input_text = chat_template(query, df)
    get_llm_answer(input_text=input_text)

if __name__ == '__main__':
    main(query="Qual é a ciência que estuda o espaço, os astros e as estrelas?")