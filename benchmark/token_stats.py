from utils import load_jsonl, dump2jsonl
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def filter(field):
    filtered_data = []
    data = load_jsonl('benchmark_v_0_5.jsonl')
    for line in tqdm(data):
        inputs = tokenizer(line[field])
        length = len(inputs["input_ids"])
        if length < 2000:
            filtered_data.append(line)
    print(len(filtered_data))
    dump2jsonl(filtered_data, "benchmark_v_0_6.jsonl")

filter("document")