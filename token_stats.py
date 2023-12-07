import config as CFG
from instruct_data import InstructData, data_names
from transformers import AutoTokenizer
from llama2_gen import dump2jsonl, load_jsonl
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, padding_side='left')
tokenizer.pad_token_id = tokenizer.bos_token_id

def plot(field):
    text_tokens = []
    for name in data_names:
        data = load_jsonl(os.path.join('data/merge', f"{name}_val.jsonl"))
        for line in tqdm(data):
            inputs = tokenizer(line[field])
            length = len(inputs["input_ids"])
            if length < 4000:
                text_tokens.append(length)
    _ = plt.hist( text_tokens, bins='auto')
    plt.savefig(f"{field}_tokens.png")
    plt.clf()

plot("document")
plot("claim")
plot("cot")