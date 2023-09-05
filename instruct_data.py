from llama2_gen import load_jsonl, dump2jsonl
from template import CONSISTENT_STRING, INCONSISTENT_STRING, inst_parse, COT_TEMPLATE
import os
import torch
from datasets import Dataset
from transformers import BatchEncoding

data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"]

class InstructData:
    def __init__(self, tokenizer) -> None:
        self.test_folder = "data/raw/"
        self.val_folder = "data/merge/"
        self.data_names = set(["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"])
        self.tokenizer = tokenizer
    
    def load_data(self, name, cut):
        print(name, cut)
        if cut == 'val':
            datas = load_jsonl(os.path.join(self.val_folder, f"{name}_{cut}.jsonl"))
        else:
            datas = load_jsonl(os.path.join(self.test_folder, f"{name}_{cut}.jsonl"))
        datas = Dataset.from_list(datas)
        datas = datas.map(self.preprocess_function, 
                          # remove_columns=self.remove_columns,
                          batched=True)
        return datas.with_transform(self.collate_fn)

    def collate_fn(self, model_inputs):
        max_length = max([len(sample) for sample in model_inputs["input_ids"]])
        batch_size = len(model_inputs["input_ids"])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = model_inputs["labels"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["labels"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i][:max_length]
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i][:max_length]
            model_inputs["labels"][i] = model_inputs["labels"][i][:max_length]
        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
        model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
        model_inputs["labels"] = torch.tensor(model_inputs["labels"])
        model_inputs = BatchEncoding(model_inputs)
        return model_inputs
        
    def preprocess_function(self, examples):
        istrain = "cot" in examples
        batch_size = len(examples["claim"])
        inputs = [COT_TEMPLATE.format(summary=claim, article=document) for claim, document in zip(examples["claim"], examples["document"])]
        model_inputs = self.tokenizer(inputs, add_special_tokens=False)
        if istrain:
            targets = [str(x) for x in examples["cot"]]
            labels = self.tokenizer(targets, add_special_tokens=False)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            if istrain: label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            else: label_input_ids = []
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            model_inputs.setdefault("labels", []).append([-100] * len(sample_input_ids) + label_input_ids)
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        return model_inputs

def read_text(path):
    with open(path, "r", encoding="UTF-8") as f:
        text = f.read()
    return text

def merge_reasoning(parse_func):
    raw_folder = "data/raw"
    reason_folder = "data/annot"
    dump_folder = "data/merge"
    cut = "val"
    for name in data_names:
        print(name, cut)
        data = load_jsonl(os.path.join(raw_folder, "_".join([name, cut])+'.jsonl'))
        size = len(data)
        for i in range(size):
            if data[i]["label"]:
                data[i]["cot"] = CONSISTENT_STRING
            else:
                reasoning = read_text(os.path.join(reason_folder, name, f"{i}.txt"))
                reasoning = parse_func(reasoning) + "\n" + INCONSISTENT_STRING
                data[i]["cot"] = reasoning
        dump2jsonl(data, os.path.join(dump_folder, "_".join([name, cut])+'.jsonl'))

if __name__ == '__main__':
    merge_reasoning(inst_parse)
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    instructdata = InstructData(tokenizer)
    instructdata.load_data("cogensumm", "val")

        