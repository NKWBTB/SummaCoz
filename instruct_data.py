from llama2_gen import load_jsonl, dump2jsonl, apply_template
from template import CONSISTENT_STRING, CONSISTENT_TEMPLATE, INCONSISTENT_STRING, inst_parse, annot_parse, COT_TEMPLATE, ZEROSHOT_TEMPLATE, DOC_FIRST
import os
import torch
from datasets import Dataset
from transformers import BatchEncoding

data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"]

class InstructData:
    def __init__(self, tokenizer, label_only = False, cot_folder = "data/final", debug = False) -> None:
        self.raw_folder = "data/raw/"
        self.cot_folder = cot_folder
        self.data_names = set(["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"])
        self.tokenizer = tokenizer
        self.template = COT_TEMPLATE.format(input=DOC_FIRST)
        self.label_only = label_only
        self.debug = debug
        if label_only:
            self.template = ZEROSHOT_TEMPLATE.format(input=DOC_FIRST)
    
    def load_data(self, desired_set, cut, load_label=False, exclude_set=None):
        print(desired_set, cut)
        datas = []
        for name in desired_set:
            if cut == 'train':
                data = load_jsonl(os.path.join(self.cot_folder, exclude_set, f"{name}_val.jsonl"))
            elif cut == 'val':
                if load_label:
                    data = load_jsonl(os.path.join(self.cot_folder, exclude_set, f"{name}_val.jsonl"))
                else:
                    data = load_jsonl(os.path.join(self.raw_folder, f"{name}_val.jsonl"))
            else:
                data = load_jsonl(os.path.join(self.raw_folder, f"{name}_{cut}.jsonl"))
            datas.extend(data)
        if self.debug: datas = datas[:100]
        datas = Dataset.from_list(datas)
        datas = datas.map(lambda x : self.preprocess_function(x, load_label=load_label), 
                          # remove_columns=self.remove_columns,
                          batched=True)
        return datas
        
    def load_train(self, name):
        desired_set = self.data_names - set([name])
        self.train_set = self.load_data(desired_set, 'train', load_label=True, exclude_set=name)
        return self.train_set.with_transform(self.collate_fn)

    def load_val(self, name, load_label):
        desired_set = set([name])
        self.val_set = self.load_data(desired_set, 'val', load_label=load_label, exclude_set=name)
        return self.val_set.with_transform(self.collate_fn)

    def load_test(self, name):
        desired_set = set([name])
        test_set = self.load_data(desired_set, 'test', load_label=False)
        return test_set.with_transform(self.collate_fn)

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
        
    def preprocess_function(self, examples, load_label):
        batch_size = len(examples["claim"])
        input_text = apply_template(examples, self.template, self.tokenizer)
        inputs = input_text["input"]
        model_inputs = self.tokenizer(inputs, add_special_tokens=False)
        if load_label:
            if self.label_only:
                targets = ["yes" if x else "no" for x in examples["label"]]
            elif "cot" in examples:
                targets = [str(x) for x in examples["cot"]]
            labels = self.tokenizer(targets, add_special_tokens=False)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            if load_label: label_input_ids = labels["input_ids"][i] + [self.tokenizer.eos_token_id]
            else: label_input_ids = []
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            model_inputs.setdefault("labels", []).append([-100] * len(sample_input_ids) + label_input_ids)
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        return model_inputs

def read_text(path):
    with open(path, "r", encoding="UTF-8") as f:
        text = f.read()
    return text

def merge_reasoning(parse_func, neg_reason_folder, pos_reason_folder=None, fallback_func=None):
    raw_folder = "data/raw"
    dump_folder = "data/merge"
    cut = "val"
    for name in data_names:
        print(name, cut)
        data = load_jsonl(os.path.join(raw_folder, "_".join([name, cut])+'.jsonl'))
        size = len(data)
        index2delete = set()
        for i in range(size):
            data[i]["cot"] = CONSISTENT_TEMPLATE
            if data[i]["label"]:
                if pos_reason_folder is None: continue
                annot_file = os.path.join(pos_reason_folder, name, f"{i}.txt")
            else:
                annot_file = os.path.join(neg_reason_folder, name, f"{i}.txt")
            reasoning = ""
            if os.path.exists(annot_file):
                reasoning = read_text(annot_file)
            parsed = parse_func(reasoning)
            parsed_fallback = ""
            if not fallback_func is None: 
                parsed_fallback = fallback_func(reasoning, filter=not data[i]["label"])
            if len(parsed) > 3:
                reasoning = parsed + "\n" + (CONSISTENT_STRING if data[i]["label"] else INCONSISTENT_STRING)
                data[i]["cot"] = reasoning
            elif len(parsed_fallback) > 3:
                reasoning = parsed_fallback + "\n" + (CONSISTENT_STRING if data[i]["label"] else INCONSISTENT_STRING)
                data[i]["cot"] = reasoning
            else:
                index2delete.add(i)
        print("Deleted", len(index2delete))
        data = [data[i] for i in range(size) if not i in index2delete]
        dump2jsonl(data, os.path.join(dump_folder, "_".join([name, cut])+'.jsonl'))

if __name__ == '__main__':
    # merge_reasoning(parse_func=lambda x : inst_parse(x, filter=True), reason_folder="data/raw_annot")
    # merge_reasoning(parse_func=annot_parse, reason_folder="data/annot")
    merge_reasoning(parse_func=annot_parse, neg_reason_folder="data/annot", pos_reason_folder="data/positive", fallback_func=inst_parse)
    import pdb
    pdb.set_trace()

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    instructdata = InstructData(tokenizer)
    instructdata.load_train("cogensumm")

        