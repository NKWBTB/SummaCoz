import os
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, load_dataset
from utils import dump2jsonl, load_jsonl
from template import swap_punctuation
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from t5_trainer import PROMPT
# PROMPT = \
# """Premise: {article} 

# Hypothesis: {summary}

# Is the hypothesis true given the premise?
# """

def main_t5(t5, tokenizer, data, dump_to, max_new_tokens=10):
    from transformers import pipeline
    
    dump_folder = os.path.dirname(dump_to)
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder, exist_ok=True)

    ppl = pipeline("text2text-generation",
                   model=t5,
                   tokenizer=tokenizer)
    print(ppl(data[0]["prompt"], do_sample=False, max_new_tokens=max_new_tokens))
    generated = [out[0]["generated_text"] for out in tqdm(ppl(KeyDataset(data, "prompt"), 
                                                            max_new_tokens=max_new_tokens,
                                                            num_return_sequences=1,
                                                            do_sample=False))]
    dump2jsonl(generated, dump_to)

def posthoc_parse(input:str):
    generation_part = input.partition("\n\n")[0].lower()
    words = swap_punctuation(generation_part).split()
    # assert ("not" in words) or ("no" in words) or ("yes" in words)
    return 1 if ("yes" in words) else 0

def apply_template(example, tokenizer):
    example["prompt"] = PROMPT.format(summary=example["claim"], article=example["document"])
    encoded = tokenizer.encode(example["prompt"])
    if len(encoded) > 2000:
        example["prompt"] = tokenizer.decode(encoded[:2000], skip_special_tokens=True)
    return example

def load_summac(input_path, name, cut, tokenizer):
    input_path = os.path.join(input_path, "_".join([name, cut])+'.jsonl')
    data = load_jsonl(input_path)
    datas = Dataset.from_list(data)
    datas = datas.map(lambda x: apply_template(x, tokenizer=tokenizer))
    return datas

def load_ragtruth(data_dir, split):
    response = load_jsonl(os.path.join(data_dir, "response.jsonl"))
    if split is not None:
        response = [item for item in response if item["split"] == split]
    source_info = load_jsonl(os.path.join(data_dir, "source_info.jsonl"))
    source_info = {item["source_id"]:item for item in source_info}

    data = []
    for line in response:
        source = source_info[line["source_id"]]
        if source["task_type"] != "Summary": continue
        sample = {
            "model": line["model"],
            "document": source["source_info"],
            "claim": line["response"],
            "label": 1 if len(line["labels"]) == 0 else 0
        }
        data.append(sample)
    datas = Dataset.from_list(data)
    datas = datas.map(lambda x: apply_template(x, tokenizer=tokenizer))
    return datas

if __name__ == '__main__':
    MODEL_ID = "google/flan-t5-base"
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, 
                                                  device_map="auto")
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        "./t5_output/checkpoint-4500",
        is_trainable=False,
    )
    model = model.merge_and_unload()

    data_names = ["cogensumm", "summeval", "xsumfaith", "polytope", "factcc", "frank"]
    cut = "test"
    for name in data_names:
        print(name, cut)
        data = load_summac("data/raw", name, cut, tokenizer)
        dump_file = f"test/{name}_response.jsonl"
        if not os.path.exists(dump_file):
            main_t5(model, tokenizer, data, dump_file)
        generated = load_jsonl(dump_file)
        pred = [posthoc_parse(line) for line in generated]
        label = data["label"]
        print(balanced_accuracy_score(label, pred))
        print(confusion_matrix(label, pred))

    name = "RAGTruth"
    print(name)
    data = load_ragtruth("RAGTruth/dataset", "test")
    dump_file = f"test/{name}_response.jsonl"
    if not os.path.exists(dump_file):
        main_t5(model, tokenizer, data, dump_file)
    generated = load_jsonl(dump_file)
    pred = [posthoc_parse(line) for line in generated]
    label = data["label"]
    print("BA", balanced_accuracy_score(label, pred))
    print(confusion_matrix(label, pred))
    import numpy as np
    pred = 1 - np.array(pred)
    label = 1 - np.array(label)
    print("Precision", precision_score(label, pred))
    print("Recall", recall_score(label, pred))
    print("F1", f1_score(label, pred))

    name = "AggreFact_SoTA"
    print(name)
    data = load_dataset("csv", data_files="AggreFact/aggre_fact_sota.csv", split="train")
    data = data.rename_column("doc", "document")
    data = data.rename_column("summary", "claim")
    data = data.map(lambda x : apply_template(x, tokenizer))
    data = data.filter(lambda x : x["cut"]=="test")
    dump_file = f"test/{name}_response.jsonl"
    if not os.path.exists(dump_file):
        main_t5(model, tokenizer, data, dump_file)
    generated = load_jsonl(dump_file)
    pred = [posthoc_parse(line) for line in generated]
    label = data["label"]
    print("BA", balanced_accuracy_score(label, pred))
    print(confusion_matrix(label, pred))

    name = "HHEM"
    print(name)
    data = load_dataset("csv", data_files="benchmark/leaderboard_summaries.csv", split="train")
    data = data.rename_column("source", "document")
    data = data.rename_column("summary", "claim")
    data = data.map(lambda x : apply_template(x, tokenizer))
    dump_file = f"test/{name}_response.jsonl"
    if not os.path.exists(dump_file):
        main_t5(model, tokenizer, data, dump_file)
    generated = load_jsonl(dump_file)
    pred = [posthoc_parse(line) for line in generated]

    df = data.to_pandas()
    df["pred"] = pred

    import pdb
    pdb.set_trace()

    models = set(df["model"])
    result = []
    import numpy as np
    for model in models:
        all = np.array((df[df["model"] == model]["pred"]))
        rate = np.sum(all) / len(all)
        result.append((rate, model))
    sorted_res = sorted(result, reverse=True)
    for item in sorted_res: print(item)

    

