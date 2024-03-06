# from sft_training import *
import os
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from utils import dump2jsonl, load_jsonl
from template import swap_punctuation
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from t5_trainer import PROMPT

lora_adapter = LoRARequest("adapter", 1, "sft_output/checkpoint-2500")

def initialize_vllm(MODEL, dtype):
    llm = LLM(model=MODEL, dtype=dtype, max_model_len=16384, enable_lora=True)
    return llm

def main_vllm(llm, data, dump_to, max_new_tokens=50):
    dump_folder = os.path.dirname(dump_to)
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder, exist_ok=True)

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    print(llm.generate(data[0]["prompt"], sampling_params, lora_request=lora_adapter))
    generated = llm.generate(KeyDataset(data, "prompt"), sampling_params, lora_request=lora_adapter)
    lines = []
    for idx, line in enumerate(generated):
        lines.append(line.outputs[0].text)
    dump2jsonl(lines, dump_to)

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

# def load_summac(input_path, name, cut, add_response=True, add_reason=True):
#     input_path = os.path.join(input_path, "_".join([name, cut])+'.jsonl')
#     data = load_jsonl(input_path)
#     datas = Dataset.from_list(data)
#     datas = datas.rename_column("document", "premise")
#     datas = datas.rename_column("claim", "hypothesis")
#     datas = datas.map(lambda x: process_message(x, tokenizer, POSTHOC_TEXT, add_response, add_reason))
#     return datas

def load_summac(input_path, name, cut, tokenizer):
    input_path = os.path.join(input_path, "_".join([name, cut])+'.jsonl')
    data = load_jsonl(input_path)
    datas = Dataset.from_list(data)
    def apply_template(example):
        example["prompt"] = PROMPT.format(summary=example["claim"], article=example["document"])
        encoded = tokenizer.encode(example["prompt"])
        if len(encoded) > 2000:
            example["prompt"] = tokenizer.decode(encoded[:2000], skip_special_tokens=True)
        return example
    datas = datas.map(apply_template)
    return datas

if __name__ == '__main__':
    # llm = initialize_vllm("mistralai/Mistral-7B-Instruct-v0.2", "bfloat16")
    # # ESNLI Test ACC
    # esnli = load_esnli("test", False)
    # dump_file = "test/esnli_test.jsonl"
    # if not os.path.exists(dump_file):
    #     main_vllm(llm, esnli, dump_file)
    # generated = load_jsonl(dump_file)

    # pred = [posthoc_parse(line) for line in generated]
    # label = [1 if x == 0 else 0 for x in esnli["label"]]
    # print("ESNLI", accuracy_score(label, pred))

    # # ANLI Test ACC
    # for round in ["test_r1", "test_r2", "test_r3"]:
    #     anli_test = load_anli(round, False)
    #     dump_file = f"test/anli_{round}.jsonl"
    #     if not os.path.exists(dump_file):
    #         main_vllm(llm, anli_test, dump_file)
    #     generated = load_jsonl(dump_file)

    #     pred = [posthoc_parse(line) for line in generated]
    #     label = [1 if x == 0 else 0 for x in anli_test["label"]]
    #     print(f"ANLI_{round}", accuracy_score(label, pred))
    MODEL_ID = "google/flan-t5-large"
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, 
                                                  device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        "./t5_output/checkpoint-1000",
        is_trainable=False,
    )
    model = model.merge_and_unload()

    data_names = ["cogensumm", "summeval", "xsumfaith", "polytope", "factcc", "frank"]
    cut = "test"
    for name in data_names:
        print(name, cut)
        data = load_summac("data/raw", name, cut, tokenizer=tokenizer) #add_response=False)
        dump_file = f"test/{name}_response.jsonl"
        if not os.path.exists(dump_file):
            # main_vllm(llm, data, dump_file)
            main_t5(model, tokenizer, data, dump_file)
        generated = load_jsonl(dump_file)
        pred = [posthoc_parse(line) for line in generated]
        label = data["label"]
        print(balanced_accuracy_score(label, pred))
        print(confusion_matrix(label, pred))
