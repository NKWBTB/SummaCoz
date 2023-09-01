import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import deepspeed
import argparse
import json
from tqdm import tqdm
from template import COT_TEMPLATE, ZEROSHOT_TEMPLATE, SELFINST_TEMPLATE

def load_jsonl(input_path):
    with open(input_path, "r", encoding="UTF-8") as f:
        data = [json.loads(line) for line in f]
    return data

def dump2jsonl(lines, output_path):
    with open(output_path, "w", encoding="UTF-8") as f:
        for line in lines:
            f.write(json.dumps(line) + '\n') 

def main(template, dump_folder, filter=True, MODEL="meta-llama/Llama-2-13b-chat-hf", max_new_tokens=512):
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                torch_dtype=torch.float16,
                                                trust_remote_code=True)
                                                # device_map=args.local_rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ds_engine = deepspeed.init_inference(model,
                                        tensor_parallel={"tp_size": 2},
                                        dtype=torch.float16,
                                        max_out_tokens=4096,
                                        replace_with_kernel_inject=False)
    pipeline = transformers.pipeline(
        "text-generation",
        model=ds_engine,
        tokenizer=tokenizer,
        device=args.local_rank,
    )

    results = pipeline("<s>[INST] Why is earth round? [/INST]",
                    max_new_tokens=512,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,)
    dump2jsonl(results, "test_results.txt")

    def apply_template(batch):
        docs = batch["document"]
        sums = batch["claim"]
        texts = [template.format(summary=_sum, article=_doc) for _doc, _sum in zip(docs, sums)]
        return {"input": texts}

    data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"]
    data_folder = "data/raw/"
    cut = 'val'
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset

    for name in data_names:
        print(name, cut)
        input_path = os.path.join(data_folder, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        if filter: data = [line for line in data if line["label"] == 0]
        datas = Dataset.from_list(data)
        datas.set_transform(apply_template)
        generated = [out for out in tqdm(pipeline(KeyDataset(datas, "input"), 
                                        max_new_tokens=max_new_tokens,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer.eos_token_id))]
        # for line, reasoning in zip(data, generated): line["reasoning"] = reasoning
        dump2jsonl(generated, os.path.join(dump_folder, "_".join([name, cut])+'_neg.jsonl'))

def postprocess():
    data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"]
    data_folder = "data/raw/"
    gen_folder = "data/gen/"
    annot_folder = "data/annot/"
    cut = 'val'

    for name in data_names:
        print(name, cut)
        input_path = os.path.join(data_folder, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        data_len = len(data)
        idx = [i for i in range(data_len) if data[i]["label"] == 0]
        gen = load_jsonl(os.path.join(gen_folder, "_".join([name, cut])+'_neg.jsonl'))

        output_folder = os.path.join(annot_folder, name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i, line in enumerate(gen):
            sample_no = idx[i]
            filename = f"{sample_no}.txt"
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w") as f:
                f.write(line[0]["generated_text"] + "\n\n###Corrected:\n")

if __name__ == '__main__':
    # Zeroshot
    main(ZEROSHOT_TEMPLATE, "data/zeroshot", False, max_new_tokens=10)
    # postprocess()