import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import deepspeed
import argparse
import json
from tqdm import tqdm

def load_jsonl(input_path):
    with open(input_path, "r", encoding="UTF-8") as f:
        data = [json.loads(line) for line in f]
    return data

def dump2jsonl(lines, output_path):
    with open(output_path, "w", encoding="UTF-8") as f:
        for line in lines:
            f.write(json.dumps(line) + '\n') 

def main():
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    MODEL = "meta-llama/Llama-2-13b-chat-hf"
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

    template = \
    """<s>[INST] Note that consistency means all information in the summary is supported by the article. 
    It's known that the following summary is not consistent with the article. 
    Find out why.

    Summary: 
    {summary} 

    Article: 
    {article} 

    Explain your reasoning step by step:[/INST]
    """

    def apply_template(batch):
        docs = batch["document"]
        sums = batch["claim"]
        texts = [template.format(summary=_sum, article=_doc) for _doc, _sum in zip(docs, sums)]
        return {"input": texts}

    data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"]
    data_folder = "data/raw/"
    dump_folder = "data/gen/"
    cut = 'val'
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset

    for name in data_names:
        print(name, cut)
        input_path = os.path.join(data_folder, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        data = [line for line in data if line["label"] == 0]
        datas = Dataset.from_list(data)
        datas.set_transform(apply_template)
        generated = [out for out in tqdm(pipeline(KeyDataset(datas, "input"), 
                                        max_new_tokens=512,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer.eos_token_id))]
        for line, reasoning in zip(data, generated): line["reasoning"] = reasoning
        dump2jsonl(generated, os.path.join(dump_folder, "_".join([name, cut])+'_neg.jsonl'))

if __name__ == '__main__':
    main()