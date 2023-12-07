import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import deepspeed
import argparse
import json
from tqdm import tqdm
from template import COT_TEMPLATE, ZEROSHOT_TEMPLATE, SELFINST_TEMPLATE, DOC_FIRST, SUM_FIRST, SELFINST_TEMPLATE_EXTRA, XSUM_EXTRA_TEMPLATE, SELFINST_TEMPLATE_POSITIVE, CLAIM_PROMPT, TYPE_EXTRA_TEMPLATE

def load_jsonl(input_path):
    with open(input_path, "r", encoding="UTF-8") as f:
        data = [json.loads(line) for line in f]
    return data

def dump2jsonl(lines, output_path):
    with open(output_path, "w", encoding="UTF-8") as f:
        for line in lines:
            f.write(json.dumps(line) + '\n') 

def apply_template(batch, template, tokenizer):
    docs = batch["document"]
    tokens = tokenizer(docs)["input_ids"]
    bs = len(docs)
    for i in range (bs):
        if len(tokens[i]) > 2000:
            print(f"{len(tokens[i])} > 2000")
            docs[i] = " ".join([batch["doc_sents"][i][idx] for idx in batch["rel_index"][i]])
    sums = batch["claim"]
    if "{xsum_annotation}" in template:
        extras = batch["extra"]
        texts = [template.format(summary=_sum, article=_doc, xsum_annotation=extra) for _doc, _sum, extra in zip(docs, sums, extras)]
    elif "{type_annotation}" in template:
        extras = batch["extra"]
        texts = [template.format(summary=_sum, article=_doc, type_annotation=extra) for _doc, _sum, extra in zip(docs, sums, extras)]
    else:
        texts = [template.format(summary=_sum, article=_doc) for _doc, _sum in zip(docs, sums)]
    return {"input": texts}

def main(template, dump_folder, filter=None, MODEL="meta-llama/Llama-2-13b-chat-hf", max_new_tokens=512, cut='val', data_names=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # from config import gpu_device_map
    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                torch_dtype=torch.float16,
                                                # device_map="auto",
                                                trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    # ds_engine = model
    ds_engine = deepspeed.init_inference(model,
                                        # tensor_parallel={"tp_size": 2},
                                        mp_size=2,
                                        dtype=torch.float16,
                                        max_out_tokens=4096,
                                        replace_with_kernel_inject=False)
    pipeline = transformers.pipeline(
        "text-generation",
        model=ds_engine,
        tokenizer=tokenizer,
        # device=0, 
        device=args.local_rank,
    )

    results = pipeline("<s>[INST] Why is earth round? [/INST]",
                    max_new_tokens=512,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,)
    dump2jsonl(results, "test_results.txt")

    data_folder = "data/raw/"
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset

    if not os.path.exists(dump_folder) and args.local_rank == 0:
        os.makedirs(dump_folder, exist_ok=True)

    for name in data_names:
        print(name, cut)
        input_path = os.path.join(data_folder, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        if not filter is None: data = [line for line in data if filter(line)]
        datas = Dataset.from_list(data)
        datas.set_transform(lambda x: apply_template(x, template, tokenizer))
        sample = pipeline(datas[0]["input"],
            max_new_tokens=512,
            num_return_sequences=1,
            pad_token_id=tokenizer.bos_token_id,)
        if args.local_rank == 0: print(sample)
        generated = [out for out in tqdm(pipeline(KeyDataset(datas, "input"), 
                                        max_new_tokens=max_new_tokens,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer.bos_token_id))]
        # for line, reasoning in zip(data, generated): line["reasoning"] = reasoning
        if args.local_rank == 0:
            dump2jsonl(generated, os.path.join(dump_folder, "_".join([name, cut])+'.jsonl'))

def postprocess(annot_folder, data_names, gen_folder="data/gen/", filter=None):
    data_folder = "data/raw/"
    cut = 'val'

    for name in data_names:
        print(name, cut)
        input_path = os.path.join(data_folder, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        data_len = len(data)
        idx = [i for i in range(data_len) if filter(data[i])]
        gen = load_jsonl(os.path.join(gen_folder, "_".join([name, cut])+'.jsonl'))

        output_folder = os.path.join(annot_folder, name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for i, line in enumerate(gen):
            sample_no = idx[i]
            filename = f"{sample_no}.txt"
            output_path = os.path.join(output_folder, filename)
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read()
                annotation = content.partition("###Corrected:")[2].strip()
                if len(annotation) > 0:
                    continue
            
            with open(output_path, "w") as f:
                f.write(line[0]["generated_text"] + "\n\n###Corrected:\n")

positive_only = lambda x: x["label"] == 1
negative_only = lambda x: x["label"] == 0

if __name__ == '__main__':
    data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"][-2:]
    # Zeroshot
    # main(ZEROSHOT_TEMPLATE, "data/zeroshot", False, max_new_tokens=15, cut='test')
    # main(ZEROSHOT_TEMPLATE.format(input=DOC_FIRST), "data/DOCfirst/zeroshot", False, max_new_tokens=15, cut='test')
    # main(ZEROSHOT_TEMPLATE.format(input=SUM_FIRST), "data/SUMfirst/zeroshot", False, max_new_tokens=15, cut='test')
    
    # Zeroshot - Chain-of-Thought
    # main(COT_TEMPLATE, "data/zs_cot", False, max_new_tokens=512, cut='test')
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/DOCfirst/zs_cot", False, max_new_tokens=512, cut='test')
    # main(COT_TEMPLATE.format(input=SUM_FIRST), "data/SUMfirst/zs_cot", False, max_new_tokens=512, cut='test')

    # Selfinstruct
    # main(SELFINST_TEMPLATE, "data/gen")
    # main(SELFINST_TEMPLATE.format(input=DOC_FIRST), "data/gen")
    # main(SELFINST_TEMPLATE_EXTRA.format(input=DOC_FIRST, extra=XSUM_EXTRA_TEMPLATE), "data/gen/xsum_extra", data_names=["xsumfaith"])
    # main(SELFINST_TEMPLATE_POSITIVE.format(input=CLAIM_PROMPT), "data/gen/positive", data_names=data_names, filter=positive_only)
    main(SELFINST_TEMPLATE_EXTRA.format(input=DOC_FIRST, extra=TYPE_EXTRA_TEMPLATE), "data/gen/polytope_extra", data_names=["polytope"], filter=negative_only)

    # Generate data for annotation
    # postprocess(annot_folder="data/annot")
    # postprocess(annot_folder="data/positive", gen_folder="data/gen/positive", filter=positive_only)
    # postprocess(annot_folder="data/annot", gen_folder="data/gen/xsum_extra", filter=negative_only, data_names=["xsumfaith"])

    # Backup data for abalation study
    # postprocess(annot_folder="data/raw_annot")

    # 7b-label_only
    # main(ZEROSHOT_TEMPLATE.format(input=DOC_FIRST), "data/zeroshot/7b/cogensumm", False, MODEL="scratch/label_only/7b_cogensumm_10ep", max_new_tokens=2, cut='test')

    # 7b-self_inst
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/zs_cot/raw_inst/7b/cogensumm", False, MODEL="scratch/raw_inst/7b_cogensumm_1ep", max_new_tokens=512, cut='test')
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/zs_cot/raw_inst/7b/cogensumm", False, MODEL="scratch/raw_inst/7b_cogensumm_10ep", max_new_tokens=512, cut='test')
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/zs_cot/raw_inst_fix/7b/cogensumm", False, MODEL="scratch/raw_inst_fix_lora_r8/7b_cogensumm_8ep", max_new_tokens=512, cut='test')
     
    # 7b-human_inst
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/zs_cot/human/7b/cogensumm", False, MODEL="scratch/human/7b_cogensumm_1ep", max_new_tokens=512, cut='test')
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/zs_cot/human/7b/cogensumm", False, MODEL="scratch/human/7b_cogensumm_10ep", max_new_tokens=512, cut='test')
    # main(COT_TEMPLATE.format(input=DOC_FIRST), "data/zs_cot/human/7b/frank", False, MODEL="scratch/human/7b_frank_5ep", max_new_tokens=512, cut='test')