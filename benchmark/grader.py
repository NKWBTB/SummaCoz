import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from utils import load_jsonl
import re
from nltk import sent_tokenize
import string
from tqdm import tqdm
import template

RUN_LOCAL = True

if RUN_LOCAL:
    MODEL = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                torch_dtype=torch.float16,
                                                trust_remote_code=True,
                                                attn_implementation="flash_attention_2",
                                                device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

def llm_generate(input_txt):
    sequences = pipeline(
            input_txt,
            max_new_tokens=16,
            num_return_sequences=1,
            pad_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            return_full_text=False
    )
    return sequences[0]['generated_text']

def process_human_reasoning(human_reason):
    human_reason = re.sub(r'\n+', '\n', human_reason).strip()
    # items = sent_tokenize(human_reason)
    items = human_reason.split("\n")
    items = [re.sub(r'^\d+\.\s*', '', item).strip() for item in items]
    # items = [item for item in items if len(item) > 0]
    return items

def swap_punctuation(input:str):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return input.translate(translator)

def answer_parse(input:str):
    words = swap_punctuation(input.lower()).split()
    try: 
        assert ("not" in words) or ("no" in words) or ("yes" in words)
    except:
        print(words)
        import pdb
        pdb.set_trace()
    return 0 if ("no" in words) or ("not" in words) else 1

def run_llm_metric(reference, prediction, template, model_field="reason"):
    scores = []
    for idx, human in enumerate(tqdm(reference)):
        pred = prediction[idx][model_field]
        # print(predicted)
        ground_truth = process_human_reasoning(human["human_reason"])
        points = []
        for item in ground_truth:
            message = [{"role": "user", "content": template.format(input_reasoning=pred, reference=item)}]
            answer = llm_generate(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            points.append(answer_parse(answer))
        # print(points)
        scores.append(sum(points)/len(points))
    return scores

if __name__ == "__main__":
    gt = load_jsonl("benchmark_v_0_5.jsonl")
    pred = load_jsonl("llama2-13b-posthoc.jsonl")
    run_llm_metric(reference=gt, prediction=pred, template=template.GRADER_TEMPLATE)
