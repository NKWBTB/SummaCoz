from utils import load_jsonl
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, pipeline
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import pandas as pd
import os

COT_TEXT = "Explain your reasoning step by step first, and then answer (yes or no) the question in the end."
POSTHOC_TEXT = "Answer (yes or no) the question first, and then provide an explanation."
BASELINE_TEXT = "Answer (yes or no):"

YES_TEXT = "Yes, the hypothesis text is faithful."
NO_TEXT = "No, the hypothesis text is not faithful."

NLI_PROMPT = \
"""You are given a premise passage:
<premise> {premise} </premise>

Decide if the following hypothesis is faithful given the above premise:
<hypothesis> {hypothesis} </hypothesis>

{answer_prompt}
"""

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

def process_message(example, tokenizer, answer_prompt, add_response, add_reason):
    messages = [{"role": "user", "content": NLI_PROMPT.format(premise=example["premise"], hypothesis=example["hypothesis"], answer_prompt=answer_prompt)}]
    if add_response:
        answer = (YES_TEXT if example["label"] == 0 else NO_TEXT)
        if add_reason:
            if answer_prompt == COT_TEXT:
                answer = example["reason"] + "\n\n" + (YES_TEXT if example["label"] == 0 else NO_TEXT)
            elif answer_prompt == POSTHOC_TEXT:
                answer = (YES_TEXT if example["label"] == 0 else NO_TEXT) + "\n\n" + example["reason"]
        messages.append({"role": "assistant", "content": answer})
    example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if add_response and (len(example["reason"].strip()) == 0 or (not add_reason)):
        example["prompt"] = example["prompt"].replace("</s>", "")
    return example

def load_esnli(split, add_response=True, add_reason=True, answer_style=POSTHOC_TEXT):
    esnli = load_dataset("esnli", split=split)
    if split == "train":
        esnli = esnli.shard(num_shards=20, index=0) # Take 1/20 of the dataset
    esnli = esnli.rename_column("explanation_1", "reason")
    esnli = esnli.map(lambda x: process_message(x, tokenizer, answer_style, add_response, add_reason))
    return esnli

def load_anli(split, add_response=True, add_reason=True, answer_style=POSTHOC_TEXT):
    anli = load_dataset("anli", split=split)
    anli = anli.filter(lambda example: len(example["reason"]) > 0)
    anli = anli.map(lambda x: process_message(x, tokenizer, answer_style, add_response, add_reason))
    return anli

def load_summae(file, add_response=True, add_reason=True, answer_style=POSTHOC_TEXT):
    from datasets import Features, Value, ClassLabel
    features = Features({'hypothesis': Value(dtype='string'), 
                        'premise': Value(dtype='string'), 
                        'reason': Value(dtype='string'), 
                        "origin": Value(dtype='string'),
                        "dataset": Value(dtype='string'),
                        'label': ClassLabel(names=['entailment', 'neutral', 'contradiction'])
    })
    summae = load_dataset("json", data_files=file, split="train", features=features)
    summae = summae.map(lambda x: process_message(x, tokenizer, answer_style, add_response, add_reason))
    summae = summae.filter(lambda example: len(example["prompt"].split()) <= 2000)
    return summae


if __name__ == '__main__':
    GRADIENT_CHECKPOINTING = True
    
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                                 torch_dtype="auto",
                                                 device_map="auto",
                                                 attn_implementation="flash_attention_2")
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # anli_train = concatenate_datasets([load_anli("train_r1"), load_anli("train_r2"), load_anli("train_r3")])
    # anli_train = anli_train.remove_columns(["uid"])
    # esnli_train = load_esnli("train")
    # esnli_train = esnli_train.remove_columns(["explanation_2", "explanation_3"])
    summae_train = load_summae("data/merge/train.jsonl", add_reason=False, answer_style=BASELINE_TEXT)
    summae_train = summae_train.remove_columns(["origin", "dataset"])
    train_data = summae_train
    # train_data = interleave_datasets([anli_train, esnli_train, summae_train], probabilities=[0.3, 0.4, 0.3], stopping_strategy="all_exhausted")
    
    # esnli_dev = load_esnli("validation", add_reason=False)
    # esnli_dev = esnli_dev.remove_columns(["explanation_2", "explanation_3"])
    # anli_dev = concatenate_datasets([load_anli("dev_r1", add_reason=False), load_anli("dev_r2", add_reason=False), load_anli("dev_r3", add_reason=False)])
    # anli_dev = anli_dev.remove_columns(["uid"])
    summae_dev = load_summae("data/merge/val.jsonl", add_reason=False, answer_style=BASELINE_TEXT)
    summae_dev = summae_dev.remove_columns(["origin", "dataset"])
    # eval_data = {"esnli": esnli_dev, "anli": anli_dev, "summae": summae_dev}
    eval_data = summae_dev
    
    collator = DataCollatorForCompletionOnlyLM(response_template=[13, 733, 28748, 16289, 28793], tokenizer=tokenizer)

    args = TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps", 
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps = 100,
        output_dir="./sft_output",
        optim="paged_lion_8bit",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        # fp16=True,
        bf16=True,
        tf32=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING)

    trainer = SFTTrainer(
        model,
        args = args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        max_seq_length=4000,
        packing=False,
        dataset_text_field="prompt",
        peft_config=peft_config
    )

    # import pdb
    # pdb.set_trace()

    trainer.train()
    # print(trainer.evaluate())