from utils import load_jsonl
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets, interleave_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import pandas as pd
import os

# COT_TEXT = "Explain your reasoning step by step first, and then answer (yes or no) the question in the end."
# POSTHOC_TEXT = "Answer (yes or no) the question first, and then provide an explanation."
# BASELINE_TEXT = "Answer (yes or no):"

YES_TEXT = "Yes, the hypothesis is true."
NO_TEXT = "No, the hypothesis is not true."

PROMPT = \
"""Is the hypothesis true based on the premise? Give your explanation afterwards.

Premise: 
{article}

Hypothesis:
{summary}
"""

# MODEL_ID = "google/flan-t5-xxl"
# MODEL_ID = "google/flan-t5-xl"
# MODEL_ID = "philschmid/flan-t5-xxl-sharded-fp16"
MODEL_ID = "./flan-t5-xxl-bf16"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
PROMPT_LEN = len(tokenizer.encode(PROMPT, add_special_tokens=False))
PROMPT_MAXLEN = 1556

def process_message(example, tokenizer, add_response, add_reason):
    summary_len = len(tokenizer.encode(example["summary"], add_special_tokens=False))
    article_tokens = tokenizer.encode(example["article"], add_special_tokens=False)
    article = example["article"]
    if len(article_tokens) + summary_len + PROMPT_LEN > PROMPT_MAXLEN:
        article = tokenizer.decode(article_tokens[:PROMPT_MAXLEN-PROMPT_LEN-summary_len])
    instruction = PROMPT.format(summary=example["summary"], article=article)
    answer = ""
    if add_response:
        answer = (YES_TEXT if example["label"] == 0 else NO_TEXT)
        if add_reason:
            reason = example["reason"].replace("article", "premise")
            reason = reason.replace("summary", "hypothesis")
            answer = (YES_TEXT if example["label"] == 0 else NO_TEXT) + "\n\n" + reason
    model_inputs = tokenizer(instruction, text_target=answer, max_length=PROMPT_MAXLEN, truncation=True)
    if (not add_reason) or len(example["reason"]) == 0:
        model_inputs["labels"] = model_inputs["labels"][:-1]
    return model_inputs

def load_summae(file, add_response=True, add_reason=True):
    summae = load_dataset("json", data_files=file, split="train")
    summae = summae.map(lambda x: process_message(x, tokenizer, add_response, add_reason), remove_columns=["label", "reason", "summary", "article", "origin", "dataset"])
    return summae

def postprocess_text(preds, labels):
    def posthoc_parse(input:str):
        from template import swap_punctuation
        generation_part = input.partition("\n\n")[0].lower()
        words = swap_punctuation(generation_part).split()
        # assert ("not" in words) or ("no" in words) or ("yes" in words)
        return 1 if ("yes" in words) else 0
    preds = [posthoc_parse(pred.strip()) for pred in preds]
    labels = [posthoc_parse(label.strip()) for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # import pdb
    # pdb.set_trace()

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = {"ba": balanced_accuracy_score(decoded_labels, decoded_preds),
              "f1": f1_score(decoded_labels, decoded_preds)}
    return result

if __name__ == '__main__':
    GRADIENT_CHECKPOINTING = False
    LORA = True
    LOAD_IN_8BIT = False
    summae_train = load_summae("data/merge/train_filtered.jsonl", add_reason=False)
    summae_dev = load_summae("data/merge/val_filtered.jsonl", add_reason=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, 
                                                  device_map="auto",
                                                  torch_dtype="auto",
                                                  load_in_8bit=LOAD_IN_8BIT)
                                                #   attn_implementation="flash_attention_2")
    

    if LORA:
        if LOAD_IN_8BIT:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
        peft_config = LoraConfig(
            task_type = TaskType.SEQ_2_SEQ_LM,
            r = 16,
            lora_alpha = 32,
            lora_dropout = 0.05,
            inference_mode = False,
            bias='lora_only'
        )
        model = get_peft_model(model, peft_config)
        if GRADIENT_CHECKPOINTING and not LOAD_IN_8BIT:
            model = model._prepare_model_for_gradient_checkpointing(model)
        model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps", 
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps = 100,
        output_dir="./t5_output",
        optim="paged_adamw_8bit",
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="constant_with_warmup",
        # fp16=True,
        bf16=True,
        tf32=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        predict_with_generate=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_ID)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=summae_train,
        eval_dataset=summae_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    import pdb
    pdb.set_trace()

    from functools import partial
    trainer.evaluate = partial(trainer.evaluate, max_new_tokens=1)
    # print(trainer.evaluate())
    trainer.train()#resume_from_checkpoint=True)
