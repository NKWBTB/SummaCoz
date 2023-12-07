import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import config as CFG
from instruct_data import InstructData
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from template import cot_parse
import argparse
from sklearn.metrics import balanced_accuracy_score, f1_score
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', default="output/human/")
parser.add_argument('--data_name', default="cogensumm", choices=["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"])
parser.add_argument('--lora', default=True, action="store_true")

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, padding_side='left')
tokenizer.pad_token_id = tokenizer.bos_token_id

def main():
    args = parser.parse_args()
    print(args)

    data = InstructData(tokenizer, label_only=False, cot_folder="data/final_human") #, debug=True)
    train_data = data.load_train(args.data_name)
    val_data = data.load_val(args.data_name, load_label=False)
    print(len(train_data), len(val_data))

    model = AutoModelForCausalLM.from_pretrained(CFG.MODEL_NAME,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16)
    if args.lora:
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            r = 16,
            lora_alpha = 16,
            lora_dropout = 0.1,
            inference_mode = False,
            bias='lora_only'
        )
        model = get_peft_model(model, peft_config)
        if CFG.GRADIENT_CHECKPOINTING:
            model = model._prepare_model_for_gradient_checkpointing(model)
        model.print_trainable_parameters()

    def compute_metrics(eval_pred, data):
        logits, labels = eval_pred
        results = {"ba": 0}
        logits[logits == -100] = tokenizer.eos_token_id
        txt = tokenizer.batch_decode(logits, skip_special_tokens=True)
        pred = [cot_parse(x, default=0) for x in txt]
        y_gt = data["label"]
        results = {"ba": balanced_accuracy_score(y_gt, pred),
               "f1": f1_score(y_gt, pred, average="macro")}
        return results

    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.output_folder, args.data_name), 
                                    save_strategy="epoch",
                                    evaluation_strategy="epoch",
                                    logging_strategy="epoch",
                                    num_train_epochs=CFG.EPOCH,
                                    optim="paged_lion_8bit",
                                    learning_rate=CFG.LR,
                                    per_device_train_batch_size=CFG.BATCH_SIZE,
                                    per_device_eval_batch_size=CFG.BATCH_SIZE * 3,
                                    tf32=True,
                                    fp16=True,
                                    gradient_checkpointing=CFG.GRADIENT_CHECKPOINTING,
                                    gradient_accumulation_steps=CFG.GRAD_ACC_STEPS,
                                    predict_with_generate=True,
                                    metric_for_best_model='ba',)
    model.config.pad_token_id = tokenizer.bos_token_id
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=lambda x : compute_metrics(x, data.val_set),
    )
    trainer.evaluate = partial(trainer.evaluate, max_new_tokens = 25, pad_token_id = tokenizer.bos_token_id)
    trainer.train()
    # trainer.evaluate()

if __name__ == '__main__':
    main()