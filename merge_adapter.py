from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_PATH = "output/label_only/cogensumm/checkpoint-270"
DUMP_PATH = "scratch/7b_cogensumm_1ep"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(DUMP_PATH)