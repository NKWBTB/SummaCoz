from sklearn.metrics import balanced_accuracy_score
from llama2_gen import load_jsonl
from template import zeroshot_parse, cot_parse
import os

raw_folder = "data/raw"
data_names = ["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"]
cut = "test"

def main(test_folder, parse_fun):
    for name in data_names:
        print(name, cut)
        input_path = os.path.join(raw_folder, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        gt = [line["label"] for line in data]

        pred_data = load_jsonl(os.path.join(test_folder, "_".join([name, cut])+'.jsonl'))
        y_pred = [parse_fun(line[0]["generated_text"]) for line in pred_data]

        print("BA:", balanced_accuracy_score(gt, y_pred))

if __name__ == "__main__":
    # Zero-shot
    main("data/zeroshot", zeroshot_parse)

    # COT-ZS
    # main("data/zs_cot", cot_parse)