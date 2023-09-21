import json
from tqdm import tqdm
import os.path
from difflib import SequenceMatcher
from llama2_gen import load_jsonl, dump2jsonl

def sequence_match(text_1, text2):
    s = SequenceMatcher(None, text_1, text2)
    if s.real_quick_ratio() > 0.95:
        if s.quick_ratio() > 0.95:
            if s.ratio() > 0.95:
                return True
    return False

def check_overlap(test_documents, val_file, use_joblib=False):
    """
    This function check if the test file and validation file share the same document
    :param test_documents: list of test documents
    :param val_file: Path of validation file
    :return overlap_index: The duplicate index in validation file
    :return count: number of overlap pairs
    """
    if use_joblib:
        from joblib import Parallel, delayed
    
    val_list = load_jsonl(val_file)
    print(f"val file has {len(val_list)} elements")

    count = 0
    overlap_index = []
    for i in tqdm(range(len(val_list))):
        doc_val = val_list[i]['document']
        # Sequential
        if not use_joblib:
            for doc_test in test_documents:
                if sequence_match(doc_val, doc_test):
                    count += 1
                    overlap_index.append(i)
                    break
        else:
            result = Parallel(n_jobs=15)(delayed(sequence_match)(doc_val, doc_test) for doc_test in test_documents)
            if any(result):
                count += 1
                overlap_index.append(i)
    return overlap_index, count


def delete_overlap_item(val_file, val_file_out, index2del):
    """
    This function delete the duplicate items from validation file based on the given index
    and generate the new validation file without duplicate
    :param val_file: The path of validation file
    :param val_file_out: The path of validation file after delete duplicates
    :param index2del: The duplicate's index
    """
    with open(val_file, 'r', encoding="utf-8") as f:
        val_list = f.readlines()

    with open(val_file_out, 'w', encoding="utf-8") as f:
        outputs = [val_list[i] for i in range(len(val_list)) if i not in index2del]
        f.write(''.join(outputs))

    print(f"val file has {len(val_list)} items, after delete overlap it has {len(outputs)} items")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_names = ['cogensumm', 'factcc', 'frank', 'polytope', 'summeval', 'xsumfaith']
    DATA_FOLDER = 'data/'
    RAW_FOLDER = 'raw/'
    COT_FOLDER = 'merge/'
    DUMP_FOLDER = 'final/'
    test_dict = {}
    for name in data_names:
        test_file = os.path.join(DATA_FOLDER, RAW_FOLDER, name + '_test.jsonl')
        test_set = load_jsonl(test_file)
        print("original test size", len(test_set))
        test_docs = set([sample["document"] for sample in test_set])
        test_dict[name] = test_docs

    # Process TrainNval1
    for test_name in data_names:
        print("###TEST", test_name)
        for target_name in data_names:
            path_val_cot = os.path.join(DATA_FOLDER, COT_FOLDER, target_name + '_val.jsonl')
            output_folder = os.path.join(DATA_FOLDER, DUMP_FOLDER, test_name)
            path_val_dump = os.path.join(output_folder, target_name + '_val.jsonl')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            overlap_index = []
            print("@@@VAL", target_name)
            if target_name != test_name:
                overlap_index, count = check_overlap(test_dict[test_name], path_val_cot, use_joblib=True)
            else:
                count = 0
            print(count)
            delete_overlap_item(path_val_cot, path_val_dump, overlap_index)
    