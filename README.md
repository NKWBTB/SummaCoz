# SummaCoz

## File structure
- ``train.py``: training script
- ``testing.py``: testing script for Balanced Accuracy on test sets
- ``template.py``: prompt template and functions for parsing the LLM output
- ``merge_adapter.py``: merge LoRA adapter to the original model
- ``llama2_gen.py``: generate with llama2
- ``instruct_data.py``: dataset utility for training, merge the generated reasoning back to the dataset
- ``config.py``: model and hyperparameters setting
- ``check_overlap.py``: check the overlap between different datasets, remove the duplications for training
