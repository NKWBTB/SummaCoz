gpu_device_map = {
        'model.embed_tokens': 0, 
        'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 
        'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 
        'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 
        'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 
        'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 
        'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 
        'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 
        'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 
        'model.layers.32': 2, 'model.layers.33': 2, 'model.layers.34': 2, 'model.layers.35': 2, 
        'model.layers.36': 2, 'model.layers.37': 2, 'model.layers.38': 2, 'model.layers.39': 2, 
        'model.norm': 2, 'lm_head': 2}

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EPOCH = 20
LR = 1e-4
BATCH_SIZE = 1
GRADIENT_CHECKPOINTING=True
GRAD_ACC_STEPS = 8