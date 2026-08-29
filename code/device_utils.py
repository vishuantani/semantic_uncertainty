import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()
# fp16 is unreliable on MPS for logit/log-prob math; fp32 is cheap at these sizes
DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32

def empty_cache():
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE.type == 'mps':
        torch.mps.empty_cache()