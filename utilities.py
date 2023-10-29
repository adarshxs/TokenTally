import json
import math
import pandas as pd

def load_base_models():
    with open("models.json", "r") as f:
        return json.load(f)

def load_quantization():
    with open("quantization.json", "r") as f:
        return json.load(f)

def load_gpus():
    with open("gpus.json", "r") as f:
        return json.load(f)

def load_gpu_providers():
    return pd.read_csv('cloud-gpus.csv')

def convert_params(params):
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

def compute_bound_tokens_p_sec(flops_per_token, flops_per_gpu, num_gpus):
    return (flops_per_gpu * num_gpus * 10**12) / (flops_per_token * 10**9)

def memory_bound_tokens_p_sec(memory_bandwidth_per_gpu, flops_per_token, num_gpus):
    return (memory_bandwidth_per_gpu * num_gpus * 10**12) / (flops_per_token * 10**9)

def cost_per_1k_tokens(flops_per_token, flops_per_gpu, num_gpus, cost_per_hour, memory_bandwidth_per_gpu):
    tokens_p_sec_compute = compute_bound_tokens_p_sec(flops_per_token, flops_per_gpu, num_gpus)
    tokens_p_sec_memory = memory_bound_tokens_p_sec(memory_bandwidth_per_gpu, flops_per_token, num_gpus)
    
    cost_p_sec = cost_per_hour / 3600  # cost per second
    
    cost_p_token_compute = cost_p_sec / tokens_p_sec_compute
    cost_p_token_memory = cost_p_sec / tokens_p_sec_memory
    
    cost_p_1k_tokens_compute = cost_p_token_compute * 1000
    cost_p_1k_tokens_memory = cost_p_token_memory * 1000
    
    return cost_p_1k_tokens_compute, cost_p_1k_tokens_memory