import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="/kaggle/temp_model",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="/kaggle/temp_model",
    device_map="auto",         
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    repetition_penalty=1.15
)
