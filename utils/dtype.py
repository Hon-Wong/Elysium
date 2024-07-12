import torch


DTYPE_MAPPING = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "fp32": torch.float32,
}
