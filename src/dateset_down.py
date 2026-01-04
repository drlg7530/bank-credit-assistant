"""
huggingface下载客户贷款意向数据集
"""
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Ravichandrachilde/loan-prediction-dataset", cache_dir='./data/customer')

print(ds)
