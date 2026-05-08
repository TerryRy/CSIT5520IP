# utils.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import json

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
MODELS = {
    "bert_prompt": "bert-base-uncased",
    "deberta_prompt": "microsoft/deberta-base",
    "qwen_prompt": "Qwen/Qwen3-8B",          # 改成你的实际路径
    "bert_finetuned": "./bert_finetuned"     # fine-tune 后路径
}

def load_multinli_jsonl(file_path: str) -> pd.DataFrame:
    """加载课程提供的 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append({
                    'premise': item['sentence1'],
                    'hypothesis': item['sentence2'],
                    'label': item['gold_label'],           # "entailment", "neutral", "contradiction"
                    'gold_label': item['gold_label'],
                    'genre': item.get('genre', ''),
                    'pairID': item.get('pairID', '')
                })
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples from {file_path}")
    print("Label distribution:\n", df['label'].value_counts())
    return df

def load_model_and_tokenizer(model_name_or_path, task="classification"):
    """统一加载模型（支持 fine-tuned 路径）"""
    print(f"Loading model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if task == "classification":
        # 支持 fine-tuned 模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            num_labels=3, 
            id2label=id2label, 
            label2id=label2id,
            ignore_mismatched_sizes=True  # 防止 head 维度问题
        )
    else:  # Causal LM (Qwen)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
            device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer