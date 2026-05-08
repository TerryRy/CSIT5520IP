# utils.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import json

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
verbalizer = ["Entailment", "Neutral", "Contradiction"]

MODELS = {
    "gpt2_prompt": "gpt2",                    # Causal LM Prompting（经典示例）
    "flan_t5_prompt": "google/flan-t5-base",  # Seq-to-Seq Prompting（效果较好）
    "qwen_prompt": "Qwen/Qwen3-8B",           # 你已有的强模型
    "bert_finetuned": "./bert_finetuned"      # Fine-tuning 代表
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

def load_model_and_tokenizer(model_name_or_path, task="causal"):
    """明确区分模型类型"""
    print(f"Loading model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "bert" in model_name_or_path.lower() or "deberta" in model_name_or_path.lower() or "finetuned" in model_name_or_path:
        # Classification 模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=3, ignore_mismatched_sizes=True
        )
    else:
        # Causal LM (GPT-2, Qwen 等)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if "qwen" in model_name_or_path.lower() else None
        )
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer