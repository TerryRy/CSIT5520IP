# utils.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import json

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
verbalizer = ["entailment", "neutral", "contradiction"]

MODELS = {
    "gpt2_prompt": "gpt2",                    # Causal LM Prompting（经典示例）
    "flan_t5_prompt": "google/flan-t5-base",  # Seq-to-Seq Prompting（效果较好）
    "qwen_prompt": "Qwen/Qwen3-8B",           # 你已有的强模型
    # Fine-tuned BERT 模型（不同数据量）
    "bert_finetuned_1000": "./bert_finetuned_1000",
    "bert_finetuned_5000": "./bert_finetuned_5000",
    "bert_finetuned_10000": "./bert_finetuned_10000",
    "bert_finetuned_50000": "./bert_finetuned_50000"
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
    elif "t5" in model_name_or_path.lower():
        # Seq-to-Seq 模型
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 混合精度
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

def map_t5_output_to_label(output_text):
    """将 T5 生成的文本映射到标签 ID"""
    output_text = output_text.lower().strip()
    
    if "entailment" in output_text or "entail" in output_text:
        return 0  # entailment
    elif "contradiction" in output_text or "contradict" in output_text:
        return 2  # contradiction
    elif "neutral" in output_text:
        return 1  # neutral
    else:
        # 默认返回 neutral
        print(f"Warning: Unexpected T5 output '{output_text}', defaulting to neutral")
        return 1
    
# utils.py 修改部分

def load_few_shot_examples(file_path="few-shots.jsonl"):
    """
    读取 few-shots.jsonl，每条json仅保留premise, hypothesis, label三个字段
    返回格式: [{"premise": "...", "hypothesis": "...", "label": "entailment"}, ...]
    """
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # entry格式: {"genre_name": {"entailment": {...}, "neutral": {...}, "contradiction": {...}}}
                for genre, labels_dict in entry.items():
                    for label, data in labels_dict.items():
                        examples.append({
                            "premise": data["premise"],
                            "hypothesis": data["hypothesis"],
                            "label": data["gold_label"]  # 统一使用 gold_label
                        })
    
    print(f"Loaded {len(examples)} few-shot examples")
    
    # 按标签统计
    label_counts = {}
    for ex in examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    print(f"Label distribution: {label_counts}")
    
    return examples