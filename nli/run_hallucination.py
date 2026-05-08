# run_hallucination_detection.py
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from datasets import load_dataset
from utils import load_model_and_tokenizer, id2label
from nliEvaluator import predict as nli_predict
import json

# ================== 配置 ==================
# 根据2.1的结果选择最好的配置
MODELS_FOR_HALLUCINATION = {
    "qwen_prompt_0shot": ("Qwen/Qwen3-8B", "qwen_prompt", 0),  # 使用zero-shot
    "flan_t5_prompt_0shot": ("google/flan-t5-base", "flan_t5_prompt", 0),
    "bert_finetuned_10000": ("./bert_finetuned_10000", "bert_finetuned", 0),  # 最佳bert
}
# ========================================

def load_hallucination_data(split="test"):
    """加载数据（只取test集）"""
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    
    data = []
    for item in dataset[split]:
        wiki_bio_text = item["wiki_bio_text"]
        gpt3_sentences = item["gpt3_sentences"]
        annotations = item["annotation"]
        
        for sentence, annotation in zip(gpt3_sentences, annotations):
            binary_label = 1 if annotation != 0 else 0  # 0=factual, 1=hallucination
            data.append({
                "premise": wiki_bio_text,
                "hypothesis": sentence,
                "binary_label": binary_label
            })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} sentence-level samples")
    print(f"Label distribution:\n{df['binary_label'].value_counts()}")
    return df

def nli_predict_with_score(model, tokenizer, premise, hypothesis, model_key):
    """获取NLI预测和概率分数"""
    if "gpt" in model_key.lower() or "qwen" in model_key.lower():
        prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            # 计算三个标签的概率
            scores = {}
            for word in ["entailment", "neutral", "contradiction"]:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) > 1:
                    score = sum(next_token_logits[tid].item() for tid in token_ids) / len(token_ids)
                else:
                    score = next_token_logits[token_ids[0]].item()
                scores[word] = score
            
            probs = torch.softmax(torch.tensor(list(scores.values())), dim=-1).tolist()
            score_dict = dict(zip(["entailment", "neutral", "contradiction"], probs))
            
            pred_id = torch.argmax(torch.tensor(list(scores.values()))).item()
            pred_label = id2label[pred_id]
            
    elif "t5" in model_key.lower():
        prompt = f"premise: {premise} hypothesis: {hypothesis} label:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
        from utils import map_t5_output_to_label
        pred_id = map_t5_output_to_label(generated_text)
        pred_label = id2label[pred_id]
        score_dict = {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2}  # T5不输出概率，用占位
        
    else:
        # BERT
        inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
            score_dict = dict(zip(["entailment", "neutral", "contradiction"], probs))
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            pred_label = id2label[pred_id]
    
    return pred_label, score_dict

def detect_hallucination(nli_label, nli_scores, method="label_only", threshold=0.5):
    """将NLI输出转换为幻觉检测结果"""
    if method == "label_only":
        # entailment → factual (0), 其他 → hallucination (1)
        return 0 if nli_label == "entailment" else 1
    
    elif method == "score_based":
        # 使用entailment概率
        return 0 if nli_scores.get("entailment", 0) >= threshold else 1

def evaluate_hallucination_detection(df, model_key, model, tokenizer, method="label_only", threshold=0.5):
    """评估幻觉检测性能"""
    predictions = []
    ground_truth = df["binary_label"].tolist()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # NLI预测
        nli_label, nli_scores = nli_predict_with_score(model, tokenizer, row["premise"], row["hypothesis"], model_key)
        
        # 转换为幻觉检测
        pred = detect_hallucination(nli_label, nli_scores, method, threshold)
        predictions.append(pred)
    
    # 计算指标
    acc = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="binary", pos_label=1
    )
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "method": method,
        "threshold": threshold
    }

# 主程序
print("Loading hallucination data...")
test_df = load_hallucination_data(split="test")

results = []
for model_key, (model_path, model_type, shot) in MODELS_FOR_HALLUCINATION.items():
    print(f"\n{'='*50}")
    print(f"Evaluating {model_key}")
    print('='*50)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 方法1: 仅用标签
    result_label = evaluate_hallucination_detection(
        test_df, model_key, model, tokenizer, 
        method="label_only"
    )
    result_label["model"] = model_key
    results.append(result_label)
    print(f"Label-only: {result_label}")
    
    # 方法2: 基于分数（尝试不同阈值）
    for threshold in [0.3, 0.5, 0.7]:
        result_score = evaluate_hallucination_detection(
            test_df, model_key, model, tokenizer,
            method="score_based", threshold=threshold
        )
        result_score["model"] = model_key
        results.append(result_score)
        print(f"Score-based (threshold={threshold}): {result_score}")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("hallucination_detection_results.csv", index=False)
results_df.to_markdown("hallucination_detection_results.md")
print("\nAll results saved!")
print(results_df.round(4))