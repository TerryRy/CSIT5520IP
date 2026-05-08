# run_nli.py
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score
from nliEvaluator import predict
from tqdm import tqdm
from utils import load_model_and_tokenizer, MODELS, load_multinli_jsonl, id2label

# ================== 配置区域 ==================


shots = [0]
# =============================================

# 加载课程提供的两个数据集
matched = load_multinli_jsonl("dev_matched_sampled-1.jsonl")     # ← 修改为你的实际文件名
mismatched = load_multinli_jsonl("dev_mismatched_sampled-1.jsonl") # ← 修改为你的实际文件名

def evaluate_dataset(df, model_key, shot=0):
    model, tokenizer = load_model_and_tokenizer(MODELS[model_key], 
                                                task="causal" if "qwen" in model_key else "classification")
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred = predict(model, tokenizer, row['premise'], row['hypothesis'], model_key, shot=shot)
        # print(f"{model_key} (shot={shot}) pred: {id2label[pred]} gold: {row['label']}") 
        preds.append(id2label[pred])
    
    acc = accuracy_score(df['label'], preds)
    print(f"{model_key} (shot={shot}) Accuracy: {acc:.4f}")
    return preds

# 运行所有实验并保存
results = []
for model_key in MODELS:
    for shot in shots:
        if "finetuned" in model_key and shot > 0: continue  # finetune不需要few-shot
        print(f"\nEvaluating {model_key} shot={shot}")
        matched_pred = evaluate_dataset(matched, model_key, shot)
        mismatched_pred = evaluate_dataset(mismatched, model_key, shot)
        
        results.append({
            "model": model_key,
            "shot": shot,
            "matched_acc": accuracy_score(matched['label'], matched_pred),
            "mismatched_acc": accuracy_score(mismatched['label'], mismatched_pred)
        })

summary_df = pd.DataFrame(results)
summary_df.to_csv("nli_2.1_results_summary.csv", index=False)
summary_df.to_markdown("nli_2.1_results.md")   # 生成 Markdown 方便写报告
print("All 2.1 results saved!")
print(summary_df.round(4))