# run_hallucination.py
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nliEvaluator import predict_prompting
from utils import load_model_and_tokenizer, MODELS
from tqdm import tqdm

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")

# model_key = "qwen_prompt"   # ← 改成你想用的模型
shot = 0

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")

for model_key in MODELS:
    print(f"\n=== Running Hallucination Detection with {model_key} (shot={shot}) ===")
    
    model, tokenizer = load_model_and_tokenizer(
        MODELS[model_key], 
        task="causal" if "qwen" in model_key else "classification"
    )
    
    results = []
    for ex in tqdm(dataset.select(range(300))):   # 先用 300 条测试，全部跑太慢
        wiki_text = ex['wiki_bio_text']
        for sent, anno in zip(ex['gpt3_sentences'], ex['annotation']):
            pred_label = predict_prompting(model, tokenizer, wiki_text, sent, shot=shot)
            gold = 0 if anno == 0 else 1   
            
            results.append({
                "sentence": sent[:300],
                "gold": gold,
                "pred": pred_label,
                "raw_anno": anno
            })
    
    df = pd.DataFrame(results)
    df.to_csv(f"hallucination_results_{model_key}_shot{shot}.csv", index=False)
    
    acc = accuracy_score(df.gold, df.pred)
    p, r, f1, _ = precision_recall_fscore_support(df.gold, df.pred, average='binary', zero_division=0)
    
    print(f"Results for {model_key}:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1       : {f1:.4f}\n")