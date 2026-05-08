# bias_evaluator.py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np

# ================== 配置区域（易修改） ==================
MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "deberta": "microsoft/deberta-base",
    # "roberta": "roberta-base",   # 可添加
}

CHOSEN_DOMAIN = "nationality"      # 推荐选项: nationality, gender, race-color, religion 等
SAMPLE_SIZE = 80                   # 如果 domain 样本多，可采样
# =======================================================

def pseudo_log_likelihood(model, tokenizer, sentence: str):
    """计算 Pseudo-Log-Likelihood (PLL)"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)
    total_log_prob = 0.0
    mask_token_id = tokenizer.mask_token_id

    with torch.no_grad():
        for i in range(1, len(input_ids[0]) - 1):   # 不 mask 首尾特殊 token
            masked_input = input_ids.clone()
            original_token_id = masked_input[0, i].item()
            masked_input[0, i] = mask_token_id

            outputs = model(masked_input)
            logits = outputs.logits[0, i]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_prob = log_probs[original_token_id].item()
            total_log_prob += token_log_prob

    return total_log_prob


def evaluate_bias(model_name: str):
    print(f"\nEvaluating bias for {model_name} on domain: {CHOSEN_DOMAIN}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    dataset = load_dataset("nyu-mll/crows_pairs", split="test")
    
    results = []
    stereo_count = 0
    total = 0
    
    # 过滤所选 domain
    domain_data = [ex for ex in dataset if ex['bias_type'] == CHOSEN_DOMAIN]
    if len(domain_data) > SAMPLE_SIZE:
        import random
        random.seed(42)
        domain_data = random.sample(domain_data, SAMPLE_SIZE)
    
    for ex in tqdm(domain_data):
        sent_more = ex['sent_more']      # stereotypical
        sent_less = ex['sent_less']      # anti-stereotypical
        
        pll_more = pseudo_log_likelihood(model, tokenizer, sent_more)
        pll_less = pseudo_log_likelihood(model, tokenizer, sent_less)
        
        if pll_more > pll_less:
            stereo_count += 1
        total += 1
        
        results.append({
            "sent_more": sent_more,
            "sent_less": sent_less,
            "pll_more": pll_more,
            "pll_less": pll_less,
            "bias_type": ex['bias_type'],
            "stereotypical_preferred": pll_more > pll_less
        })
    
    bias_score = (stereo_count / total * 100) if total > 0 else 0
    print(f"Total pairs: {total}")
    print(f"Stereotype Score: {bias_score:.2f}% (lower is better, 50% = unbiased)")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(f"bias_results_{model_name.split('/')[-1]}_{CHOSEN_DOMAIN}.csv", index=False)
    
    return bias_score, df


if __name__ == "__main__":
    all_results = []
    for name, model_path in MODEL_NAMES.items():
        score, df = evaluate_bias(model_path)
        all_results.append({
            "model": name,
            "domain": CHOSEN_DOMAIN,
            "num_pairs": len(df),
            "stereotype_score": score
        })
    
    summary = pd.DataFrame(all_results)
    summary.to_csv(f"bias_summary_{CHOSEN_DOMAIN}.csv", index=False)
    print("\n=== Summary ===")
    print(summary)