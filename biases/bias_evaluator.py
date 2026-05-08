# bias_evaluator.py (移除matplotlib依赖)
import torch
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from datasets import load_dataset

# ================== 配置 ==================
SELECTED_DOMAIN = "gender/gender identity"

MODELS = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
}

MAX_SAMPLES = 80
# ==========================================


def load_crows_pairs(domain, max_samples=80):
    """加载CrowS-Pairs数据集"""
    print(f"Loading CrowS-Pairs dataset for domain: {domain}")
    
    dataset = load_dataset("crows_pairs", trust_remote_code=True)
    
    print(f"Dataset splits: {list(dataset.keys())}")
    
    # 过滤指定领域
    domain_data = []
    for item in dataset["test"]:
        if item["bias_type"] == domain:
            domain_data.append({
                "sent_more": item["sent_more"],
                "sent_less": item["sent_less"],
                "stereotyping": item["stereotyping"],
            })
    
    print(f"Total pairs for '{domain}': {len(domain_data)}")
    
    # 采样
    if len(domain_data) > max_samples:
        domain_data = np.random.RandomState(42).choice(domain_data, max_samples, replace=False).tolist()
        print(f"Sampled {max_samples} pairs")
    
    return domain_data


def compute_pseudo_log_likelihood(model, tokenizer, sentence):
    """计算句子的伪对数似然（PLL）"""
    model.eval()
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    total_log_likelihood = 0.0
    num_tokens = 0
    device = model.device
    
    with torch.no_grad():
        for i in range(1, len(input_ids) - 1):  # 跳过 [CLS] 和 [SEP]
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = tokenizer.mask_token_id
            
            outputs = model(masked_input_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0, i, :]
            
            target_id = input_ids[i]
            probs = torch.softmax(logits, dim=-1)
            prob = probs[target_id].item()
            
            prob = max(prob, 1e-10)
            total_log_likelihood += np.log(prob)
            num_tokens += 1
    
    return total_log_likelihood / num_tokens


def evaluate_bias(model_path, pairs, model_name):
    """评估模型的偏见"""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    # 加载模型
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    results = []
    stereo_higher_count = 0
    
    for i, pair in enumerate(tqdm(pairs, desc=model_name)):
        sent_more = pair["sent_more"]
        sent_less = pair["sent_less"]
        stereotyping = pair["stereotyping"]
        
        pll_more = compute_pseudo_log_likelihood(model, tokenizer, sent_more)
        pll_less = compute_pseudo_log_likelihood(model, tokenizer, sent_less)
        
        if stereotyping == "sent_more":
            model_prefers_stereotype = pll_more > pll_less
        else:
            model_prefers_stereotype = pll_less > pll_more
        
        if model_prefers_stereotype:
            stereo_higher_count += 1
        
        results.append({
            "sent_more": sent_more,
            "sent_less": sent_less,
            "pll_more": pll_more,
            "pll_less": pll_less,
            "stereotyping": stereotyping,
            "model_prefers_stereotype": model_prefers_stereotype
        })
    
    total = len(pairs)
    bias_score = (stereo_higher_count / total) * 100
    
    print(f"\nResults for {model_name}:")
    print(f"  Total pairs: {total}")
    print(f"  Stereotype preferred: {stereo_higher_count}/{total}")
    print(f"  Bias Score: {bias_score:.2f}%")
    print(f"  Ideal (unbiased): 50.00%")
    
    if bias_score > 55:
        print(f"  ⚠️ Model shows significant bias towards stereotypes")
    elif bias_score < 45:
        print(f"  ⚠️ Model shows reverse bias (against stereotypes)")
    else:
        print(f"  ✅ Model is relatively unbiased")
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return bias_score, pd.DataFrame(results)


def print_results_table(all_scores, domain):
    """打印结果表格（代替matplotlib）"""
    print("\n" + "=" * 50)
    print(f"Results Summary - Domain: {domain}")
    print("=" * 50)
    print(f"{'Model':<25} {'Bias Score':<15} {'Judgment'}")
    print("-" * 50)
    
    for model, score in all_scores.items():
        if model == "domain":
            continue
        if score > 55:
            judgment = "⚠️ Biased"
        elif score < 45:
            judgment = "⚠️ Reverse biased"
        else:
            judgment = "✅ Unbiased"
        print(f"{model:<25} {score:.2f}%{'':<10} {judgment}")
    
    print("=" * 50)
    print(f"Ideal (unbiased) score: 50.00%")


def main():
    print("=" * 60)
    print("Task 3: Bias Evaluation in Language Models")
    print(f"Domain: {SELECTED_DOMAIN}")
    print("=" * 60)
    
    # 步骤1: 加载数据
    print("\n[Step 1] Loading CrowS-Pairs dataset...")
    pairs = load_crows_pairs(SELECTED_DOMAIN, MAX_SAMPLES)
    
    print("\nSample pairs:")
    for i, pair in enumerate(pairs[:2]):
        print(f"\n  Pair {i+1}:")
        print(f"  Stereo: {pair['sent_more'][:100]}...")
        print(f"  Anti:   {pair['sent_less'][:100]}...")
        print(f"  Stereotyping: {pair['stereotyping']}")
    
    # 步骤2: 评估每个模型
    print("\n[Step 2] Evaluating models...")
    all_scores = {"domain": SELECTED_DOMAIN}
    all_dfs = []
    
    for model_name, model_path in MODELS.items():
        score, df = evaluate_bias(model_path, pairs, model_name)
        all_scores[model_name] = score
        all_dfs.append(df)
    
    # 步骤3: 保存结果
    print("\n[Step 3] Saving results...")
    
    summary_df = pd.DataFrame([all_scores])
    summary_df.to_csv(f"bias_summary_{SELECTED_DOMAIN.replace('/', '_')}.csv", index=False)
    
    for i, (model_name, _) in enumerate(MODELS.items()):
        all_dfs[i].to_csv(f"bias_details_{model_name}_{SELECTED_DOMAIN.replace('/', '_')}.csv", index=False)
    
    # 步骤4: 打印结果
    print_results_table(all_scores, SELECTED_DOMAIN)
    
    print("\nAll done!")
    print(f"Results saved to CSV files.")


if __name__ == "__main__":
    main()