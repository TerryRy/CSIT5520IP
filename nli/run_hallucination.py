# run_hallucination.py
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

# 复用第一个任务的工具
from utils import load_model_and_tokenizer, MODELS, id2label, verbalizer, load_hallucination_data
from nliEvaluator import predict as nli_predict

# 设置随机种子
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def nli_to_hallucination(nli_label):
    """
    将NLI标签转换为幻觉检测标签
    
    NLI标签: entailment, neutral, contradiction
    幻觉标签: 0=factual, 1=hallucination
    
    映射规则:
    - entailment → factual (0): 前提蕴含假设，说明假设是事实
    - neutral/contradiction → hallucination (1): 无法确定或矛盾，说明可能幻觉
    """
    if nli_label == "entailment":
        return 0  # factual
    else:
        return 1  # hallucination


def evaluate_hallucination_detection(model, tokenizer, df, model_key):
    """
    使用NLI模型进行幻觉检测评估
    """
    print(f"\nEvaluating {model_key} on hallucination detection...")
    
    predictions = []
    nli_predictions = []  # 保存NLI中间结果用于分析
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_key}"):
        # 步骤1: 使用NLI模型预测
        # premise = wiki bio text, hypothesis = gpt3 sentence
        nli_pred_id = nli_predict(model, tokenizer, row["premise"], row["hypothesis"], model_key, shot=0)
        nli_label = id2label[nli_pred_id]
        nli_predictions.append(nli_label)
        
        # 步骤2: 将NLI标签转换为幻觉检测标签
        hallucination_pred = nli_to_hallucination(nli_label)
        predictions.append(hallucination_pred)
    
    # 计算指标
    ground_truth = df["binary_label"].tolist()
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="binary", pos_label=1
    )
    
    # 打印详细结果
    print(f"\n{'='*50}")
    print(f"Results for {model_key}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(ground_truth, predictions, 
                                target_names=["factual", "hallucination"]))
    
    # NLI标签分布分析
    nli_series = pd.Series(nli_predictions)
    print(f"\nNLI label distribution:")
    print(nli_series.value_counts())
    print(f"  entailment: {nli_series.value_counts().get('entailment', 0)}")
    print(f"  neutral: {nli_series.value_counts().get('neutral', 0)}")
    print(f"  contradiction: {nli_series.value_counts().get('contradiction', 0)}")
    
    return {
        "model": model_key,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def analyze_failure_cases(df, predictions, model_key, num_examples=5):
    """
    分析失败案例，帮助理解模型的局限性
    """
    ground_truth = df["binary_label"].tolist()
    
    # 找出错误预测
    df_copy = df.copy()
    df_copy["prediction"] = predictions
    df_copy["correct"] = df_copy["binary_label"] == df_copy["prediction"]
    
    # False Positives: 预测为hallucination但实际是factual
    fp_df = df_copy[(df_copy["prediction"] == 1) & (df_copy["binary_label"] == 0)]
    # False Negatives: 预测为factual但实际是hallucination
    fn_df = df_copy[(df_copy["prediction"] == 0) & (df_copy["binary_label"] == 1)]
    
    print(f"\nFailure Case Analysis for {model_key}:")
    print(f"  False Positives (predicted hallucination, actual factual): {len(fp_df)}")
    print(f"  False Negatives (predicted factual, actual hallucination): {len(fn_df)}")
    
    # 显示一些例子
    if len(fp_df) > 0:
        print(f"\n  Sample False Positives:")
        for i, (_, row) in enumerate(fp_df.head(num_examples).iterrows()):
            print(f"  {i+1}. Premise: {row['premise'][:100]}...")
            print(f"     Hypothesis: {row['hypothesis'][:100]}...")
            print(f"     Original label: {row['original_label']}")
            print()
    
    if len(fn_df) > 0:
        print(f"\n  Sample False Negatives:")
        for i, (_, row) in enumerate(fn_df.head(num_examples).iterrows()):
            print(f"  {i+1}. Premise: {row['premise'][:100]}...")
            print(f"     Hypothesis: {row['hypothesis'][:100]}...")
            print(f"     Original label: {row['original_label']}")
            print()


def main():
    """主函数"""
    print("=" * 60)
    print("Task 2.2: NLI for Hallucination Detection")
    print("=" * 60)
    
    # 步骤1: 加载数据（使用test集）
    print("\n[Step 1] Loading test data...")
    test_df = load_hallucination_data(split="evaluation", max_samples=10)
    # TODO 验证完删掉这行注释
    test_df = test_df.head(5)
    # 步骤2: 选择第二个任务要使用的模型
    # 复用第一个任务的 MODELS 字典
    selected_models = {
        "flan_t5_prompt": MODELS["flan_t5_prompt"],
        "qwen_prompt": MODELS["qwen_prompt"],
        "bert_finetuned_10000": MODELS["bert_finetuned_10000"],
        "bert_finetuned_50000": MODELS["bert_finetuned_50000"]
    }
    
    # 步骤3: 对每个模型进行评估
    print("\n[Step 2] Evaluating models on hallucination detection...")
    all_results = []
    
    for model_key, model_path in selected_models.items():
        print(f"\n{'#'*50}")
        print(f"Processing model: {model_key}")
        print(f"{'#'*50}")
        
        try:
            # 加载模型
            model, tokenizer = load_model_and_tokenizer(model_path)
            
            # 评估
            result = evaluate_hallucination_detection(model, tokenizer, test_df, model_key)
            all_results.append(result)
            
            # 分析失败案例（只对最好的模型做）
            if model_key == "bert_finetuned_50000":
                # 重新获取预测用于失败分析
                predictions = []
                for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Failure analysis"):
                    nli_pred_id = nli_predict(model, tokenizer, row["premise"], row["hypothesis"], model_key, shot=0)
                    nli_label = id2label[nli_pred_id]
                    hallucination_pred = nli_to_hallucination(nli_label)
                    predictions.append(hallucination_pred)
                
                analyze_failure_cases(test_df, predictions, model_key)
            
            # 显存清理
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
            continue
    
    # 步骤4: 保存结果
    print("\n[Step 3] Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("hallucination_detection_results.csv", index=False)
    
    # 打印汇总表
    print("\n" + "=" * 60)
    print("Summary of Hallucination Detection Results")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to hallucination_detection_results.csv")
    
    return results_df

if __name__ == "__main__":
    main()