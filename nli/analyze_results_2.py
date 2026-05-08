# analyze_results_2.py
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
results = pd.read_csv("hallucination_detection_results.csv")

# 比较不同模型的F1
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：各模型F1对比
model_f1 = results[results['method'] == 'label_only'][['model', 'f1']]
axes[0].bar(model_f1['model'], model_f1['f1'])
axes[0].set_title('F1 Score by Model (Label-only)')
axes[0].set_ylabel('F1 Score')
axes[0].tick_params(axis='x', rotation=45)

# 右图：不同阈值下的F1
for model in results['model'].unique():
    model_data = results[results['model'] == model]
    thresholds = model_data[model_data['method'] == 'score_based']['threshold']
    f1_scores = model_data[model_data['method'] == 'score_based']['f1']
    axes[1].plot(thresholds, f1_scores, marker='o', label=model)

axes[1].set_title('F1 Score vs Threshold')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('F1 Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('hallucination_results.png')