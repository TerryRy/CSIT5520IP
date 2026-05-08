# # create_few_shots_jsonl.py
# import json
# import random

# def create_few_shots_jsonl(input_file="dev_matched_sampled-1.jsonl", output_file="few-shots.jsonl"):
#     """从dev_matched随机选择5个领域，每个领域每类标签各1条"""
    
#     # 读取数据并按genre和label组织
#     data_by_genre_label = {}
    
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 item = json.loads(line)
#                 genre = item.get('genre', 'unknown')
#                 # 跳过无效的 genre（如 "-"）
#                 if genre == '-' or not genre:
#                     continue
                
#                 label = item['gold_label']
#                 # 跳过无效的 label
#                 if label not in ["entailment", "neutral", "contradiction"]:
#                     continue
                
#                 if genre not in data_by_genre_label:
#                     data_by_genre_label[genre] = {"entailment": [], "neutral": [], "contradiction": []}
                
#                 data_by_genre_label[genre][label].append({
#                     'premise': item['sentence1'],
#                     'hypothesis': item['sentence2'],
#                     'gold_label': item['gold_label']
#                 })
    
#     # 选择有足够数据的5个领域
#     genres_with_all_labels = [
#         g for g, labels in data_by_genre_label.items()
#         if all(len(labels[l]) >= 1 for l in ["entailment", "neutral", "contradiction"])
#     ]
    
#     selected_genres = random.sample(genres_with_all_labels, min(5, len(genres_with_all_labels)))
    
#     # 为每个领域每类标签选择1条
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for genre in selected_genres:
#             entry = {
#                 genre: {}
#             }
            
#             for label in ["entailment", "neutral", "contradiction"]:
#                 candidates = data_by_genre_label[genre][label]
#                 chosen = random.choice(candidates)
#                 entry[genre][label] = {
#                     "gold_label": chosen['gold_label'],
#                     "premise": chosen['premise'],
#                     "hypothesis": chosen['hypothesis']
#                 }
            
#             f.write(json.dumps(entry) + '\n')
    
#     print(f"Created {output_file} with {len(selected_genres)} genres: {selected_genres}")

# if __name__ == "__main__":
#     create_few_shots_jsonl()

# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（如果系统有的话）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
models = ['GPT-2', 'Flan-T5', 'Qwen3']
matched_acc = [0.3092, 0.5784, 0.7188]
mismatched_acc = [0.3112, 0.5556, 0.7176]

# Baseline数据
baselines = {
    'Random (0.33)': 0.33,
    'BERT-1000 (0.38/0.40)': 0.38,  # 使用matched
    'BERT-5000 (0.63/0.67)': 0.63,
    'BERT-10000 (0.70/0.72)': 0.70,
    'BERT-50000 (0.78/0.78)': 0.78,
}

baselines_mismatched = {
    'Random (0.33)': 0.33,
    'BERT-1000 (0.38/0.40)': 0.40,  # 使用mismatched
    'BERT-5000 (0.63/0.67)': 0.67,
    'BERT-10000 (0.70/0.72)': 0.72,
    'BERT-50000 (0.78/0.78)': 0.78,
}

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 颜色设置
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# ============ Matched 子图 ============
x = np.arange(len(models))
width = 0.5

bars1 = ax1.bar(x, matched_acc, width, color=colors, edgecolor='black', linewidth=1.2)

# 在柱状图顶部标出具体数值
for i, (bar, acc) in enumerate(zip(bars1, matched_acc)):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加baseline水平虚线
colors_baselines = ['#808080', '#FFA07A', '#98FB98', '#87CEEB', '#DDA0DD']
for (name, value), color in zip(baselines.items(), colors_baselines):
    ax1.axhline(y=value, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(2.3, value + 0.005, name, color=color, fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Matched Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.9)
ax1.grid(axis='y', alpha=0.3)

# ============ Mismatched 子图 ============
bars2 = ax2.bar(x, mismatched_acc, width, color=colors, edgecolor='black', linewidth=1.2)

for i, (bar, acc) in enumerate(zip(bars2, mismatched_acc)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

for (name, value), color in zip(baselines_mismatched.items(), colors_baselines):
    ax2.axhline(y=value, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(2.3, value + 0.005, name, color=color, fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Mismatched Accuracy', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 0.9)
ax2.grid(axis='y', alpha=0.3)

# 总标题
fig.suptitle('Task 2.1: Zero-shot NLI Performance Comparison\nwith Fine-tuned BERT Baselines', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('nli_results_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved to nli_results_comparison.png")