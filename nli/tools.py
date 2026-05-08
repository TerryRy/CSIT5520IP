# create_few_shots_jsonl.py
import json
import random

def create_few_shots_jsonl(input_file="dev_matched_sampled-1.jsonl", output_file="few-shots.jsonl"):
    """从dev_matched随机选择5个领域，每个领域每类标签各1条"""
    
    # 读取数据并按genre和label组织
    data_by_genre_label = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                genre = item.get('genre', 'unknown')
                # 跳过无效的 genre（如 "-"）
                if genre == '-' or not genre:
                    continue
                
                label = item['gold_label']
                # 跳过无效的 label
                if label not in ["entailment", "neutral", "contradiction"]:
                    continue
                
                if genre not in data_by_genre_label:
                    data_by_genre_label[genre] = {"entailment": [], "neutral": [], "contradiction": []}
                
                data_by_genre_label[genre][label].append({
                    'premise': item['sentence1'],
                    'hypothesis': item['sentence2'],
                    'gold_label': item['gold_label']
                })
    
    # 选择有足够数据的5个领域
    genres_with_all_labels = [
        g for g, labels in data_by_genre_label.items()
        if all(len(labels[l]) >= 1 for l in ["entailment", "neutral", "contradiction"])
    ]
    
    selected_genres = random.sample(genres_with_all_labels, min(5, len(genres_with_all_labels)))
    
    # 为每个领域每类标签选择1条
    with open(output_file, 'w', encoding='utf-8') as f:
        for genre in selected_genres:
            entry = {
                genre: {}
            }
            
            for label in ["entailment", "neutral", "contradiction"]:
                candidates = data_by_genre_label[genre][label]
                chosen = random.choice(candidates)
                entry[genre][label] = {
                    "gold_label": chosen['gold_label'],
                    "premise": chosen['premise'],
                    "hypothesis": chosen['hypothesis']
                }
            
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created {output_file} with {len(selected_genres)} genres: {selected_genres}")

if __name__ == "__main__":
    create_few_shots_jsonl()