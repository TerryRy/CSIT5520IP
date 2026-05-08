# load_hallucination_data.py
from datasets import load_dataset
import pandas as pd
import json

def load_wikibio_hallucination(split="train"):
    """加载 wikibio-gpt3-hallucination 数据集"""
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    
    data = []
    for item in dataset[split]:
        # 每条数据包含多个句子
        wiki_bio_text = item["wiki_bio_text"]  # premise
        gpt3_sentences = item["gpt3_sentences"]  # list of hypotheses
        annotations = item["annotation"]  # list of labels
        
        for sentence, annotation in zip(gpt3_sentences, annotations):
            # 将标签二值化：0=factual, 1=non-factual
            if annotation == 0:  # Accurate
                binary_label = 0  # factual
            else:  # Major (1) or Minor (0.5) Inaccurate
                binary_label = 1  # non-factual (hallucination)
            
            data.append({
                "premise": wiki_bio_text,
                "hypothesis": sentence,
                "annotation": annotation,
                "binary_label": binary_label  # 0=factual, 1=hallucination
            })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} sentence-level samples")
    print(f"Label distribution:\n{df['binary_label'].value_counts()}")
    return df