# finetune_bert.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

print(f"Transformers version: {transformers.__version__}")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载数据集（只加载一次）
dataset = load_dataset("nyu-mll/multi_nli")

# 预处理函数
def preprocess(examples):
    result = tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        truncation=True, 
        padding="max_length", 
        max_length=256
    )
    result["labels"] = examples["label"]
    return result

# 定义不同数据量的训练配置
DATA_SIZES = [1000, 5000, 10000]  # 50000 已完成，跳过

# 验证集（保持一致性）
val_matched = dataset["validation_matched"].shuffle(seed=42).select(range(2000))
val_matched = val_matched.map(preprocess, batched=True)
val_matched = val_matched.filter(lambda x: x["label"] != -1)

val_mismatched = dataset["validation_mismatched"].shuffle(seed=42).select(range(2000))
val_mismatched = val_mismatched.map(preprocess, batched=True)
val_mismatched = val_mismatched.filter(lambda x: x["label"] != -1)

def evaluate_model(model, tokenizer, dataset, device, batch_size=32):
    """评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    
    premises = dataset["premise"]
    hypotheses = dataset["hypothesis"]
    labels = dataset["labels"]
    
    n_samples = len(premises)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_premises = premises[i:batch_end]
        batch_hypotheses = hypotheses[i:batch_end]
        batch_labels = labels[i:batch_end]
        
        inputs = tokenizer(
            batch_premises,
            batch_hypotheses,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        labels_tensor = torch.tensor(batch_labels).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_tensor.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    valid_mask = all_labels != -1
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return acc, f1

# 循环训练不同数据量的模型
for data_size in DATA_SIZES:
    print(f"\n{'='*50}")
    print(f"Training BERT with {data_size} samples")
    print('='*50)
    
    # 创建新的模型（每次从头训练）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    
    # 选择对应数量的训练数据
    train_dataset = dataset["train"].shuffle(seed=42).select(range(data_size))
    train_dataset = train_dataset.map(preprocess, batched=True)
    train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
    
    # 输出目录
    output_dir = f"./bert_finetuned_{data_size}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=5000,
        fp16=True,
        seed=42,
        save_total_limit=1,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 训练
    print(f"\nStarting training with {data_size} samples...")
    trainer.train()
    
    # 保存模型
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # 评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("\nEvaluating on validation sets...")
    matched_acc, matched_f1 = evaluate_model(model, tokenizer, val_matched, device)
    mismatched_acc, mismatched_f1 = evaluate_model(model, tokenizer, val_mismatched, device)
    
    print(f"\n=== Results for {data_size} samples ===")
    print(f"Matched Accuracy: {matched_acc:.4f}")
    print(f"Mismatched Accuracy: {mismatched_acc:.4f}")
    
    # 保存结果到日志
    with open("bert_finetune_results.txt", "a") as f:
        f.write(f"{data_size}\t{matched_acc:.4f}\t{mismatched_acc:.4f}\n")

print("\nAll fine-tuning completed!")