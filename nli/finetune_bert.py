# finetune_bert.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 检查版本
import transformers
print(f"Transformers version: {transformers.__version__}")

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3,
    ignore_mismatched_sizes=True
)

# 加载数据集
dataset = load_dataset("nyu-mll/multi_nli")

# 使用更多数据
train_dataset = dataset["train"].shuffle(seed=42).select(range(50000))
val_dataset = dataset["validation_matched"].shuffle(seed=42).select(range(2000))

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

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# 过滤无效标签
train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
val_dataset = val_dataset.filter(lambda x: x["label"] != -1)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# 打印数据统计
print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Label distribution in train:")
train_labels = [x["labels"].item() for x in train_dataset]
print(f"  0 (entailment): {train_labels.count(0)}")
print(f"  1 (neutral): {train_labels.count(1)}")
print(f"  2 (contradiction): {train_labels.count(2)}")

# === 最简化的 TrainingArguments ===
training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="no",  # 不保存中间检查点，只保存最终模型
    fp16=True,
    seed=42,
    remove_unused_columns=False,  # 保留所有列
    dataloader_drop_last=False,
    prediction_loss_only=False,
    evaluation_strategy="no",  # 训练时不评估
)

# 使用最简化的 Trainer（旧版本兼容）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 旧版本支持
)

# 开始训练
print("\nStarting fine-tuning...")
trainer.train()

# 保存模型
print("\nSaving model...")
trainer.save_model("./bert_finetuned")
tokenizer.save_pretrained("./bert_finetuned")
print("Model saved!")

# === 手动评估 ===
print("\n=== Manual Evaluation ===")

# 评估函数
def manual_evaluate(trainer, dataset, name="dataset"):
    """手动计算准确率"""
    predictions = trainer.predict(dataset)
    
    # predictions.predictions 是 logits
    logits = predictions.predictions
    labels = predictions.label_ids
    
    # 过滤 -1 标签
    valid_mask = labels != -1
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    
    # 取 argmax
    preds = np.argmax(logits, axis=-1)
    
    # 计算指标
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1:.4f}")
    print(f"  Samples: {len(labels)}")
    
    return acc, f1

# 评估 matched
print("\nEvaluating on validation_matched...")
matched_acc, matched_f1 = manual_evaluate(trainer, val_dataset, "Validation Matched")

# 评估 mismatched
print("\nEvaluating on validation_mismatched...")
val_mismatched = dataset["validation_mismatched"].shuffle(seed=42).select(range(2000))
val_mismatched = val_mismatched.map(preprocess, batched=True)
val_mismatched = val_mismatched.filter(lambda x: x["label"] != -1)
val_mismatched.set_format("torch")

mismatched_acc, mismatched_f1 = manual_evaluate(trainer, val_mismatched, "Validation Mismatched")

print(f"\n=== Final Results ===")
print(f"Matched Accuracy: {matched_acc:.4f}")
print(f"Mismatched Accuracy: {mismatched_acc:.4f}")