# finetune_bert.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3,
    ignore_mismatched_sizes=True
)

# 加载数据集
dataset = load_dataset("nyu-mll/multi_nli")

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
print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

# === 兼容旧版 transformers 5.7.0 ===
# 旧版 TrainingArguments 只支持最基本的参数
training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    num_train_epochs=3,                    # 训练轮数
    per_device_train_batch_size=32,        # 批次大小
    learning_rate=2e-5,                    # 学习率
    warmup_steps=500,                      # 预热步数
    weight_decay=0.01,                     # 权重衰减
    logging_steps=100,                     # 日志步数
    save_steps=5000,                       # 保存步数（设大一点，减少保存次数）
    fp16=True,                             # 混合精度
    seed=42,                               # 随机种子
    save_total_limit=1,                    # 只保留1个检查点
    remove_unused_columns=False,           # 保留所有列
)

# 旧版 Trainer 不支持 compute_metrics，也不支持 eval_dataset
# 我们改为训练完成后手动评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
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

def evaluate_model(model, tokenizer, dataset, device):
    """手动评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for i in range(0, len(dataset), 32):  # batch size = 32
        batch = dataset[i:i+32]
        
        inputs = tokenizer(
            [x["premise"] for x in batch] if "premise" in batch else batch,
            [x["hypothesis"] for x in batch] if "hypothesis" in batch else batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        labels = torch.tensor([x["labels"] for x in batch]).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # 转换为 numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 过滤 -1 标签
    valid_mask = all_labels != -1
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    return acc, f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 评估 validation_matched
print("\nEvaluating on validation_matched...")
matched_acc, matched_f1 = evaluate_model(model, tokenizer, val_dataset, device)
print(f"Validation Matched: acc={matched_acc:.4f}, f1={matched_f1:.4f}")

# 评估 validation_mismatched
print("\nEvaluating on validation_mismatched...")
val_mismatched = dataset["validation_mismatched"].shuffle(seed=42).select(range(2000))
val_mismatched = val_mismatched.map(preprocess, batched=True)
val_mismatched = val_mismatched.filter(lambda x: x["label"] != -1)
val_mismatched.set_format("torch")

mismatched_acc, mismatched_f1 = evaluate_model(model, tokenizer, val_mismatched, device)
print(f"Validation Mismatched: acc={mismatched_acc:.4f}, f1={mismatched_f1:.4f}")

print(f"\n=== Final Results ===")
print(f"Matched Accuracy: {matched_acc:.4f}")
print(f"Mismatched Accuracy: {mismatched_acc:.4f}")