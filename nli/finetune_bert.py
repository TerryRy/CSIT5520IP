# finetune_bert.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 检查 transformers 版本
import transformers
print(f"Transformers version: {transformers.__version__}")

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# === 关键改进: 使用 ignore_mismatched_sizes 处理 UNEXPECTED 警告 ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3,
    ignore_mismatched_sizes=True  # 忽略分类头尺寸不匹配的警告
)

# 加载完整训练集
dataset = load_dataset("nyu-mll/multi_nli")

# 增加训练数据量
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

# === 关键改进: 使用兼容旧版本的 TrainingArguments ===
# 旧版本不支持 evaluation_strategy，改用预测时评估
training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    
    # 训练轮数
    num_train_epochs=3,
    
    # 批次大小
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    
    # 学习率
    learning_rate=2e-5,
    warmup_steps=500,  # 旧版本使用 warmup_steps 而不是 warmup_ratio
    
    # 优化器
    weight_decay=0.01,
    
    # 保存策略
    save_strategy="epoch",  # 旧版本每轮保存
    save_total_limit=2,     # 只保留最后2个检查点
    
    # 日志
    logging_steps=100,
    logging_dir="./logs",
    
    # 混合精度
    fp16=True,
    
    # === 不使用 evaluation_strategy（旧版本不支持）===
    # 改为在训练结束后手动评估
    
    report_to="none",
    
    # 禁用 wandb 等
    disable_tqdm=False,
    
    # 梯度累积（如果显存不足）
    # gradient_accumulation_steps=2,
    
    # 随机种子
    seed=42,
    data_seed=42,
    
    # 旧版本兼容
    dataloader_drop_last=False,
    remove_unused_columns=False,
)

# === 评估函数 ===
def compute_metrics(pred):
    """旧版本 compute_metrics 的签名"""
    predictions, labels = pred
    
    # 如果是 logits，取 argmax
    if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
        predictions = np.argmax(predictions, axis=-1)
    
    # 过滤 -1 标签
    valid_mask = labels != -1
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }

# 用旧版方式创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 旧版本支持 eval_dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
print("Starting fine-tuning...")
trainer.train()

# 保存模型
trainer.save_model("./bert_finetuned")
tokenizer.save_pretrained("./bert_finetuned")
print("Fine-tuning completed and model saved!")

# === 手动评估 ===
print("\n=== Manual Evaluation ===")

# 在验证集上评估
print("Evaluating on validation_matched...")
val_results = trainer.evaluate(val_dataset)
print(f"Validation matched results: {val_results}")

# 在 mismatched 上评估
val_mismatched = dataset["validation_mismatched"].shuffle(seed=42).select(range(2000))
val_mismatched = val_mismatched.map(preprocess, batched=True)
val_mismatched = val_mismatched.filter(lambda x: x["label"] != -1)
val_mismatched.set_format("torch")

print("Evaluating on validation_mismatched...")
mm_results = trainer.evaluate(val_mismatched)
print(f"Validation mismatched results: {mm_results}")

# 打印最终准确率
print(f"\nFinal Results:")
print(f"  Matched accuracy: {val_results.get('eval_accuracy', 'N/A'):.4f}")
print(f"  Mismatched accuracy: {mm_results.get('eval_accuracy', 'N/A'):.4f}")