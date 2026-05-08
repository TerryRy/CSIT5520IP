# finetune_bert.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 加载完整训练集
dataset = load_dataset("nyu-mll/multi_nli")

# === 关键改进1: 增加训练数据量 ===
# 之前只用了10000条，对于BERT来说太少
train_dataset = dataset["train"].shuffle(seed=42).select(range(50000))  # 增加到50000条
# 如果显存够，可以用100000条

# === 关键改进2: 添加验证集用于early stopping ===
val_dataset = dataset["validation_matched"].shuffle(seed=42).select(range(2000))

def preprocess(examples):
    # === 关键改进3: 确保label映射正确 ===
    # MultiNLI的label: 0=entailment, 1=neutral, 2=contradiction
    # 但数据集中可能有-1（无标签），需要过滤
    result = tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        truncation=True, 
        padding="max_length", 
        max_length=256
    )
    result["labels"] = examples["label"]
    return result

# 过滤掉label为-1的样本
def filter_invalid_labels(examples):
    valid_indices = [i for i, label in enumerate(examples["label"]) if label != -1]
    filtered = {k: [v[i] for i in valid_indices] for k, v in examples.items()}
    return filtered

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# 过滤无效标签
train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
val_dataset = val_dataset.filter(lambda x: x["label"] != -1)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# === 关键改进4: 优化训练参数 ===
training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    
    # 训练轮数
    num_train_epochs=3,  # 增加到3轮，配合early stopping
    
    # 批次大小（根据显存调整）
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    
    # 学习率（BERT标准是2e-5）
    learning_rate=2e-5,
    warmup_ratio=0.1,  # 学习率预热
    
    # 优化器
    optim="adamw_torch",
    weight_decay=0.01,  # 权重衰减防过拟合
    
    # 评估策略
    evaluation_strategy="steps",
    eval_steps=500,  # 每500步评估一次
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,  # 加载最佳模型
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    
    # 防止过拟合
    logging_steps=100,
    fp16=True,  # 混合精度训练
    
    # 梯度累积（如果显存不足）
    # gradient_accumulation_steps=2,
    
    report_to="none"  # 不报告到外部
)

# === 关键改进5: 添加评估指标 ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 过滤掉-1标签
    valid_mask = labels != -1
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted")
    }

# === 关键改进6: 使用Early Stopping ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 验证集3次不提升就停止
)

# 开始训练
print("Starting fine-tuning...")
trainer.train()

# 保存最终模型和tokenizer
trainer.save_model("./bert_finetuned")
tokenizer.save_pretrained("./bert_finetuned")
print("Fine-tuning completed and model saved!")

# === 关键改进7: 在更多验证集上测试 ===
print("\nEvaluating on full validation sets...")

# 加载测试数据
test_datasets = {
    "matched": dataset["validation_matched"],
    "mismatched": dataset["validation_mismatched"]
}

for split_name, test_data in test_datasets.items():
    # 过滤无效标签
    test_data = test_data.filter(lambda x: x["label"] != -1)
    
    # 预处理
    test_data = test_data.map(preprocess, batched=True)
    test_data.set_format("torch")
    
    # 评估
    results = trainer.evaluate(test_data)
    print(f"{split_name} accuracy: {results['eval_accuracy']:.4f}")