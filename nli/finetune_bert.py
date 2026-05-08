# finetune_bert.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "bert-base-uncased"   # ← 改这里

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 加载完整训练集（课程允许）
dataset = load_dataset("nyu-mll/multi_nli")
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # 可减少样本加速

def preprocess(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(preprocess, batched=True)
train_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./bert_finetuned",
    num_train_epochs=1,                    # 1个epoch通常就够
    per_device_train_batch_size=32,
    save_strategy="no",
    logging_steps=100,
    fp16=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
trainer.save_model("./bert_finetuned")
print("Fine-tuning completed!")