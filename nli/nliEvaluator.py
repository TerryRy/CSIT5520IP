# nli_evaluator.py
from utils import label2id
import torch

def create_prompt(premise: str, hypothesis: str, shot: int = 0, examples=None):
    base = f"""Given the premise and hypothesis, determine their relationship: entailment, neutral, or contradiction.

Premise: {premise}
Hypothesis: {hypothesis}

Answer (only one word):"""
    
    if shot == 0:
        return base
    
    # Few-shot
    few_shot = "Here are some examples:\n\n"
    for ex in examples[:shot]:
        few_shot += f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nAnswer: {ex['label']}\n\n"
    return few_shot + base

# nliEvaluator.py
import torch

def predict_classification(model, tokenizer, premise, hypothesis):
    """专门给 BERT 类分类模型使用"""
    inputs = tokenizer(premise, hypothesis, 
                      truncation=True, 
                      padding=True, 
                      max_length=256, 
                      return_tensors="pt")
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred


def predict_generation(model, tokenizer, premise, hypothesis, shot=0):
    """专门给 Qwen 这类生成模型使用"""
    prompt = f"""Premise: {premise}
Hypothesis: {hypothesis}

Does the hypothesis entail, contradict, or is neutral to the premise?
Answer with only one word: Entailment, Neutral, or Contradiction.

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.lower()
    
    if "entail" in answer:
        return 0
    elif "contradict" in answer:
        return 2
    else:
        return 1


def predict(model, tokenizer, premise, hypothesis, model_key, shot=0):
    """统一调用函数"""
    if "qwen" in model_key.lower():
        return predict_generation(model, tokenizer, premise, hypothesis, shot)
    else:
        # BERT, DeBERTa, finetuned
        return predict_classification(model, tokenizer, premise, hypothesis)