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

def predict_prompting(model, tokenizer, premise, hypothesis, shot=0, examples=None, max_new=10):
    prompt = create_prompt(premise, hypothesis, shot, examples)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip().lower()
    
    if "entail" in answer:
        return 0
    elif "contradict" in answer:
        return 2
    else:
        return 1  # neutral as default