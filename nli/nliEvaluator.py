# nliEvaluator.py
import torch
from utils import id2label, verbalizer


def create_prompt(premise, hypothesis, shot=0):
    base = f"""Premise: {premise}
Hypothesis: {hypothesis}

Determine the relationship: Entailment, Neutral, or Contradiction.
Answer with exactly one word.

Answer:"""
    # 可以后续加 few-shot examples
    return base


# nliEvaluator.py
import torch



def predict(model, tokenizer, premise, hypothesis, model_key, shot=0):
    
    # ====================== GPT-2 / Qwen 等 Causal LM (Verbalizer 方式) ======================
    if "gpt" in model_key.lower() or "qwen" in model_key.lower():
        prompt = f"""Premise: {premise}
Hypothesis: {hypothesis}

The relationship is:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]   # 最后一个位置的 logits
            
            # 计算 verbalizer 中每个词的概率
            scores = []
            for word in verbalizer:
                token_id = tokenizer.encode(word, add_special_tokens=False)[0]
                scores.append(next_token_logits[token_id].item())
            
            probs = torch.softmax(torch.tensor(scores), dim=-1)
            pred_id = torch.argmax(torch.tensor(scores)).item()
            best_prob = probs[pred_id].item()
        
        print(f"[{model_key} Verbalizer] Scores: { [round(s,3) for s in scores] } | "
              f"Probs: {probs.numpy().round(3)} → Pred: {id2label[pred_id]} ({best_prob:.3f})")
        return pred_id

    # ====================== BERT / DeBERTa / Finetuned (Classification) ======================
    else:
        inputs = tokenizer(premise, hypothesis, 
                          truncation=True, 
                          padding=True, 
                          max_length=256, 
                          return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=-1)[0]
        
        print(f"[{model_key}] Probs: {probs.cpu().numpy().round(3)} → Pred: {id2label[pred_id]}")
        return pred_id