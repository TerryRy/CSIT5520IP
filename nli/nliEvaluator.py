# nli_evaluator.py
from utils import label2id
import torch


FEW_SHOT_EXAMPLES = [
    {
        "premise": "The man is in the kitchen.",
        "hypothesis": "The man is cooking.",
        "label": "Neutral"
    },
    {
        "premise": "I am a lacto-vegetarian.",
        "hypothesis": "I enjoy eating cheese too much to abstain from dairy.",
        "label": "Neutral"
    },
    {
        "premise": "The Boston Center controller received a third transmission from American 11.",
        "hypothesis": "The Boston Center controller got a third transmission from American 11.",
        "label": "Entailment"
    },
    {
        "premise": "Met my first girlfriend that way.",
        "hypothesis": "I didn't meet my first girlfriend until later.",
        "label": "Contradiction"
    }
]

def create_strong_prompt(premise: str, hypothesis: str, shot: int = 0):
    prompt = """You are an expert in Natural Language Inference (NLI). 
Your task is to determine the relationship between a Premise and a Hypothesis.

Possible relationships:
- Entailment: The hypothesis is definitely true based on the premise.
- Contradiction: The hypothesis is definitely false based on the premise.
- Neutral: The hypothesis may or may not be true, we cannot determine.

"""
    if shot > 0:
        prompt += "Here are some examples:\n\n"
        for ex in FEW_SHOT_EXAMPLES[:shot]:
            prompt += f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nAnswer: {ex['label']}\n\n"
    
    prompt += f"""Now analyze the following:

Premise: {premise}
Hypothesis: {hypothesis}

Answer with **only one word** from: Entailment, Neutral, Contradiction.

Answer:"""
    
    return prompt

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


# def predict(model, tokenizer, premise, hypothesis, model_key, shot=0):
#     """统一调用函数"""
#     if "qwen" in model_key.lower():
#         return predict_generation(model, tokenizer, premise, hypothesis, shot)
#     else:
#         # BERT, DeBERTa, finetuned
#         return predict_classification(model, tokenizer, premise, hypothesis)
    

def predict(model, tokenizer, premise, hypothesis, model_key, shot=0):
    id2label = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    
    if "bert" in model_key.lower() or "deberta" in model_key.lower():
        # === Classification Models ===
        inputs = tokenizer(
            premise, 
            hypothesis, 
            truncation=True, 
            padding=True, 
            max_length=256, 
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(logits, dim=-1).item()
            prob = probs[0][pred_id].item()
        
        print(f"[{model_key}] Logits: {logits[0].cpu().numpy().round(3)} | "
              f"Pred: {id2label[pred_id]} ({prob:.3f})")
        
        return pred_id
    
    else:
        # === Qwen ===
        prompt = f"""Premise: {premise}
Hypothesis: {hypothesis}

Relationship: Entailment, Neutral, or Contradiction?
Answer (one word only):"""

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
        answer = response.split("Answer")[-1].strip().lower() if "Answer" in response else response.lower()
        
        if "entail" in answer:
            pred_id = 0
        elif "contradict" in answer:
            pred_id = 2
        else:
            pred_id = 1
            
        print(f"[Qwen] Raw response: {response[-100:]} → Pred: {id2label[pred_id]}")
        return pred_id