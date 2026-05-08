# nliEvaluator.py
import torch
from utils import id2label, verbalizer


FEW_SHOT_EXAMPLES = [
    {"premise": "The man is in the kitchen.", "hypothesis": "The man is cooking dinner.", "label": "Neutral"},
    {"premise": "I am a lacto-vegetarian.", "hypothesis": "I enjoy eating cheese too much to abstain from dairy.", "label": "Neutral"},
    {"premise": "The Boston Center controller received a third transmission from American 11.", "hypothesis": "The Boston Center controller got a third transmission from American 11.", "label": "Entailment"},
    {"premise": "Met my first girlfriend that way.", "hypothesis": "I didn't meet my first girlfriend until later.", "label": "Contradiction"},
]

def create_fewshot_prompt(premise, hypothesis, shot=0):
    prompt = f"""Task: Natural Language Inference (NLI)
Given a Premise and a Hypothesis, determine their logical relationship.

Premise: {premise}
Hypothesis: {hypothesis}

Possible relationships:
- Entailment: The hypothesis is definitely true given the premise.
- Contradiction: The hypothesis is definitely false given the premise.
- Neutral: The hypothesis cannot be determined as true or false from the premise.
"""    # 添加 few-shot 示例


    if shot > 0:
        prompt += "Here are some examples:\n\n"
        for ex in FEW_SHOT_EXAMPLES[:shot]:
            prompt += f"Premise: {ex['premise']}\n"
            prompt += f"Hypothesis: {ex['hypothesis']}\n"
            prompt += f"Answer: {ex['label']}\n\n"
        prompt += "Now analyze the following pair:\n\n"
    
    prompt += "\nThe relationship is:"
    
    return prompt


def predict(model, tokenizer, premise, hypothesis, model_key, shot=0):
    
    if "gpt" in model_key.lower() or "qwen" in model_key.lower():
        prompt = create_fewshot_prompt(premise, hypothesis, shot)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            scores = []
            for word in verbalizer:
                token_id = tokenizer.encode(word, add_special_tokens=False)[0]
                scores.append(next_token_logits[token_id].item())
            
            pred_id = torch.argmax(torch.tensor(scores)).item()
            probs = torch.softmax(torch.tensor(scores), dim=-1)
        
        print(f"[{model_key} {shot}-shot] Scores: {[round(s,3) for s in scores]} → {id2label[pred_id]}")
        return pred_id
    
    else:
        # BERT-finetuned 等分类模型
        inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
        return pred_id
    
