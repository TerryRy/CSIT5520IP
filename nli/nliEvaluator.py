# nliEvaluator.py
import torch
from utils import id2label, verbalizer, map_t5_output_to_label


FEW_SHOT_EXAMPLES = [
    {"premise": "The man is in the kitchen.", "hypothesis": "The man is cooking dinner.", "label": "neutral"},
    {"premise": "I am a lacto-vegetarian.", "hypothesis": "I enjoy eating cheese too much to abstain from dairy.", "label": "neutral"},
    {"premise": "The Boston Center controller received a third transmission from American 11.", "hypothesis": "The Boston Center controller got a third transmission from American 11.", "label": "entailment"},
    {"premise": "Met my first girlfriend that way.", "hypothesis": "I didn't meet my first girlfriend until later.", "label": "contradiction"},
]

def create_fewshot_prompt(premise, hypothesis, shot=0):
    prompt = f"""Task: Natural Language Inference
Given a Premise and a Hypothesis, determine their logical relationship.

Premise: {premise}
Hypothesis: {hypothesis}

Possible relationships:
- entailment
- contradiction
- neutral
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

# T5 prompt
def create_t5_prompt(premise, hypothesis, shot=0):
    """T5 seq2seq  prompt"""
    
    # T5 推荐的任务前缀
    prompt = "nli: "
    
    prompt += f"premise: {premise} hypothesis: {hypothesis} \n"
    
    # 添加 few-shot 示例
    if shot > 0:
        for ex in FEW_SHOT_EXAMPLES[:shot]:
            prompt += f"premise: {ex['premise']} hypothesis: {ex['hypothesis']} answer: {ex['label']} "
    
    # 当前要预测的样本
    prompt += "answer:"
    
    return prompt

# 或者使用更详细的 T5 prompt
def create_t5_prompt_detailed(premise, hypothesis, shot=0):
    """更详细的 T5 prompt 格式"""
    
    if shot > 0:
        prompt = "Classify the relationship between premise and hypothesis as entailment, contradiction, or neutral.\n\n"
                
        prompt += f"Premise: {premise}\n"
        prompt += f"Hypothesis: {hypothesis}\n"
        
        for ex in FEW_SHOT_EXAMPLES[:shot]:
            prompt += f"Premise: {ex['premise']}\n"
            prompt += f"Hypothesis: {ex['hypothesis']}\n"
            prompt += f"Answer: {ex['label']}\n\n"

        prompt += "Answer:"
    else:
        prompt = f"Premise: {premise} Hypothesis: {hypothesis} Is the hypothesis entailed by the premise? Answer:"
    
    return prompt

# Causal LM 的 prompt 也可以优化
def create_causal_prompt(premise, hypothesis, shot=0):
    """优化的 Causal LM prompt (GPT, Qwen 等)"""
    
    prompt = """You are an expert in natural language inference. Determine if the hypothesis follows from the premise.

"""
    
    if shot > 0:
        for ex in FEW_SHOT_EXAMPLES[:shot]:
            prompt += f"Premise: {ex['premise']}\n"
            prompt += f"Hypothesis: {ex['hypothesis']}\n"
            prompt += f"Relationship: {ex['label']}\n\n"
    
    prompt += f"Premise: {premise}\n"
    prompt += f"Hypothesis: {hypothesis}\n"
    prompt += "Relationship:"
    
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
            # probs = torch.softmax(torch.tensor(scores), dim=-1)
        
        # print(f"[{model_key} {shot}-shot] Scores: {[round(s,3) for s in scores]} → {id2label[pred_id]}")
        return pred_id
    
    elif "t5" in model_key.lower():
        # T5 Seq2Seq 推理
        prompt = create_t5_prompt_detailed(premise, hypothesis, shot)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            # T5 使用 generate 方法
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=1,  # 贪心解码，或使用 beam search
                do_sample=False,
                temperature=1.0,
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        # 映射到三个类别
        pred_id = map_t5_output_to_label(generated_text)
        
        # print(f"[{model_key} {shot}-shot] Output: '{generated_text}' → {id2label[pred_id]}")
        return pred_id
    
    else:
        # BERT-finetuned 等分类模型
        inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
        return pred_id
    
