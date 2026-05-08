# nliEvaluator.py
import torch, random
from utils import id2label, verbalizer, map_t5_output_to_label, load_few_shot_examples

# 全局加载示例池（只加载一次）
FEW_SHOT_EXAMPLE_POOL = load_few_shot_examples("few-shots.jsonl")

def sample_few_shot_examples(pool, shot, seed=None):
    """
    随机抽样策略：
    - 每类至少取 shot // 3 个（向上取整）
    - 再从剩余中随机补充
    """
    if seed is not None:
        random.seed(seed)
    
    examples = []
    min_per_class = 1
    labels = ["entailment", "neutral", "contradiction"]
    
    # 按标签分组
    pool_by_label = {label: [] for label in labels}
    for ex in pool:
        pool_by_label[ex["label"]].append(ex)
    
    # 每类取 min_per_class 个
    for label in labels:
        candidates = pool_by_label[label].copy()
        random.shuffle(candidates)
        examples.extend(candidates[:min_per_class])
    
    # 如果还不够，从合并的剩余池中随机补充
    if len(examples) < shot:
        remaining = [ex for ex in pool if ex not in examples]
        random.shuffle(remaining)
        examples.extend(remaining[:shot - len(examples)])
    
    # 打乱顺序
    random.shuffle(examples)
    
    return examples[:shot]


def create_fewshot_prompt(premise, hypothesis, shot=0, seed=None):
    f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    # 添加 few-shot 示例（使用随机抽样）
    if shot > 0:
        examples = sample_few_shot_examples(FEW_SHOT_EXAMPLE_POOL, shot, seed)
        
        prompt += "Here are some examples:\n\n"
        for ex in examples:
            prompt += f"Premise: {ex['premise']}\n"
            prompt += f"Hypothesis: {ex['hypothesis']}\n"
            prompt += f"Answer: {ex['label']}\n\n"
        prompt += "Now analyze the following pair:\n\n"
    
    prompt += "\nThe relationship is:"
    
    return prompt

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
def create_t5_prompt(premise, hypothesis, shot=0, seed=None):
    """T5 seq2seq prompt with random sampling"""
    
    f"nli premise: {premise} hypothesis: {hypothesis}"
    
    if shot > 0:
        examples = sample_few_shot_examples(FEW_SHOT_EXAMPLE_POOL, shot, seed)
        for ex in examples:
            prompt += f"premise: {ex['premise']} hypothesis: {ex['hypothesis']} answer: {ex['label']} "
    
    # 当前要预测的样本
    prompt += " answer:"
    
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
        prompt = create_t5_prompt(premise, hypothesis, shot)
        
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
    
