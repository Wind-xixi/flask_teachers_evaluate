# flask_backend/predictor.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json

# 模型路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "quantized_model")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 加载标签映射
with open(os.path.join(MODEL_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    label_data = json.load(f)
    label_map = {int(k): v for k, v in label_data["id2label"].items()}

def predict(text: str) -> str:
    """
    预测输入文本对应的标签（A-E 评分）
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_index = torch.argmax(logits, dim=1).item()
        label = label_map.get(predicted_index, "未知")
        return label
