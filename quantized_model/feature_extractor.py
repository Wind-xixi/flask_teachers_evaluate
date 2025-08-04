from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = Path("/flask_backend/quantized_model")
print("Model folder contents:", list(model_dir.iterdir()))

config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

model_weights_path = model_dir / "model.safetensors"
state_dict = torch.load(model_weights_path, map_location="cpu", weights_only=False)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    config=config,
    state_dict=state_dict,
    local_files_only=True
)
model.eval()

print("模型加载成功！")
