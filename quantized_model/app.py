import os
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer # 导入正确的“翻译官”

app = Flask(__name__)

# --- 全局变量 ---
ort_session = None
tokenizer = None

def load_models():
    """在应用启动时加载ONNX模型和对应的Hugging Face分词器。"""
    global ort_session, tokenizer
    try:
        # 假设app.py和模型/分词器文件都在同一个目录中
        # Procfile中的 --chdir 参数会确保这一点
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- 核心修正：使用AutoTokenizer加载分词器 ---
        # 它会自动寻找tokenizer.json, config.json, vocab.txt等文件
        print("Loading Hugging Face Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(current_dir)
        print("Tokenizer loaded successfully.")

        # --- 加载ONNX模型 ---
        model_path = os.path.join(current_dir, 'model_quantized.onnx') # 确保你的模型叫这个名字
        
        print(f"Checking for ONNX model at: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at path: {model_path}")
        
        print("Loading ONNX session...")
        ort_session = ort.InferenceSession(model_path)
        print("--- All models and tokenizers loaded successfully! ---")
        return True

    except Exception as e:
        print(f"!!!!!!!!!! FATAL ERROR DURING LOADING: {e} !!!!!!!!!!!")
        ort_session, tokenizer = None, None
        return False

# --- 在应用启动时，立即调用加载函数 ---
load_models()

# --- 教师评价接口 ---
@app.route('/evaluate_teacher', methods=['POST'])
def evaluate_teacher():
    if not all([ort_session, tokenizer]):
        return jsonify({"error": "Model or Tokenizer is not available due to a server loading error."}), 500

    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if 'text' not in data or not isinstance(data['text'], str):
            return jsonify({"error": "Missing or invalid 'text' field"}), 400

        text_to_evaluate = data['text']
        
        # --- 使用分词器进行“翻译” ---
        inputs = tokenizer(text_to_evaluate, return_tensors="np", padding=True, truncation=True, max_length=512)
        
        # ONNX模型通常需要一个字典作为输入
        # 输入的键名（如'input_ids', 'attention_mask'）需要和模型导出时的定义完全一致
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        # 如果你的模型还需要 'token_type_ids'，则取消下面一行的注释
        # ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)

        # --- 运行模型 ---
        ort_outs = ort_session.run(None, ort_inputs)
        
        # --- 解析结果 ---
        # 通常，分类模型的输出是一个logits数组
        logits = ort_outs[0][0]
        grades = ['A', 'B', 'C', 'D', 'E'] # 确保这个顺序和训练时一致
        predicted_index = np.argmax(logits)
        predicted_grade = grades[predicted_index]
        
        result = {
            "grade": predicted_grade,
            "summary": f"AI分析完成。根据文本内容，初步评定等级为 {predicted_grade}。"
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during teacher evaluation: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

    except Exception as e:
        print(f"Error during teacher evaluation: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500
