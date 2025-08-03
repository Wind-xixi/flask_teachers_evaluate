import os
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# --- 模型和文件的路径 ---
# 使用相对路径，确保在任何地方都能找到文件
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'quantized_model.onnx')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# --- 全局加载模型 ---
try:
    if not all([os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, SCALER_PATH]]):
        raise FileNotFoundError("CRITICAL: One or more model files are missing from the repository.")

    ort_session = ort.InferenceSession(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("All models and dependencies loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")
    ort_session = None

def preprocess_text(text):
    words = " ".join(jieba.cut(text))
    return [words]

@app.route('/evaluate_teacher', methods=['POST'])
def evaluate_teacher():
    if not ort_session:
        return jsonify({"error": "Model is not available due to a server loading error."}), 500

    try:
        # 确保请求中有JSON数据
        if not request.is_json:
            return jsonify({"error": "Invalid request: Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field in JSON body"}), 400

        text_to_evaluate = data['text']
        
        processed_text = preprocess_text(text_to_evaluate)
        tfidf_features = vectorizer.transform(processed_text).toarray()
        scaled_features = scaler.transform(tfidf_features)
        
        input_data = scaled_features.astype(np.float32)

        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        scores = ort_outs[0][0]
        
        # 假设模型输出格式为 [等级, 总结] 或其他形式，请根据你的实际情况调整
        # 这里我们先用一个模拟的返回结构
        result = {
            "grade": "A",
            "summary": f"分析完成: {text_to_evaluate[:30]}..."
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

# 注意：没有 if __name__ == '__main__': ...
