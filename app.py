import os
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# --- 全局变量 ---
ort_session, vectorizer, scaler = None, None, None

def load_models():
    """在应用启动时加载所有教师评价模型和文件。"""
    global ort_session, vectorizer, scaler
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'quantized_model.onnx')
        vectorizer_path = os.path.join(base_dir, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')

        print("Checking for teacher evaluation model files...")
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, scaler_path]):
            raise FileNotFoundError("One or more teacher model files are missing.")
        print("All teacher model files found.")

        print("Loading models for teacher evaluation...")
        ort_session = ort.InferenceSession(model_path)
        vectorizer = joblib.load(vectorizer_path)
        scaler = joblib.load(scaler_path)
        print("--- Teacher evaluation models loaded successfully! ---")
        return True
    except Exception as e:
        print(f"!!!!!!!!!! FATAL ERROR DURING MODEL LOADING: {e} !!!!!!!!!!!")
        ort_session, vectorizer, scaler = None, None, None
        return False

# --- 在应用启动时，立即调用加载函数 ---
load_models()

def preprocess_text(text):
    """对输入的文本进行分词和空格拼接处理"""
    try:
        jieba.initialize()
    except:
        pass
    return [" ".join(jieba.cut(text))]

# --- 教师评价接口 ---
@app.route('/evaluate_teacher', methods=['POST'])
def evaluate_teacher():
    if not all([ort_session, vectorizer, scaler]):
        return jsonify({"error": "Teacher model is not available due to a server loading error."}), 500

    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if 'text' not in data or not isinstance(data['text'], str):
            return jsonify({"error": "Missing or invalid 'text' field"}), 400

        text_to_evaluate = data['text']
        
        processed_text = preprocess_text(text_to_evaluate)
        tfidf_features = vectorizer.transform(processed_text).toarray()
        scaled_features = scaler.transform(tfidf_features)
        
        input_data = scaled_features.astype(np.float32)

        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        scores = ort_outs[0][0]
        grades = ['A', 'B', 'C', 'D', 'E']
        predicted_grade = grades[np.argmax(scores)]
        
        result = {
            "grade": predicted_grade,
            "summary": f"AI分析完成。根据文本内容，初步评定等级为 {predicted_grade}。"
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during teacher evaluation: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

