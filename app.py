import os
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# --- 全局变量，用于存储加载的模型 ---
ort_session = None
vectorizer = None
scaler = None

# --- 关键修复：创建一个专门的函数来加载模型，并提供详细的日志 ---
def load_models():
    """
    在应用启动时加载所有必要的模型和文件。
    如果任何文件加载失败，将打印详细错误并返回False。
    """
    global ort_session, vectorizer, scaler
    
    try:
        # 1. 构建所有文件的绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'quantized_model.onnx')
        vectorizer_path = os.path.join(base_dir, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')

        # 2. 逐一检查文件是否存在
        print("Checking for model files...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at: {vectorizer_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        print("All files found.")

        # 3. 逐一加载模型，这样出错时能知道是哪个环节的问题
        print("Loading ONNX session...")
        ort_session = ort.InferenceSession(model_path)
        print("ONNX session loaded.")

        print("Loading TF-IDF vectorizer...")
        vectorizer = joblib.load(vectorizer_path)
        print("Vectorizer loaded.")

        print("Loading scaler...")
        scaler = joblib.load(scaler_path)
        print("Scaler loaded.")
        
        print("--- All models and dependencies loaded successfully! ---")
        return True

    except Exception as e:
        # 捕获任何异常，并打印非常详细的错误信息
        print(f"!!!!!!!!!! FATAL ERROR DURING MODEL LOADING !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 将所有模型变量设为None，确保服务知道加载失败
        ort_session = None
        vectorizer = None
        scaler = None
        return False

# --- 在应用启动时，立即调用加载函数 ---
load_models()


def preprocess_text(text):
    """对输入的文本进行分词和空格拼接处理"""
    # 确保jieba正确初始化
    try:
        jieba.initialize()
    except:
        pass
    words = " ".join(jieba.cut(text))
    return [words]

@app.route('/evaluate_teacher', methods=['POST'])
def evaluate_teacher():
    # 每次请求都检查模型是否已成功加载
    if not all([ort_session, vectorizer, scaler]):
        return jsonify({"error": "Model is not available due to a server loading error."}), 500

    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request: Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'text' not in data or not isinstance(data['text'], str):
            return jsonify({"error": "Missing or invalid 'text' field in JSON body"}), 400

        text_to_evaluate = data['text']
        
        # --- 执行完整的预测流程 ---
        processed_text = preprocess_text(text_to_evaluate)
        tfidf_features = vectorizer.transform(processed_text).toarray()
        scaled_features = scaler.transform(tfidf_features)
        
        input_data = scaled_features.astype(np.float32)

        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # 假设模型的输出是一个包含多个分数的数组
        scores = ort_outs[0][0] 
        
        # --- 关键：根据你的模型输出，将scores转换为有意义的结果 ---
        # 这是一个示例，你需要根据你的模型实际输出来修改它
        # 假设模型输出5个分数，分别对应ABCDE五个等级的概率
        grades = ['A', 'B', 'C', 'D', 'E']
        predicted_grade = grades[np.argmax(scores)]
        
        result = {
            "grade": predicted_grade,
            "summary": f"AI分析完成。根据文本内容，初步评定等级为 {predicted_grade}。"
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

