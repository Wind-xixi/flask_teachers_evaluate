from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
from sentiment_api.models import DialogueEvaluator

UPLOAD_FOLDER = 'uploads'
MODEL_DIR = Path("/flask_backend/quantized_model")

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化模型（只写一次）
evaluator = DialogueEvaluator(MODEL_DIR)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 后面可以定义你的 API 路由，比如 /predict 等


def format_report(results, file_path):
    lines = []
    lines.append(f"📊 学术对话评价报告：{file_path}")

    if results['status'] == 'no_key_sentences':
        lines.append("⚠️ 未检测到包含学术关键词的句子")
        return '\n'.join(lines)

    lines.append(f"\n🔍 共找到 {results['overall_stats']['total_sentences']} 个关键评价句子")
    lines.append("\n📈 整体统计:")
    lines.append(f"- 平均置信度: {results['overall_stats']['avg_confidence']:.2f}")
    lines.append(f"- 模型大小: {results['model_size_mb']:.2f} MB")

    lines.append("- 评价等级分布:")
    for label, count in results['overall_stats']['label_distribution'].items():
        lines.append(f"  {label}: {count} 句")

    lines.append("- 场景分布:")
    for scene, count in results['overall_stats']['scene_distribution'].items():
        lines.append(f"  {scene}: {count} 次")

    lines.append("- 情感分布:")
    for sentiment, count in results['overall_stats']['sentiment_distribution'].items():
        lines.append(f"  {sentiment}: {count} 次")

    lines.append("\n🏆 评分最高的3个关键句子:")
    top_sentences = sorted(results['key_sentences'], key=lambda x: x['confidence'], reverse=True)[:3]
    for i, sent in enumerate(top_sentences, 1):
        lines.append(f"\n{i}. 置信度: {sent['confidence']:.2f} | 等级: {sent['label']}")
        lines.append(f"匹配信息: {json.dumps(sent['matched_info'], ensure_ascii=False)}")
        lines.append(f"句子: {sent['sentence'][:200]}...")

    lines.append("\n⚠️ 评分最低的3个关键句子:")
    bottom_sentences = sorted(results['key_sentences'], key=lambda x: x['confidence'])[:3]
    for i, sent in enumerate(bottom_sentences, 1):
        lines.append(f"\n{i}. 置信度: {sent['confidence']:.2f} | 等级: {sent['label']}")
        lines.append(f"匹配信息: {json.dumps(sent['matched_info'], ensure_ascii=False)}")
        lines.append(f"句子: {sent['sentence'][:200]}...")

    return '\n'.join(lines)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'status': 'fail', 'error': '未找到上传文件', 'details': '请求中缺少 file 字段'}), 400

        file = request.files['file']

        # 检查文件是否有名称
        if file.filename == '':
            return jsonify({'status': 'fail', 'error': '未选择文件', 'details': '上传的文件名为空'}), 400

        # 确保文件名安全
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 保存文件
        file.save(save_path)
        print(f"文件已保存至: {save_path}")

        # 读取文件内容
        with open(save_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 处理文件内容
        results = evaluator.evaluate_text(content)
        report_text = format_report(results, filename)

        return jsonify({
            'status': 'success',
            'report': report_text,
            'file_path': save_path,
            'filename': filename
        }), 200

    except FileNotFoundError as e:
        return jsonify({
            'status': 'fail',
            'error': '文件操作错误',
            'details': f'文件未找到: {str(e)}'
        }), 500
    except UnicodeDecodeError as e:
        return jsonify({
            'status': 'fail',
            'error': '编码错误',
            'details': f'无法以 UTF-8 编码读取文件: {str(e)}'
        }), 500
    except Exception as e:
        # 捕获所有其他异常
        return jsonify({
            'status': 'fail',
            'error': '处理请求时出错',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)