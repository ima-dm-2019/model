from flask import Flask, request, jsonify
import json
from model.model_km import Model

"""
    "url": 'http://127.0.0.1:5000/model_fit/'
    "Request Method": Post
    "Request parameters":  "格式：json（表单）   数据类型：表单data{
                                                            'weight': tf_idf矩阵(转列表tolist()),
                                                            'data': 各个文本的词集组成的列表}

    "status": 202,
    "message": "success",
    "result": "聚类后的结果（不同数字表示）",
    "title": "表示每种簇的主题， key：result中的数字， value：主题"
    
    "status": 404,
    "message": "failure",
    "result": "None",
    "title": "None"
"""



app = Flask(__name__)
app.debug = True
model = Model()

@app.route('/model_fit/', methods=['post'])
def fit_predict():
    if not request.data:  # 检测是否有数据
        json_response = {
            'status': 404,
            'result': None,
            'title': None
        }
        return jsonify(json_response)
    student = request.data.decode('utf-8')
    student_json = json.loads(student)  # 转化为Python类型
    result = model.fit_predict(student_json['weight'], student_json['data'])
    result = [float(i) for i in result]  # 转化格式不然无法序列化会报错
    title = model.return_title()
    json_response = {
        'status': 202,
        'result': result,
        'title': title
    }
    return jsonify(json_response)  # 返回JSON数据。

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # 指定地址和端口号