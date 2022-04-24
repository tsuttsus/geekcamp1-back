import cv2
import datetime
import numpy as np
import sys
from tensorflow.keras.models import  load_model
from flask import Flask, render_template, request, redirect, abort, jsonify, make_response
import base64
import json

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    img_dir = "static/imgs/"
    if request.method == 'GET':
        img_path=None
        return render_template('index.html')
    elif request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        stream = request.files['img'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, 1)
        ### 画像処理
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        #顔が１つ以上検出された時
        if len(face_list) > 0:
            for rect in face_list:
                x,y,width,height=rect
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
                img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    print("too small")
                    continue
                img = cv2.resize(image,(64,64))
                img=np.expand_dims(img,axis=0)
                name=""
                model =load_model('models/model1903.h5')
                print(model.predict(img))
                nameNumLabel=np.argmax(model.predict(img))
                if nameNumLabel== 0:
                    name="Ranze"
                else:
                    name="sonota"
                cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
        #顔が検出されなかった時
        else:
            print("no face")
        
        #### 現在時刻を名前として「imgs/」に保存する
        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_path = img_dir + dt_now + ".jpg"
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        cv2.imwrite(img_path, image)
        result, dst_data = cv2.imencode('.jpg', image)
        qr_b64str = base64.b64encode(dst_data).decode("utf-8")
        qr_b64data = "data:image/png;base64,{}".format(qr_b64str)
        return render_template('kekka.html',img = qr_b64data)

@app.route('/test', methods=["POST"])
def test_api():
    if request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        stream = request.files.stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, 1)
        ### 画像処理
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        name_list = ['寺田蘭世', '金村美玖', '宮田愛萌', '山口陽世', '与田祐希', 'その他']
        predict_value = [0]*5
        nameNumLabel = 5
        #顔が１つ検出された時
        if len(face_list) == 1:
            for rect in face_list:
                x,y,width,height=rect
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
                img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    print("too small")
                    continue
                img = cv2.resize(image,(64,64))
                img=np.expand_dims(img,axis=0)
                name=""
                model =load_model('models/model1903.h5')
                print(model.predict(img))
                nameNumLabel=np.argmax(model.predict(img))
                predict_value = model.predict(img)
                predict_value = list(predict_value[0])
                if !(nameNumLabel >= 0 and nameNumLabel <=4):
                    nameNumLabel = 5
                cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
        #顔が複数検出されたとき
        else:
            print('no face')
        
        nameValue_dict = dict(zip(name_list[0:5], predict_value))
        json_data = json.dumps({"face": len(face_list), "name": name_list[nameNumLabel], "value": predict_value[nameNumLabel], "name_value": nameValue_dict})
        response = make_response(json_data)
        response.headers["Content-type"] = "application/json"
        response.headers["Access-Control-Allow-Origin"] = "*"
        
        return response

@app.route('/hello')
    json_data = json.dumps({"hello": "world"})
    response = make_response(json_data)

    return response

# エラーのハンドリング errorhandler(xxx)を指定、複数指定可能
# ここでは400,404をハンドリングする
@app.errorhandler(400)
@app.errorhandler(404)
def error_handler(error):
    # error.code: HTTPステータスコード
    # error.description: abortで設定したdict型
    return jsonify({'error': {
        'code': error.description['code'],
        'message': error.description['message']
    }}), error.code

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)