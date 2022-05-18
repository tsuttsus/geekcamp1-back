import cv2
import datetime
import numpy as np
import sys
from tensorflow.keras.models import  load_model
from flask import Flask, render_template, request, redirect, abort, jsonify, make_response
import base64
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    supports_credentials=True
)

# 顔の個数チェック
@app.route('/start', methods=["GET", "POST"])
def face_check():
    # パスの指定する
    img_path = "static/uploads/uploads.jpg"
    if request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        stream = request.files['img'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, 1)
        ### 画像処理
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        # 画像の保存
        cv2.imwrite(img_path, image)
        # グレイスケール
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        error = ''
        if len(face_list) > 1:
            error = '顔が複数検出されました'
        elif len(face_list) == 0:
            error = '顔が検出されませんでした'

        json_data = json.dumps({"error": error, 
                                "similarActors": [{
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/kanemuramiku.jpeg",
                                        "name": "金村美玖",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/miyatamanamo.jpeg",
                                        "name": "宮田愛萌",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/yamaguchiharuyo.jpeg",
                                        "name": "山口陽世",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/yodayuki.jpeg",
                                        "name": "与田祐希",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/kageyamayuka.jpeg",
                                        "name": "影山優佳",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/moritahikaru.jpeg",
                                        "name": "森田ひかる",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/moriyarena.jpeg",
                                        "name": "守屋麗奈",
                                        "percent": 0
                                    },
                                    {
                                        "src": "https://whispering-hollows-31833.herokuapp.com/static/saitouasuka.jpeg",
                                        "name": "齋藤飛鳥",
                                        "percent": 0
                                    }],
                                }, ensure_ascii=False)
        response = make_response(json_data)
        return response


@app.route('/result', methods=["POST"])
def test_api():
    if request.method == 'POST':
        # json処理
        jsonform = request.json
        select_name=['寺田蘭世']
        for n in range(4):
            select_name.append(jsonform[n]["name"])
        select_prob=[]

        image = cv2.imread('static/uploads/uploads.jpg')
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
        name_list = ['寺田蘭世', '金村美玖', '宮田愛萌', '山口陽世', '与田祐希', '影山優佳', '森田ひかる', '守屋麗奈', '齋藤飛鳥']
        predict_value = [0]*9
        nameNumLabel = 9
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
                model =load_model('models/model2-ss.h5')
                print(model.predict(img))
                nameNumLabel=np.argmax(model.predict(img))
                predict_value = model.predict(img)
                predict_value = list(predict_value[0])
                cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
                #出力処理
                for targ_name in select_name:
                    targ_index=name_list.index(targ_name)
                    select_prob.append(predict_value[targ_index])
                #正規化処理
                mins = min(select_prob) 
                maxs=max(select_prob)
                for i, x in enumerate(select_prob):
                    select_prob[i] = (x-mins) / (maxs-mins)
                
        # カラーに戻して保存する
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        cv2.imwrite('static/uploads/result.jpg', image)
        jsonform.src = "https://whispering-hollows-31833.herokuapp.com/static/moritahikaru.jpeg"

        nameValue_dict = dict(zip(name_list[0:5], predict_value))
        #json_data = json.dumps({"face": len(face_list), "name": name_list[nameNumLabel]}, ensure_ascii=False)
        #Jsonに確率を代入
        if len(select_prob)>0:
            for n in range(5):
                jsonform["similarActors"][n]["percent"]=select_prob[n]
        json_data = json.dumps(jsonform)
        
        # "value": predict_value[nameNumLabel], "name_value": nameValue_dict})
        response = make_response(json_data)
        # response.headers["Content-type"] = "application/json"
        # response.headers["Access-Control-Allow-Origin"] = "*"
        # response.headers["Access-Control-Request-Method"] = "GET,POST,HEAD"
        
        return response

@app.route('/hello')
def hello():
    hello = "Hi! hello"

    return hello

if __name__ == "__main__":
    app.run(debug=True)