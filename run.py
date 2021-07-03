import cv2
import datetime
import numpy as np
import sys
from tensorflow.keras.models import  load_model
from flask import Flask, render_template, request
import base64

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
        if image is None:
            print("Not open:")
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
                model =load_model('ranze_model.h5')
                print(model.predict(img))
                nameNumLabel=np.argmax(model.predict(img))
                if nameNumLabel== 0:
                    name="Ranze"
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


if __name__ == "__main__":
    app.run(debug=True)