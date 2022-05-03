# モデル格納庫

## 最新モデル
- model2-ss.h5
    - 2022/05/03

## 人物
- 0:寺田蘭世
- 1:金村美玖
- 2:宮田愛萌
- 3:山口陽世
- 4:与田祐希
- 5:影山優佳
- 6:森田ひかる
- 7:守屋麗奈
- 8:齋藤飛鳥

## サンプルコード
```python3
import numpy as np
from keras.models import  load_model

model = load_model('{モデル(*.h5)のパス}')

def detect_who(img):
    #予測
    name=""
    #print(model.predict(img))
    nameNumLabel=np.argmax(model.predict(img))
    #["terada","kanemura","miyata","yamaguchi","yoda","kageyama","morita","moriya","saito"]
    if nameNumLabel== 0: 
        name="Terada Ranze"
    elif nameNumLabel==1:
        name="Kanemura Miku"
    elif nameNumLabel==2:
        name="Miyata Manamo"
    elif nameNumLabel==3:
        name="Yamaguchi Haruyo"
    elif nameNumLabel==4:
        name="Yoda Yuuki"
    elif nameNumLabel==5:
        name="Kageyama Yuuka"
    elif nameNumLabel==6:
        name="Morita Hikaru"
    elif nameNumLabel==7:
        name="Moriya Rena"
    elif nameNumLabel==8:
        name="Saito Asuka"
    return name
```

## メモ
- old/model1903.h5
    - 構成は参考サイトと同じ。
    - エポック:5
    - val_accu:0.5637
- old/model2101.h5
    - DNN 6層
    - エポック：11
    - val_accu:0.562

