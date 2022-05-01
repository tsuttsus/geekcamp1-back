# モデル格納庫

## 今一番良いモデル
- model1903.h5

## 人物
- 0:寺田蘭世
- 1:金村美玖
- 2:宮田愛萌
- 3:山口陽世
- 4:与田祐希

# モデル
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_23 (Conv2D)          (None, 64, 64, 32)        896       
                                                                 
 max_pooling2d_23 (MaxPoolin  (None, 32, 32, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_24 (Conv2D)          (None, 32, 32, 32)        9248      
                                                                 
 max_pooling2d_24 (MaxPoolin  (None, 16, 16, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_25 (Conv2D)          (None, 16, 16, 32)        9248      
                                                                 
 max_pooling2d_25 (MaxPoolin  (None, 8, 8, 32)         0         
 g2D)                                                            
                                                                 
 conv2d_26 (Conv2D)          (None, 8, 8, 32)          9248      
                                                                 
 max_pooling2d_26 (MaxPoolin  (None, 4, 4, 32)         0         
 g2D)                                                            
                                                                 
 flatten_5 (Flatten)         (None, 512)               0         
                                                                 
 dense_23 (Dense)            (None, 256)               131328    
                                                                 
 activation_23 (Activation)  (None, 256)               0         
                                                                 
 dense_24 (Dense)            (None, 96)                24672     
                                                                 
 activation_24 (Activation)  (None, 96)                0         
                                                                 
 dense_25 (Dense)            (None, 64)                6208      
                                                                 
 activation_25 (Activation)  (None, 64)                0         
                                                                 
 dense_26 (Dense)            (None, 5)                 325       
                                                                 
 activation_26 (Activation)  (None, 5)                 0         
                                                                 
=================================================================
Total params: 191,173
Trainable params: 191,173
Non-trainable params: 0
_________________________________________________________________
None


## メモ
- model1903.h5
    - 構成は参考サイトと同じ。
    - エポック:5
    - val_accu:0.5637
- model2101.h5
    - DNN 6層
    - エポック：11
    - val_accu:0.562

