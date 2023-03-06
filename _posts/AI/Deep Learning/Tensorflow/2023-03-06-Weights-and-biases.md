---
title: "[Tensorflow] 모델의 가중치 확인"
date: 2023-03-06

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## 모델 가중치 확인
모델을 생성하면 모델에 구성되어 있는 레이어와 레이어 안에 들어 있는 가중치와 biase를 확인할 수 있다.


```python
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
```


```python
# 모델 생성
inputs = Input(shape=(28, 28, 1))
x = Flatten(input_shape=(28, 28, 1))(inputs)
x = Dense(units=300, activation='relu')(x)
x = Dense(units=100, activation='relu')(x)
x = Dense(units=10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                     
     flatten_1 (Flatten)         (None, 784)               0         
                                                                     
     dense_3 (Dense)             (None, 300)               235500    
                                                                     
     dense_4 (Dense)             (None, 100)               30100     
                                                                     
     dense_5 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# 하나의 레이어에 접근
model.layers
```




    [<keras.engine.input_layer.InputLayer at 0x20d39983340>,
     <keras.layers.core.flatten.Flatten at 0x20d39983700>,
     <keras.layers.core.dense.Dense at 0x20d399838e0>,
     <keras.layers.core.dense.Dense at 0x20d39983a00>,
     <keras.layers.core.dense.Dense at 0x20d39a43d30>]




```python
hidden_2 = model.layers[2]
hidden_2.name
```




    'dense_3'




```python
# dense_3 레이어가 hidden 레이어가 맞는지
model.get_layer('dense_3') is hidden_2
```




    True




```python
weights, biases = hidden_2.get_weights()
print(weights.shape)
print(biases.shape)
```

    (784, 300)
    (300,)
    


```python
# 가중치 값
print(weights)
```

    [[ 0.04680964 -0.01588077  0.04116296 ...  0.02233758 -0.01330332
       0.03937691]
     [-0.05316085 -0.01026929 -0.02134814 ... -0.04345966  0.07150589
       0.04673789]
     [-0.02852385  0.04749756  0.0213156  ... -0.00167229 -0.00253534
      -0.06003164]
     ...
     [ 0.01177286 -0.00260537 -0.02730675 ... -0.05742957 -0.05629922
      -0.00065816]
     [ 0.03312647  0.05357176  0.05823182 ... -0.04483525 -0.02097017
       0.04719541]
     [ 0.00601588  0.03708929 -0.0306447  ...  0.00033265  0.03019155
       0.06975248]]
    


```python
# biases
print(biases)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    

