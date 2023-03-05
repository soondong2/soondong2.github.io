---
title: "[Tensorflow] 함수형 API Model"
date: 2023-03-05

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## Model
- `Sequential()`
- 서브 클래싱(Subclassing)
- 함수형 API

### 함수형 API
- 가장 권장되는 방법
- 모델을 복잡하고 유연하게 구성 가능
- 다중 입출력을 다룰 수 있음


```python
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
     input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                     
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense (Dense)               (None, 300)               235500    
                                                                     
     dense_1 (Dense)             (None, 100)               30100     
                                                                     
     dense_2 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________
    


```python
plot_model(model)
```

![image](https://user-images.githubusercontent.com/100760303/222957716-41f2c5f5-b0dc-40d4-99f5-4e62635ddea2.png)


#### 복잡한 모델 구성 방법


```python
from tensorflow.keras.layers import Concatenate
```


```python
input_layer = Input(shape=(28, 28))
hidden1 = Dense(units=100, activation='relu')(input_layer)
hidden2 = Dense(units=30, activation='relu')(hidden1)

# input_layer와 hidden2를 concat
concat = Concatenate()([input_layer, hidden2])
# 최종적으로 output은 dense layer로 하나만 고름
output = Dense(units=1)(concat)

model = Model(inputs=[input_layer], outputs=[output])
model.summary()
```

    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 28, 28)]     0           []                               
                                                                                                      
     dense_3 (Dense)                (None, 28, 100)      2900        ['input_2[0][0]']                
                                                                                                      
     dense_4 (Dense)                (None, 28, 30)       3030        ['dense_3[0][0]']                
                                                                                                      
     concatenate (Concatenate)      (None, 28, 58)       0           ['input_2[0][0]',                
                                                                      'dense_4[0][0]']                
                                                                                                      
     dense_5 (Dense)                (None, 28, 1)        59          ['concatenate[0][0]']            
                                                                                                      
    ==================================================================================================
    Total params: 5,989
    Trainable params: 5,989
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
plot_model(model)
```

![image](https://user-images.githubusercontent.com/100760303/222957739-325dd642-65c8-48fd-b05b-d786cfe21879.png)




#### 여러 개의 Input을 가지는 모델


```python
# 입력층
input_1 = Input(shape=(10, 10), name='input_1')
input_2 = Input(shape=(10, 28), name='input_2')

# 은닉층
hidden1 = Dense(units=100, activation='relu')(input_2)
hidden2 = Dense(units=10, activation='relu')(hidden1)

# input_1과 hidden2를 Concatenate
concat = Concatenate()([input_1, hidden2])
output = Dense(units=1, activation='sigmoid', name='output')(concat)

model = Model(inputs=[input_1, input_2], outputs=[output])
model.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 10, 28)]     0           []                               
                                                                                                      
     dense_6 (Dense)                (None, 10, 100)      2900        ['input_2[0][0]']                
                                                                                                      
     input_1 (InputLayer)           [(None, 10, 10)]     0           []                               
                                                                                                      
     dense_7 (Dense)                (None, 10, 10)       1010        ['dense_6[0][0]']                
                                                                                                      
     concatenate_1 (Concatenate)    (None, 10, 20)       0           ['input_1[0][0]',                
                                                                      'dense_7[0][0]']                
                                                                                                      
     output (Dense)                 (None, 10, 1)        21          ['concatenate_1[0][0]']          
                                                                                                      
    ==================================================================================================
    Total params: 3,931
    Trainable params: 3,931
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
# 입력은 두 개인데 출력은 한 개
plot_model(model)
```

![image](https://user-images.githubusercontent.com/100760303/222957771-5b4febf3-0171-4895-86a3-3f4e36b53e50.png)


#### 여러 개의 출력을 가지는 모델


```python
# 입력층
input_ = Input(shape=(10, 10), name='input')

# 은닉층
hidden1 = Dense(units=100, activation='relu')(input_)
hidden2 = Dense(units=10, activation='relu')(hidden1)

# 출력층
output = Dense(units=1, activation='sigmoid', name='main_output')(hidden2)
sub_out = Dense(units=1, name='sum_output')(hidden2)

model = Model(inputs=[input_], outputs=[output, sub_out])
model.summary()
```

    Model: "model_3"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input (InputLayer)             [(None, 10, 10)]     0           []                               
                                                                                                      
     dense_8 (Dense)                (None, 10, 100)      1100        ['input[0][0]']                  
                                                                                                      
     dense_9 (Dense)                (None, 10, 10)       1010        ['dense_8[0][0]']                
                                                                                                      
     main_output (Dense)            (None, 10, 1)        11          ['dense_9[0][0]']                
                                                                                                      
     sum_output (Dense)             (None, 10, 1)        11          ['dense_9[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 2,132
    Trainable params: 2,132
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
# 입력은 하나인데 출력은 두 개
plot_model(model)
```

![image](https://user-images.githubusercontent.com/100760303/222957795-994c2f88-3107-498f-aec8-1637548d7d6e.png)


```python
input_1 = Input(shape=(10, 10), name='input_1')
input_2 = Input(shape=(10, 28), name='input_2')

hidden1 = Dense(units=100, activation='relu')(input_2)
hidden2 = Dense(units=10, activation='relu')(hidden1)

concat = Concatenate()([input_1, hidden2])

output = Dense(units=1, activation='sigmoid', name='main_output')(concat)
sub_out = Dense(units=1, name='sum_output')(hidden2)

model = Model(inputs=[input_1, input_2], outputs=[output, sub_out])
model.summary()
```

    Model: "model_6"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 10, 28)]     0           []                               
                                                                                                      
     dense_14 (Dense)               (None, 10, 100)      2900        ['input_2[0][0]']                
                                                                                                      
     input_1 (InputLayer)           [(None, 10, 10)]     0           []                               
                                                                                                      
     dense_15 (Dense)               (None, 10, 10)       1010        ['dense_14[0][0]']               
                                                                                                      
     concatenate_4 (Concatenate)    (None, 10, 20)       0           ['input_1[0][0]',                
                                                                      'dense_15[0][0]']               
                                                                                                      
     main_output (Dense)            (None, 10, 1)        21          ['concatenate_4[0][0]']          
                                                                                                      
     sum_output (Dense)             (None, 10, 1)        11          ['dense_15[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 3,942
    Trainable params: 3,942
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
plot_model(model)
```
![image](https://user-images.githubusercontent.com/100760303/222957823-bb71e460-900b-4cdb-8d2b-a3261f3ff518.png)
