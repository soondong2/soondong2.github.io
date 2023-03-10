---
title: "[Tensorflow] MNIST 딥러닝 예제"
date: 2023-03-07

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## MNIST 딥러닝 예제

### Library Import


```python
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

### Data Load


```python
# seed 고정
tf.random.set_seed(111)
```


```python
# Data Load
(X_train_full, y_train_full), (X_test, y_test) = load_data(path='mnist.npz')

# Train Data 중 30%를 Validation Data로 활용
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                  test_size=0.3, random_state=111)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 1s 0us/step
    11501568/11490434 [==============================] - 1s 0us/step
    


```python
num_x_train = (X_train.shape[0])
num_x_val = (X_val.shape[0])
num_x_test = (X_test.shape[0])

print('학습 데이터: {}\t레이블: {}'.format(X_train_full.shape, y_train_full.shape))
print('학습 데이터: {}\t레이블: {}'.format(X_train_full.shape, y_train_full.shape))
print('검증 데이터: {}\t레이블: {}'.format(X_val.shape, y_val.shape))
print('테스트 데이터: {}\t레이블: {}'.format(X_test.shape, y_test.shape))
```

    학습 데이터: (60000, 28, 28)	레이블: (60000,)
    학습 데이터: (60000, 28, 28)	레이블: (60000,)
    검증 데이터: (18000, 28, 28)	레이블: (18000,)
    테스트 데이터: (10000, 28, 28)	레이블: (10000,)
    

임의의 샘플 5개 확인


```python
# 60000개에서 5개를 랜덤하게 추출
num_sample = 5
random_idx = np.random.randint(60000, size=num_sample)

plt.figure(figsize=(15, 3))

for i, idx in enumerate(random_idx):
    img = X_train_full[idx, :]
    label = y_train_full[idx] # 0 ~ 9

    # 그래프
    plt.subplot(1, len(random_idx), i + 1)
    plt.imshow(img)
    plt.title('Index: {}, Label: {}'.format(idx, label))
```

![image](https://user-images.githubusercontent.com/100760303/223306642-7e3bd709-c981-47a3-a03e-a1afb2a39588.png)



### Preprocessing
0 ~ 255 (256개)로 이루어진 흑백 이미지 이므로 normalization을 해준다.


```python
# 0 ~ 1 사이의 값으로 만들기
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# to_categorical을 이용해 정답을 categorical로 변환
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
```

### 모델 구성(Sequential)


```python
model = Sequential([Input(shape=(28, 28), name='input'),
                    Flatten(input_shape=(28, 28), name='flatten'),
                    Dense(units=100, activation='relu', name='dense1'),
                    Dense(units=64, activation='relu', name='dense2'),
                    Dense(units=32, activation='relu', name='dense3'),
                    Dense(units=10, activation='softmax', name='output')])
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense1 (Dense)              (None, 100)               78500     
                                                                     
     dense2 (Dense)              (None, 64)                6464      
                                                                     
     dense3 (Dense)              (None, 32)                2080      
                                                                     
     output (Dense)              (None, 10)                330       
                                                                     
    =================================================================
    Total params: 87,374
    Trainable params: 87,374
    Non-trainable params: 0
    _________________________________________________________________
    

### 모델 구성(함수형 API)


```python
# 함수형 API
inputs = Input(shape=(28, 28), name='input')
x = Flatten(input_shape=(28, 28), name='flatten')(inputs)
x = Dense(units=100, activation='relu', name='dense1')(x)
x = Dense(units=64, activation='relu', name='dense2')(x)
x = Dense(units=32, activation='relu', name='dense3')(x)
x = Dense(units=10, activation='softmax', name='output')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 28, 28)]          0         
                                                                     
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense1 (Dense)              (None, 100)               78500     
                                                                     
     dense2 (Dense)              (None, 64)                6464      
                                                                     
     dense3 (Dense)              (None, 32)                2080      
                                                                     
     output (Dense)              (None, 10)                330       
                                                                     
    =================================================================
    Total params: 87,374
    Trainable params: 87,374
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# 모델의 입력과 출력
plot_model(model, show_shapes=True)
```
![eunji_16_0](https://user-images.githubusercontent.com/100760303/223306748-03b166e2-b33f-4c80-82b4-d9e3e42a2329.png)

 


### 모델 컴파일 및 학습


```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```


```python
# history를 통해 모델의 학습 결과를 받아옴
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_data=(X_val, y_val))
```

    Epoch 1/50
    329/329 [==============================] - 8s 14ms/step - loss: 1.6564 - accuracy: 0.5211 - val_loss: 0.9154 - val_accuracy: 0.7944
    Epoch 2/50
    329/329 [==============================] - 5s 14ms/step - loss: 0.6523 - accuracy: 0.8350 - val_loss: 0.5028 - val_accuracy: 0.8645
    Epoch 3/50
    329/329 [==============================] - 3s 10ms/step - loss: 0.4500 - accuracy: 0.8771 - val_loss: 0.4035 - val_accuracy: 0.8857
    Epoch 4/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.3804 - accuracy: 0.8937 - val_loss: 0.3610 - val_accuracy: 0.8963
    Epoch 5/50
    329/329 [==============================] - 3s 10ms/step - loss: 0.3423 - accuracy: 0.9031 - val_loss: 0.3347 - val_accuracy: 0.9039
    Epoch 6/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.3169 - accuracy: 0.9102 - val_loss: 0.3239 - val_accuracy: 0.9046
    Epoch 7/50
    329/329 [==============================] - 2s 6ms/step - loss: 0.2970 - accuracy: 0.9150 - val_loss: 0.3097 - val_accuracy: 0.9081
    Epoch 8/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.2808 - accuracy: 0.9193 - val_loss: 0.2772 - val_accuracy: 0.9196
    Epoch 9/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.2669 - accuracy: 0.9243 - val_loss: 0.2667 - val_accuracy: 0.9222
    Epoch 10/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.2547 - accuracy: 0.9269 - val_loss: 0.2575 - val_accuracy: 0.9238
    Epoch 11/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.2438 - accuracy: 0.9300 - val_loss: 0.2469 - val_accuracy: 0.9273
    Epoch 12/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.2342 - accuracy: 0.9329 - val_loss: 0.2927 - val_accuracy: 0.9085
    Epoch 13/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.2254 - accuracy: 0.9352 - val_loss: 0.2305 - val_accuracy: 0.9335
    Epoch 14/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.2170 - accuracy: 0.9381 - val_loss: 0.2384 - val_accuracy: 0.9294
    Epoch 15/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.2091 - accuracy: 0.9396 - val_loss: 0.2210 - val_accuracy: 0.9337
    Epoch 16/50
    329/329 [==============================] - 3s 9ms/step - loss: 0.2022 - accuracy: 0.9418 - val_loss: 0.2130 - val_accuracy: 0.9387
    Epoch 17/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.1953 - accuracy: 0.9439 - val_loss: 0.2119 - val_accuracy: 0.9399
    Epoch 18/50
    329/329 [==============================] - 5s 16ms/step - loss: 0.1887 - accuracy: 0.9457 - val_loss: 0.2098 - val_accuracy: 0.9388
    Epoch 19/50
    329/329 [==============================] - 3s 9ms/step - loss: 0.1833 - accuracy: 0.9475 - val_loss: 0.1966 - val_accuracy: 0.9412
    Epoch 20/50
    329/329 [==============================] - 3s 10ms/step - loss: 0.1773 - accuracy: 0.9489 - val_loss: 0.2833 - val_accuracy: 0.9139
    Epoch 21/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.1729 - accuracy: 0.9503 - val_loss: 0.1866 - val_accuracy: 0.9457
    Epoch 22/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.1672 - accuracy: 0.9517 - val_loss: 0.1921 - val_accuracy: 0.9421
    Epoch 23/50
    329/329 [==============================] - 3s 9ms/step - loss: 0.1624 - accuracy: 0.9535 - val_loss: 0.1775 - val_accuracy: 0.9469
    Epoch 24/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.1575 - accuracy: 0.9551 - val_loss: 0.1791 - val_accuracy: 0.9472
    Epoch 25/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.1536 - accuracy: 0.9557 - val_loss: 0.1760 - val_accuracy: 0.9473
    Epoch 26/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.1491 - accuracy: 0.9572 - val_loss: 0.1689 - val_accuracy: 0.9491
    Epoch 27/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.1454 - accuracy: 0.9581 - val_loss: 0.1636 - val_accuracy: 0.9503
    Epoch 28/50
    329/329 [==============================] - 3s 10ms/step - loss: 0.1417 - accuracy: 0.9589 - val_loss: 0.1621 - val_accuracy: 0.9501
    Epoch 29/50
    329/329 [==============================] - 3s 9ms/step - loss: 0.1379 - accuracy: 0.9605 - val_loss: 0.1604 - val_accuracy: 0.9521
    Epoch 30/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.1343 - accuracy: 0.9609 - val_loss: 0.1574 - val_accuracy: 0.9519
    Epoch 31/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.1309 - accuracy: 0.9626 - val_loss: 0.1550 - val_accuracy: 0.9521
    Epoch 32/50
    329/329 [==============================] - 4s 11ms/step - loss: 0.1280 - accuracy: 0.9638 - val_loss: 0.1580 - val_accuracy: 0.9524
    Epoch 33/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.1245 - accuracy: 0.9640 - val_loss: 0.1524 - val_accuracy: 0.9539
    Epoch 34/50
    329/329 [==============================] - 3s 9ms/step - loss: 0.1216 - accuracy: 0.9653 - val_loss: 0.1640 - val_accuracy: 0.9508
    Epoch 35/50
    329/329 [==============================] - 4s 12ms/step - loss: 0.1188 - accuracy: 0.9659 - val_loss: 0.1465 - val_accuracy: 0.9551
    Epoch 36/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.1159 - accuracy: 0.9672 - val_loss: 0.1441 - val_accuracy: 0.9562
    Epoch 37/50
    329/329 [==============================] - 3s 8ms/step - loss: 0.1132 - accuracy: 0.9673 - val_loss: 0.1421 - val_accuracy: 0.9562
    Epoch 38/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.1105 - accuracy: 0.9683 - val_loss: 0.1401 - val_accuracy: 0.9570
    Epoch 39/50
    329/329 [==============================] - 5s 14ms/step - loss: 0.1081 - accuracy: 0.9685 - val_loss: 0.1417 - val_accuracy: 0.9568
    Epoch 40/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.1059 - accuracy: 0.9699 - val_loss: 0.1430 - val_accuracy: 0.9562
    Epoch 41/50
    329/329 [==============================] - 5s 14ms/step - loss: 0.1035 - accuracy: 0.9701 - val_loss: 0.1342 - val_accuracy: 0.9594
    Epoch 42/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.1009 - accuracy: 0.9712 - val_loss: 0.1718 - val_accuracy: 0.9486
    Epoch 43/50
    329/329 [==============================] - 2s 7ms/step - loss: 0.0986 - accuracy: 0.9719 - val_loss: 0.1322 - val_accuracy: 0.9592
    Epoch 44/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.0966 - accuracy: 0.9724 - val_loss: 0.1292 - val_accuracy: 0.9604
    Epoch 45/50
    329/329 [==============================] - 3s 9ms/step - loss: 0.0945 - accuracy: 0.9733 - val_loss: 0.1285 - val_accuracy: 0.9616
    Epoch 46/50
    329/329 [==============================] - 3s 8ms/step - loss: 0.0924 - accuracy: 0.9731 - val_loss: 0.1278 - val_accuracy: 0.9594
    Epoch 47/50
    329/329 [==============================] - 4s 11ms/step - loss: 0.0904 - accuracy: 0.9742 - val_loss: 0.1301 - val_accuracy: 0.9592
    Epoch 48/50
    329/329 [==============================] - 4s 13ms/step - loss: 0.0884 - accuracy: 0.9748 - val_loss: 0.1344 - val_accuracy: 0.9579
    Epoch 49/50
    329/329 [==============================] - 3s 10ms/step - loss: 0.0869 - accuracy: 0.9751 - val_loss: 0.1260 - val_accuracy: 0.9615
    Epoch 50/50
    329/329 [==============================] - 5s 15ms/step - loss: 0.0848 - accuracy: 0.9760 - val_loss: 0.1238 - val_accuracy: 0.9629
    


```python
# history
history.history.keys()
```




    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])




```python
# dict
history_dict = history.history

# Loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, loss, color='blue', label='Train Loss')
ax1.plot(epochs, val_loss, color='red', label='Valid Loss')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, acc, color='blue', label='Train Accuracy')
ax2.plot(epochs, val_acc, color='red', label='Valid Accuracy')
ax2.set_title('Train and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.show()
```

![eunji_21_0](https://user-images.githubusercontent.com/100760303/223306782-07b0f428-d481-47e8-8e0f-a216042440f0.png)

    


### 모델 학습 및 평가, 예측
#### 지표(Metrics)
- 모니터링할 지표
- `MSE`나 `Accuracy` 사용

#### fit()
- `x` : 학습 데이터
- `y` : 학습 데이터의 정답 레이블
- `epochs` : 학습 횟수
- `batch_size` : 단일 배치에 있는 학습 데이터의 크기
- `validation_data` : 검증을 위한 데이터

#### evaluate()
- 테스트 데이터를 이용한 평가

#### predict()
- 임의의 데이터를 사용해 예측



```python
model.evaluate(X_test, y_test)
```

    313/313 [==============================] - 2s 6ms/step - loss: 0.1210 - accuracy: 0.9639
    




    [0.121041439473629, 0.9639000296592712]




```python
pred_ys = model.predict(X_test)
print(pred_ys.shape)

np.set_printoptions(precision=7)
print(pred_ys[0])
```

    (10000, 10)
    [2.1240667e-06 4.3259917e-07 2.2279673e-04 3.7801344e-04 1.4935571e-08
     3.0192384e-06 4.5466300e-12 9.9938560e-01 1.4121780e-06 6.5638724e-06]
    


```python
arg_pred = np.argmax(pred_ys, axis=1)

plt.imshow(X_test[0])
plt.title('Predicted Label: {}'.format(arg_pred[0]))
```




    Text(0.5, 1.0, 'Predicted Label: 7')

![image](https://user-images.githubusercontent.com/100760303/223306865-0b1a8793-a8e3-449d-85dd-95a3eb460655.png)




#### 혼동 행렬(Confusion Matrix)


```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```


```python
plt.figure(figsize=(8, 8))

# (정답, 예측)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred_ys, axis=-1))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

![image](https://user-images.githubusercontent.com/100760303/223306899-4bc67f5b-bf2b-4c13-8534-abcc27c3db18.png)

    


#### 분류 보고서


```python
print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred_ys, axis=-1)))
```

                  precision    recall  f1-score   support
    
               0       0.97      0.98      0.98       980
               1       0.98      0.99      0.98      1135
               2       0.97      0.96      0.97      1032
               3       0.94      0.97      0.96      1010
               4       0.95      0.97      0.96       982
               5       0.96      0.94      0.95       892
               6       0.97      0.96      0.96       958
               7       0.95      0.97      0.96      1028
               8       0.97      0.94      0.96       974
               9       0.96      0.95      0.96      1009
    
        accuracy                           0.96     10000
       macro avg       0.96      0.96      0.96     10000
    weighted avg       0.96      0.96      0.96     10000
    
    

### 모델 저장과 복원
- `save()` : 저장
- `models.load_model()` : 복원
- Sequential API, 함수형 API에서는 모델의 저장 및 로드가 가능하지만 서브 클래싱 방법에서는 불가능
- JSON 형식
    - `model.to_json()` : 저장
    - `tf.keras.models.model_from_json(file_path)` : 복원
- YAML로 직렬화
    - `model.to_yaml()` : 저장
    - `tf.keras.models.model_from_yaml(file_path)`: 복원


```python
# 모델 저장
model.save('mnist_model.h5')
```


```python
# 모델 불러오기
load_model = models.load_model('mnist_model.h5')
```


```python
load_model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input (InputLayer)          [(None, 28, 28)]          0         
                                                                     
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense1 (Dense)              (None, 100)               78500     
                                                                     
     dense2 (Dense)              (None, 64)                6464      
                                                                     
     dense3 (Dense)              (None, 32)                2080      
                                                                     
     output (Dense)              (None, 10)                330       
                                                                     
    =================================================================
    Total params: 87,374
    Trainable params: 87,374
    Non-trainable params: 0
    _________________________________________________________________
