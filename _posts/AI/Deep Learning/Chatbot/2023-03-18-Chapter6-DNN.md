---
title: "[Chatbot] Chapter6 챗봇 엔진에 필요한 딥러닝 모델1"
date: 2023-03-18

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Chatbot
---

## 딥러닝 분류 모델

### Library Call


```python
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
```

### Data Load


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 정규화
X_train, X_test = X_train / 255.0, X_test / 255.0
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 2s 0us/step
    


```python
# tf.data를 사용해 데이터 셋을 섞고 배치 만들기
ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000)
train_size = int(len(X_train) * 0.7) # 학습:검증 7:3
train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).batch(20)
```

### Modeling
- 교재에서는 Sequential 모델을 생성하지만, 일부러 Functional API를 사용해 모델을 생성하였다. (연습을 위해서)


```python
# Model 구성
input = Input(shape=(28, 28))
x = Flatten(input_shape=(28, 28))(input)
x = Dense(units=20, activation='relu')(x)
x = Dense(units=20, activation='relu')(x)
output = Dense(units=10, activation='softmax')(x)

model = Model(inputs=input, outputs=output)
model.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 28, 28)]          0         
                                                                     
     flatten_1 (Flatten)         (None, 784)               0         
                                                                     
     dense_3 (Dense)             (None, 20)                15700     
                                                                     
     dense_4 (Dense)             (None, 20)                420       
                                                                     
     dense_5 (Dense)             (None, 10)                210       
                                                                     
    =================================================================
    Total params: 16,330
    Trainable params: 16,330
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Model 생성
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```


```python
# Model 학습
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
```

    Epoch 1/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.8520 - accuracy: 0.7533 - val_loss: 0.3964 - val_accuracy: 0.8810
    Epoch 2/10
    2100/2100 [==============================] - 7s 4ms/step - loss: 0.3569 - accuracy: 0.8975 - val_loss: 0.3093 - val_accuracy: 0.9091
    Epoch 3/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.2893 - accuracy: 0.9170 - val_loss: 0.2750 - val_accuracy: 0.9201
    Epoch 4/10
    2100/2100 [==============================] - 9s 4ms/step - loss: 0.2537 - accuracy: 0.9277 - val_loss: 0.2243 - val_accuracy: 0.9353
    Epoch 5/10
    2100/2100 [==============================] - 7s 4ms/step - loss: 0.2268 - accuracy: 0.9343 - val_loss: 0.2134 - val_accuracy: 0.9378
    Epoch 6/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.2081 - accuracy: 0.9411 - val_loss: 0.1958 - val_accuracy: 0.9446
    Epoch 7/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.1925 - accuracy: 0.9452 - val_loss: 0.1878 - val_accuracy: 0.9466
    Epoch 8/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.1821 - accuracy: 0.9476 - val_loss: 0.1783 - val_accuracy: 0.9489
    Epoch 9/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.1734 - accuracy: 0.9503 - val_loss: 0.1717 - val_accuracy: 0.9522
    Epoch 10/10
    2100/2100 [==============================] - 8s 4ms/step - loss: 0.1638 - accuracy: 0.9525 - val_loss: 0.1667 - val_accuracy: 0.9522
    


```python
# Model 평가
model.evaluate(X_test, y_test)
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.1792 - accuracy: 0.9509
    




    [0.17917940020561218, 0.9509000182151794]




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
![image](https://user-images.githubusercontent.com/100760303/226100489-cb57536b-07b1-4602-8887-417695ce6a83.png)
