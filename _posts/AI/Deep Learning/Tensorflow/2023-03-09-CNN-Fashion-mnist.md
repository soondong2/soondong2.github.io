---
title: "[Tensorflow] CNN Fashion MNIST 예제"
date: 2023-03-09

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## Fashion MNIST

<img src="https://www.tensorflow.org/tutorials/keras/classification_files/output_oZTImqg_CaW1_0.png?hl=ko" width="500">

## Library Import


```python
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets.fashion_mnist import load_data
```

## Data Load & Image Preprocessing


```python
# Data Load
(X_train, y_train), (X_test, y_test) = load_data()

print(X_train.shape)
print(X_test.shape)

# 축 추가
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape)
print(X_test.shape)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    29515/29515 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26421880/26421880 [==============================] - 3s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    5148/5148 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4422102/4422102 [==============================] - 0s 0us/step
    (60000, 28, 28)
    (10000, 28, 28)
    (60000, 28, 28, 1)
    (10000, 28, 28, 1)
    


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## Model


```python
# Model 정의
def build_model():
    model = Sequential()

    input = Input(shape=(28, 28, 1))
    output = Conv2D(filters=32, kernel_size=(3, 3))(input)
    output = Conv2D(filters=64, kernel_size=(3, 3))(output)
    output = Conv2D(filters=64, kernel_size=(3, 3))(output)
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = Model(inputs=input, outputs=output)

    # Model Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
```


```python
model_1 = build_model()
model_1.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_4 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_18 (Conv2D)          (None, 26, 26, 32)        320       
                                                                     
     conv2d_19 (Conv2D)          (None, 24, 24, 64)        18496     
                                                                     
     conv2d_20 (Conv2D)          (None, 22, 22, 64)        36928     
                                                                     
     flatten_7 (Flatten)         (None, 30976)             0         
                                                                     
     dense_15 (Dense)            (None, 128)               3965056   
                                                                     
     dense_16 (Dense)            (None, 64)                8256      
                                                                     
     dense_17 (Dense)            (None, 10)                650       
                                                                     
    =================================================================
    Total params: 4,029,706
    Trainable params: 4,029,706
    Non-trainable params: 0
    _________________________________________________________________
    

accuracy는 오르고 있지만 validation accuracy가 오르지 않는 Overfitting 현상 발생


```python
# Model fit
history1 = model_1.fit(X_train, y_train,
                       batch_size=128,
                       epochs=25,
                       validation_split=0.3)
```

    Epoch 1/25
    329/329 [==============================] - 6s 17ms/step - loss: 0.3269 - accuracy: 0.8804 - val_loss: 0.3522 - val_accuracy: 0.8736
    Epoch 2/25
    329/329 [==============================] - 4s 13ms/step - loss: 0.2733 - accuracy: 0.8995 - val_loss: 0.3382 - val_accuracy: 0.8782
    Epoch 3/25
    329/329 [==============================] - 5s 15ms/step - loss: 0.2323 - accuracy: 0.9140 - val_loss: 0.3505 - val_accuracy: 0.8827
    Epoch 4/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.1983 - accuracy: 0.9265 - val_loss: 0.3772 - val_accuracy: 0.8753
    Epoch 5/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.1712 - accuracy: 0.9358 - val_loss: 0.4028 - val_accuracy: 0.8741
    Epoch 6/25
    329/329 [==============================] - 5s 16ms/step - loss: 0.1531 - accuracy: 0.9436 - val_loss: 0.4374 - val_accuracy: 0.8751
    Epoch 7/25
    329/329 [==============================] - 4s 13ms/step - loss: 0.1345 - accuracy: 0.9507 - val_loss: 0.4659 - val_accuracy: 0.8768
    Epoch 8/25
    329/329 [==============================] - 5s 15ms/step - loss: 0.1212 - accuracy: 0.9552 - val_loss: 0.5363 - val_accuracy: 0.8724
    Epoch 9/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.1024 - accuracy: 0.9616 - val_loss: 0.5711 - val_accuracy: 0.8724
    Epoch 10/25
    329/329 [==============================] - 4s 14ms/step - loss: 0.0930 - accuracy: 0.9657 - val_loss: 0.6218 - val_accuracy: 0.8639
    Epoch 11/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.0864 - accuracy: 0.9681 - val_loss: 0.6617 - val_accuracy: 0.8666
    Epoch 12/25
    329/329 [==============================] - 5s 16ms/step - loss: 0.0845 - accuracy: 0.9705 - val_loss: 0.7101 - val_accuracy: 0.8617
    Epoch 13/25
    329/329 [==============================] - 5s 15ms/step - loss: 0.0785 - accuracy: 0.9713 - val_loss: 0.7027 - val_accuracy: 0.8673
    Epoch 14/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.0685 - accuracy: 0.9761 - val_loss: 0.7472 - val_accuracy: 0.8660
    Epoch 15/25
    329/329 [==============================] - 4s 13ms/step - loss: 0.0613 - accuracy: 0.9784 - val_loss: 0.8054 - val_accuracy: 0.8684
    Epoch 16/25
    329/329 [==============================] - 5s 15ms/step - loss: 0.0602 - accuracy: 0.9785 - val_loss: 0.7927 - val_accuracy: 0.8677
    Epoch 17/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.0576 - accuracy: 0.9794 - val_loss: 0.9067 - val_accuracy: 0.8662
    Epoch 18/25
    329/329 [==============================] - 4s 13ms/step - loss: 0.0609 - accuracy: 0.9787 - val_loss: 0.8520 - val_accuracy: 0.8731
    Epoch 19/25
    329/329 [==============================] - 5s 16ms/step - loss: 0.0547 - accuracy: 0.9819 - val_loss: 0.8942 - val_accuracy: 0.8672
    Epoch 20/25
    329/329 [==============================] - 4s 13ms/step - loss: 0.0522 - accuracy: 0.9810 - val_loss: 0.9241 - val_accuracy: 0.8693
    Epoch 21/25
    329/329 [==============================] - 4s 13ms/step - loss: 0.0428 - accuracy: 0.9850 - val_loss: 0.9956 - val_accuracy: 0.8685
    Epoch 22/25
    329/329 [==============================] - 5s 16ms/step - loss: 0.0378 - accuracy: 0.9868 - val_loss: 1.0312 - val_accuracy: 0.8626
    Epoch 23/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.0475 - accuracy: 0.9833 - val_loss: 1.0124 - val_accuracy: 0.8623
    Epoch 24/25
    329/329 [==============================] - 5s 16ms/step - loss: 0.0368 - accuracy: 0.9873 - val_loss: 1.1394 - val_accuracy: 0.8602
    Epoch 25/25
    329/329 [==============================] - 5s 14ms/step - loss: 0.0475 - accuracy: 0.9849 - val_loss: 1.0265 - val_accuracy: 0.8673
    


```python
history1.history.keys()
```




    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])




```python
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], 'b--', label='loss')
plt.plot(history1.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history1.history['accuracy'], 'b--', label='accuracy')
plt.plot(history1.history['val_accuracy'], 'r:', label='val_accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.show()
```
![image](https://user-images.githubusercontent.com/100760303/223929742-d5f400cf-cff9-4803-bae3-c1386c0d1fb5.png)
    



```python
# 모델 평가
model_1.evaluate(X_test, y_test)
```

    313/313 [==============================] - 1s 4ms/step - loss: 1.1313 - accuracy: 0.8558
    




    [1.1313254833221436, 0.8557999730110168]


과적합 문제를 해결하기 위한 방법을 다음 포스팅에서 다룰 예정이다.
