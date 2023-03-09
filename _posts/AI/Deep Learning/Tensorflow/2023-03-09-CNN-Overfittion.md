---
title: "[Tensorflow] CNN 과대적합 방지"
date: 2023-03-09

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## Library Import


```python
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets.fashion_mnist import load_data
```

## 모델 성능 높이기

### 레이어 추가


```python
from tensorflow.keras.layers import BatchNormalization
```


```python
# Model 정의
def build_model3():
    model = Sequential()

    input = Input(shape=(28, 28, 1))
    output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input)
    output = BatchNormalization()(output) # BatchNormalization
    output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid')(output)
    output = Dropout(0.5)(output)
    output = MaxPool2D(strides=(2, 2))(output) 

    output = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(output)
    output = BatchNormalization()(output) # BatchNormalization
    output = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(output)
    output = Dropout(0.5)(output) 
    output = MaxPool2D(strides=(2, 2))(output) 

    output = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(output)
    output = BatchNormalization()(output) # BatchNormalization
    output = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='valid')(output)
    output = Dropout(0.5)(output) 
    output = MaxPool2D(strides=(2, 2))(output) 

    output = Flatten()(output)
    output = Dense(units=512, activation='relu')(output)
    output = Dropout(0.5)(output) 
    output = Dense(units=256, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(units=10, activation='softmax')(output)

    model = Model(inputs=input, outputs=output)

    # Model Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
```


```python
model3 = build_model3()
model3.summary()
```

    Model: "model_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_7 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_27 (Conv2D)          (None, 28, 28, 64)        640       
                                                                     
     batch_normalization (BatchN  (None, 28, 28, 64)       256       
     ormalization)                                                   
                                                                     
     conv2d_28 (Conv2D)          (None, 26, 26, 64)        36928     
                                                                     
     dropout_8 (Dropout)         (None, 26, 26, 64)        0         
                                                                     
     max_pooling2d_10 (MaxPoolin  (None, 13, 13, 64)       0         
     g2D)                                                            
                                                                     
     conv2d_29 (Conv2D)          (None, 13, 13, 128)       73856     
                                                                     
     batch_normalization_1 (Batc  (None, 13, 13, 128)      512       
     hNormalization)                                                 
                                                                     
     conv2d_30 (Conv2D)          (None, 11, 11, 128)       147584    
                                                                     
     dropout_9 (Dropout)         (None, 11, 11, 128)       0         
                                                                     
     max_pooling2d_11 (MaxPoolin  (None, 5, 5, 128)        0         
     g2D)                                                            
                                                                     
     conv2d_31 (Conv2D)          (None, 5, 5, 256)         295168    
                                                                     
     batch_normalization_2 (Batc  (None, 5, 5, 256)        1024      
     hNormalization)                                                 
                                                                     
     conv2d_32 (Conv2D)          (None, 3, 3, 256)         590080    
                                                                     
     dropout_10 (Dropout)        (None, 3, 3, 256)         0         
                                                                     
     max_pooling2d_12 (MaxPoolin  (None, 1, 1, 256)        0         
     g2D)                                                            
                                                                     
     flatten_10 (Flatten)        (None, 256)               0         
                                                                     
     dense_24 (Dense)            (None, 512)               131584    
                                                                     
     dropout_11 (Dropout)        (None, 512)               0         
                                                                     
     dense_25 (Dense)            (None, 256)               131328    
                                                                     
     dropout_12 (Dropout)        (None, 256)               0         
                                                                     
     dense_26 (Dense)            (None, 10)                2570      
                                                                     
    =================================================================
    Total params: 1,411,530
    Trainable params: 1,410,634
    Non-trainable params: 896
    _________________________________________________________________
    


```python
# Model fit
history = model3.fit(X_train, y_train,
                       batch_size=128,
                       epochs=25,
                       validation_split=0.3)
```

    Epoch 1/25
    329/329 [==============================] - 17s 33ms/step - loss: 0.7737 - accuracy: 0.7189 - val_loss: 2.3274 - val_accuracy: 0.0983
    Epoch 2/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.4611 - accuracy: 0.8349 - val_loss: 1.0401 - val_accuracy: 0.7234
    Epoch 3/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.3759 - accuracy: 0.8701 - val_loss: 0.6177 - val_accuracy: 0.8683
    Epoch 4/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.3396 - accuracy: 0.8816 - val_loss: 0.6567 - val_accuracy: 0.8325
    Epoch 5/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.3120 - accuracy: 0.8919 - val_loss: 0.6944 - val_accuracy: 0.7979
    Epoch 6/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.2956 - accuracy: 0.8982 - val_loss: 0.5524 - val_accuracy: 0.8707
    Epoch 7/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.2788 - accuracy: 0.9028 - val_loss: 0.5265 - val_accuracy: 0.8628
    Epoch 8/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.2677 - accuracy: 0.9050 - val_loss: 0.5243 - val_accuracy: 0.8761
    Epoch 9/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.2578 - accuracy: 0.9101 - val_loss: 0.4724 - val_accuracy: 0.8939
    Epoch 10/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.2454 - accuracy: 0.9138 - val_loss: 0.3873 - val_accuracy: 0.9097
    Epoch 11/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.2376 - accuracy: 0.9177 - val_loss: 0.5193 - val_accuracy: 0.8693
    Epoch 12/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.2251 - accuracy: 0.9212 - val_loss: 0.3592 - val_accuracy: 0.9066
    Epoch 13/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.2278 - accuracy: 0.9202 - val_loss: 0.3483 - val_accuracy: 0.9015
    Epoch 14/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.2111 - accuracy: 0.9263 - val_loss: 0.4262 - val_accuracy: 0.8937
    Epoch 15/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.2075 - accuracy: 0.9280 - val_loss: 0.3437 - val_accuracy: 0.9078
    Epoch 16/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.1995 - accuracy: 0.9301 - val_loss: 0.2929 - val_accuracy: 0.9191
    Epoch 17/25
    329/329 [==============================] - 10s 30ms/step - loss: 0.1920 - accuracy: 0.9332 - val_loss: 0.3614 - val_accuracy: 0.9068
    Epoch 18/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.1869 - accuracy: 0.9340 - val_loss: 0.4148 - val_accuracy: 0.9100
    Epoch 19/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.1899 - accuracy: 0.9343 - val_loss: 0.3416 - val_accuracy: 0.9107
    Epoch 20/25
    329/329 [==============================] - 10s 32ms/step - loss: 0.1826 - accuracy: 0.9361 - val_loss: 0.3715 - val_accuracy: 0.9025
    Epoch 21/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.1818 - accuracy: 0.9379 - val_loss: 0.3454 - val_accuracy: 0.9166
    Epoch 22/25
    329/329 [==============================] - 12s 36ms/step - loss: 0.1670 - accuracy: 0.9402 - val_loss: 0.3038 - val_accuracy: 0.9175
    Epoch 23/25
    329/329 [==============================] - 11s 33ms/step - loss: 0.1698 - accuracy: 0.9415 - val_loss: 0.3227 - val_accuracy: 0.9036
    Epoch 24/25
    329/329 [==============================] - 10s 32ms/step - loss: 0.1625 - accuracy: 0.9432 - val_loss: 0.2868 - val_accuracy: 0.9173
    Epoch 25/25
    329/329 [==============================] - 10s 31ms/step - loss: 0.1542 - accuracy: 0.9465 - val_loss: 0.3098 - val_accuracy: 0.9137
    

- 과대적합은 되지 않았고, 층을 늘려도 좋은 성능을 낼 수 있음


```python
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r:', label='val_accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.show()
```




```python
model3.evaluate(X_test, y_test)
```

    313/313 [==============================] - 2s 5ms/step - loss: 0.3139 - accuracy: 0.9109
    




    [0.3139271140098572, 0.9108999967575073]



###이미지 보강(Image Augmentation)

- 주요 인자 참고: https://keras.io/ko/preprocessing/image/


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

- `shear_range` : 이미지 뒤틀기
- `width_shift_range` : 너비 Shift
- `height_shift_range` : 높이 Shift
- `horizontal_flip` : 좌우 뒤집기
- `vertical_flip` : 수직 뒤집기 


```python
image_generator = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.6,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

augment_size = 30000
random_mask = np.random.randint(X_train.shape[0], size=augment_size)
x_augmented = X_train[random_mask].copy()
y_augmented = y_train[random_mask].copy()

# 증강 전 이미지 shape
print(X_train.shape)
print(X_train[0].shape)
```

    (60000, 28, 28, 1)
    (28, 28, 1)
    


```python
x_augment = image_generator.flow(x_augmented,
                                 np.zeros(augment_size),
                                 batch_size=augment_size,
                                 shuffle=False
                                 ).next()[0]
```


```python
# 원래 이미지와 증강 이미지 합치기
X_train = np.concatenate([X_train, x_augmented])
y_train = np.concatenate([y_train, y_augmented])
```


```python
# 60000 + 30000 = 90000
print(X_train.shape, y_train.shape)
```

    (90000, 28, 28, 1) (90000,)
    


```python
# 증강된 데이터 확인
plt.figure(figsize=(10, 10))

for i in range(1, 101):
    plt.subplot(10, 10, i)
    plt.axis('off')
    plt.imshow(x_augment[i-1].reshape(28, 28), cmap='gray')
```

    



```python
model4 = build_model3()
model4.summary()
```

    Model: "model_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_8 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_33 (Conv2D)          (None, 28, 28, 64)        640       
                                                                     
     batch_normalization_3 (Batc  (None, 28, 28, 64)       256       
     hNormalization)                                                 
                                                                     
     conv2d_34 (Conv2D)          (None, 26, 26, 64)        36928     
                                                                     
     dropout_13 (Dropout)        (None, 26, 26, 64)        0         
                                                                     
     max_pooling2d_13 (MaxPoolin  (None, 13, 13, 64)       0         
     g2D)                                                            
                                                                     
     conv2d_35 (Conv2D)          (None, 13, 13, 128)       73856     
                                                                     
     batch_normalization_4 (Batc  (None, 13, 13, 128)      512       
     hNormalization)                                                 
                                                                     
     conv2d_36 (Conv2D)          (None, 11, 11, 128)       147584    
                                                                     
     dropout_14 (Dropout)        (None, 11, 11, 128)       0         
                                                                     
     max_pooling2d_14 (MaxPoolin  (None, 5, 5, 128)        0         
     g2D)                                                            
                                                                     
     conv2d_37 (Conv2D)          (None, 5, 5, 256)         295168    
                                                                     
     batch_normalization_5 (Batc  (None, 5, 5, 256)        1024      
     hNormalization)                                                 
                                                                     
     conv2d_38 (Conv2D)          (None, 3, 3, 256)         590080    
                                                                     
     dropout_15 (Dropout)        (None, 3, 3, 256)         0         
                                                                     
     max_pooling2d_15 (MaxPoolin  (None, 1, 1, 256)        0         
     g2D)                                                            
                                                                     
     flatten_11 (Flatten)        (None, 256)               0         
                                                                     
     dense_27 (Dense)            (None, 512)               131584    
                                                                     
     dropout_16 (Dropout)        (None, 512)               0         
                                                                     
     dense_28 (Dense)            (None, 256)               131328    
                                                                     
     dropout_17 (Dropout)        (None, 256)               0         
                                                                     
     dense_29 (Dense)            (None, 10)                2570      
                                                                     
    =================================================================
    Total params: 1,411,530
    Trainable params: 1,410,634
    Non-trainable params: 896
    _________________________________________________________________
    


```python
# Model fit
history = model4.fit(X_train, y_train,
                       batch_size=128,
                       epochs=40,
                       validation_split=0.3)
```

    Epoch 1/40
    493/493 [==============================] - 20s 34ms/step - loss: 0.6800 - accuracy: 0.7559 - val_loss: 2.2364 - val_accuracy: 0.1920
    Epoch 2/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.3883 - accuracy: 0.8652 - val_loss: 0.7200 - val_accuracy: 0.7874
    Epoch 3/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.3307 - accuracy: 0.8858 - val_loss: 0.6744 - val_accuracy: 0.8166
    Epoch 4/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.2993 - accuracy: 0.8944 - val_loss: 0.6065 - val_accuracy: 0.8475
    Epoch 5/40
    493/493 [==============================] - 16s 32ms/step - loss: 0.2813 - accuracy: 0.9034 - val_loss: 0.6051 - val_accuracy: 0.8205
    Epoch 6/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.2605 - accuracy: 0.9089 - val_loss: 0.4728 - val_accuracy: 0.8764
    Epoch 7/40
    493/493 [==============================] - 15s 31ms/step - loss: 0.2523 - accuracy: 0.9123 - val_loss: 0.4579 - val_accuracy: 0.8689
    Epoch 8/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.2430 - accuracy: 0.9154 - val_loss: 0.4819 - val_accuracy: 0.8330
    Epoch 9/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.2268 - accuracy: 0.9209 - val_loss: 0.3955 - val_accuracy: 0.8644
    Epoch 10/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.2181 - accuracy: 0.9226 - val_loss: 0.3610 - val_accuracy: 0.8876
    Epoch 11/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.2128 - accuracy: 0.9257 - val_loss: 0.4482 - val_accuracy: 0.8237
    Epoch 12/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.2013 - accuracy: 0.9299 - val_loss: 0.3289 - val_accuracy: 0.9152
    Epoch 13/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.2008 - accuracy: 0.9304 - val_loss: 0.3223 - val_accuracy: 0.9133
    Epoch 14/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1918 - accuracy: 0.9342 - val_loss: 0.3189 - val_accuracy: 0.9122
    Epoch 15/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1842 - accuracy: 0.9347 - val_loss: 0.2854 - val_accuracy: 0.8976
    Epoch 16/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1835 - accuracy: 0.9360 - val_loss: 0.2719 - val_accuracy: 0.9260
    Epoch 17/40
    493/493 [==============================] - 15s 31ms/step - loss: 0.1748 - accuracy: 0.9388 - val_loss: 0.2931 - val_accuracy: 0.9068
    Epoch 18/40
    493/493 [==============================] - 15s 31ms/step - loss: 0.1699 - accuracy: 0.9408 - val_loss: 0.2671 - val_accuracy: 0.9206
    Epoch 19/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1706 - accuracy: 0.9412 - val_loss: 0.2273 - val_accuracy: 0.9309
    Epoch 20/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1600 - accuracy: 0.9441 - val_loss: 0.2479 - val_accuracy: 0.9248
    Epoch 21/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1576 - accuracy: 0.9456 - val_loss: 0.2218 - val_accuracy: 0.9500
    Epoch 22/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1517 - accuracy: 0.9484 - val_loss: 0.2256 - val_accuracy: 0.9305
    Epoch 23/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1496 - accuracy: 0.9472 - val_loss: 0.2488 - val_accuracy: 0.9216
    Epoch 24/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1493 - accuracy: 0.9489 - val_loss: 0.2583 - val_accuracy: 0.9144
    Epoch 25/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1432 - accuracy: 0.9511 - val_loss: 0.1931 - val_accuracy: 0.9427
    Epoch 26/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1409 - accuracy: 0.9513 - val_loss: 0.2070 - val_accuracy: 0.9363
    Epoch 27/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1389 - accuracy: 0.9519 - val_loss: 0.2351 - val_accuracy: 0.9304
    Epoch 28/40
    493/493 [==============================] - 16s 32ms/step - loss: 0.1365 - accuracy: 0.9521 - val_loss: 0.2065 - val_accuracy: 0.9413
    Epoch 29/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1342 - accuracy: 0.9537 - val_loss: 0.2572 - val_accuracy: 0.9132
    Epoch 30/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1354 - accuracy: 0.9531 - val_loss: 0.1924 - val_accuracy: 0.9429
    Epoch 31/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1237 - accuracy: 0.9566 - val_loss: 0.2111 - val_accuracy: 0.9280
    Epoch 32/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1295 - accuracy: 0.9559 - val_loss: 0.2023 - val_accuracy: 0.9480
    Epoch 33/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1202 - accuracy: 0.9578 - val_loss: 0.1589 - val_accuracy: 0.9635
    Epoch 34/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1193 - accuracy: 0.9588 - val_loss: 0.1370 - val_accuracy: 0.9669
    Epoch 35/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1288 - accuracy: 0.9567 - val_loss: 0.2268 - val_accuracy: 0.9308
    Epoch 36/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1158 - accuracy: 0.9606 - val_loss: 0.1768 - val_accuracy: 0.9548
    Epoch 37/40
    493/493 [==============================] - 16s 33ms/step - loss: 0.1098 - accuracy: 0.9617 - val_loss: 0.1735 - val_accuracy: 0.9455
    Epoch 38/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1118 - accuracy: 0.9623 - val_loss: 0.1586 - val_accuracy: 0.9453
    Epoch 39/40
    493/493 [==============================] - 15s 31ms/step - loss: 0.1105 - accuracy: 0.9626 - val_loss: 0.1686 - val_accuracy: 0.9513
    Epoch 40/40
    493/493 [==============================] - 15s 30ms/step - loss: 0.1146 - accuracy: 0.9615 - val_loss: 0.2113 - val_accuracy: 0.9397
    


```python
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r:', label='val_loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r:', label='val_accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

plt.show()
```



```python
model4.evaluate(X_test, y_test)
```

    313/313 [==============================] - 1s 4ms/step - loss: 0.3133 - accuracy: 0.9003
    




    [0.3133423626422882, 0.9003000259399414]

