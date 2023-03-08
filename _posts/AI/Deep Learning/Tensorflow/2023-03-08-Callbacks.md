---
title: "[Tensorflow] Callbacks 기능"
date: 2023-03-08

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## 콜백(Callbacks)
- `fit()` 함수의 callbacks 매개변수를 사용하여 케라스가 훈련의 시작이나 끝에 호출할 객체 리스트를 지정할 수 있음
- 여러 개 사용 가능


```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
```

### ModelCheckpoint
- `tf.keras.callbacks.ModelCheckpoint` : 정기적으로 모델의 체크포인트를 저장하고 문제가 발생할 때 복구하는데 사용


```python
# 모델 체크 포인트의 저장할 Path 지정
check_point_cb = ModelCheckpoint('keras_mnist_model.h5')
history = model.fit(X_train, y_train, epochs=10, callbacks=[check_point_cb])
```

    Epoch 1/10
    1313/1313 [==============================] - 11s 8ms/step - loss: 0.0835 - accuracy: 0.9752
    Epoch 2/10
    1313/1313 [==============================] - 11s 8ms/step - loss: 0.0779 - accuracy: 0.9765
    Epoch 3/10
    1313/1313 [==============================] - 10s 7ms/step - loss: 0.0713 - accuracy: 0.9792
    Epoch 4/10
    1313/1313 [==============================] - 10s 8ms/step - loss: 0.0664 - accuracy: 0.9802
    Epoch 5/10
    1313/1313 [==============================] - 9s 7ms/step - loss: 0.0611 - accuracy: 0.9817
    Epoch 6/10
    1313/1313 [==============================] - 8s 6ms/step - loss: 0.0571 - accuracy: 0.9833
    Epoch 7/10
    1313/1313 [==============================] - 9s 7ms/step - loss: 0.0528 - accuracy: 0.9844
    Epoch 8/10
    1313/1313 [==============================] - 10s 8ms/step - loss: 0.0492 - accuracy: 0.9857
    Epoch 9/10
    1313/1313 [==============================] - 9s 7ms/step - loss: 0.0452 - accuracy: 0.9867
    Epoch 10/10
    1313/1313 [==============================] - 8s 6ms/step - loss: 0.0425 - accuracy: 0.9876
    

- 최상의 모델만은 저장 : `save_best_only=True`
- validation data 값으로 최상의 모델이 저장되는 것이므로 validation_data 지정해주어야 함


```python
# 모델 체크 포인트의 저장할 Path 지정
# save_best_only=True를 통해 학습 중 어떤 게 best인지 check
check_point_cb = ModelCheckpoint('keras_mnist_model.h5', save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_val, y_val),
                    callbacks=[check_point_cb])
```

    Epoch 1/10
    1313/1313 [==============================] - 15s 11ms/step - loss: 0.0177 - accuracy: 0.9960 - val_loss: 0.0977 - val_accuracy: 0.9731
    Epoch 2/10
    1313/1313 [==============================] - 14s 10ms/step - loss: 0.0168 - accuracy: 0.9963 - val_loss: 0.1003 - val_accuracy: 0.9734
    Epoch 3/10
    1313/1313 [==============================] - 15s 11ms/step - loss: 0.0152 - accuracy: 0.9970 - val_loss: 0.0994 - val_accuracy: 0.9727
    Epoch 4/10
    1313/1313 [==============================] - 14s 11ms/step - loss: 0.0141 - accuracy: 0.9974 - val_loss: 0.1039 - val_accuracy: 0.9733
    Epoch 5/10
    1313/1313 [==============================] - 12s 9ms/step - loss: 0.0126 - accuracy: 0.9979 - val_loss: 0.1002 - val_accuracy: 0.9744
    Epoch 6/10
    1313/1313 [==============================] - 12s 9ms/step - loss: 0.0118 - accuracy: 0.9981 - val_loss: 0.1053 - val_accuracy: 0.9725
    Epoch 7/10
    1313/1313 [==============================] - 18s 13ms/step - loss: 0.0110 - accuracy: 0.9984 - val_loss: 0.1003 - val_accuracy: 0.9743
    Epoch 8/10
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0102 - accuracy: 0.9986 - val_loss: 0.0997 - val_accuracy: 0.9745
    Epoch 9/10
    1313/1313 [==============================] - 11s 8ms/step - loss: 0.0091 - accuracy: 0.9990 - val_loss: 0.1078 - val_accuracy: 0.9726
    Epoch 10/10
    1313/1313 [==============================] - 12s 9ms/step - loss: 0.0090 - accuracy: 0.9988 - val_loss: 0.1007 - val_accuracy: 0.9747
    

### EalyStopping
- `tf.keras.callbacks.EarlyStopping` : 검증 성능이 한동안 개선되지 않을 경우 학습을 중단할 때 사용
- 3번을 봐서 더이상 성능이 향상되지 않으면 멈춤


```python
check_point_cb = ModelCheckpoint('keras_mnist_model.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(patience=3, monitor='val_loss',
                                  restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_val, y_val),
                    callbacks=[check_point_cb, early_stopping_cb])
```

    Epoch 1/10
    1313/1313 [==============================] - 16s 12ms/step - loss: 0.0079 - accuracy: 0.9993 - val_loss: 0.1017 - val_accuracy: 0.9743
    Epoch 2/10
    1313/1313 [==============================] - 14s 11ms/step - loss: 0.0076 - accuracy: 0.9993 - val_loss: 0.1054 - val_accuracy: 0.9748
    Epoch 3/10
    1313/1313 [==============================] - 15s 12ms/step - loss: 0.0069 - accuracy: 0.9994 - val_loss: 0.1039 - val_accuracy: 0.9748
    Epoch 4/10
    1313/1313 [==============================] - 14s 10ms/step - loss: 0.0065 - accuracy: 0.9995 - val_loss: 0.1069 - val_accuracy: 0.9739
    

### LearninRateScheduler
- `tf.keras.callbacks.LearningRateScheduler` : 최적화를 하는 동안 학습률을 동적으로 변경할 때 사용


```python
def scheduler(epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    else:
        # learning rate 줄이기
        return learning_rate * tf.math.exp(-0.1)
```


```python
# 기존의 learning rate
round(model.optimizer.lr.numpy(), 5)
```




    0.01



- `varbose=0`으로 지정해서 로그가 뜨지 않음


```python
lr_scheduler_cb = LearningRateScheduler(scheduler)

history = model.fit(X_train, y_train, epochs=15,
                    callbacks=[lr_scheduler_cb], verbose=0)

# 현재 모델의 optimizer의 learning rate 변화 확인
round(model.optimizer.lr.numpy(), 5)
```




    0.00607



### Tensorboard
- `tf.keras.callbacks.TensorBoard` : 모델의 경과를 모니터링할 때 사용
- 텐서보드를 사용하기 위해 logs 폴더를 만들고, 학습이 진행되는 동안 로그 파일을 생성
- `histogram_freq` : 히스토그램을 몇 단위로 볼 건지
- `write_graph=True` : 그래프 보여주기
- `write_image=True` : 이미지로 보여주기


```python
log_dir = './logs'
tensor_board_cb = [TensorBoard(log_dir=log_dir, histogram_freq=1,
                               write_graph=True, write_images=True)]

model.fit(X_train, y_train,
          epochs=30, batch_size=32,
          validation_data=(X_val, y_val),
          callbacks=tensor_board_cb)
```

    Epoch 1/30
    1313/1313 [==============================] - 18s 13ms/step - loss: 0.0029 - accuracy: 0.9999 - val_loss: 0.1104 - val_accuracy: 0.9743
    Epoch 2/30
    1313/1313 [==============================] - 17s 13ms/step - loss: 0.0028 - accuracy: 0.9999 - val_loss: 0.1116 - val_accuracy: 0.9753
    Epoch 3/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0027 - accuracy: 0.9999 - val_loss: 0.1114 - val_accuracy: 0.9754
    Epoch 4/30
    1313/1313 [==============================] - 14s 10ms/step - loss: 0.0027 - accuracy: 0.9999 - val_loss: 0.1125 - val_accuracy: 0.9752
    Epoch 5/30
    1313/1313 [==============================] - 15s 11ms/step - loss: 0.0026 - accuracy: 0.9999 - val_loss: 0.1116 - val_accuracy: 0.9750
    Epoch 6/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0026 - accuracy: 0.9999 - val_loss: 0.1133 - val_accuracy: 0.9749
    Epoch 7/30
    1313/1313 [==============================] - 12s 9ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.1120 - val_accuracy: 0.9748
    Epoch 8/30
    1313/1313 [==============================] - 10s 7ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.1128 - val_accuracy: 0.9753
    Epoch 9/30
    1313/1313 [==============================] - 10s 7ms/step - loss: 0.0024 - accuracy: 0.9999 - val_loss: 0.1130 - val_accuracy: 0.9753
    Epoch 10/30
    1313/1313 [==============================] - 10s 8ms/step - loss: 0.0023 - accuracy: 1.0000 - val_loss: 0.1136 - val_accuracy: 0.9750
    Epoch 11/30
    1313/1313 [==============================] - 10s 8ms/step - loss: 0.0023 - accuracy: 0.9999 - val_loss: 0.1137 - val_accuracy: 0.9754
    Epoch 12/30
    1313/1313 [==============================] - 14s 10ms/step - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.1136 - val_accuracy: 0.9749
    Epoch 13/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.1149 - val_accuracy: 0.9750
    Epoch 14/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0021 - accuracy: 0.9999 - val_loss: 0.1144 - val_accuracy: 0.9747
    Epoch 15/30
    1313/1313 [==============================] - 11s 9ms/step - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.1152 - val_accuracy: 0.9753
    Epoch 16/30
    1313/1313 [==============================] - 10s 7ms/step - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.1150 - val_accuracy: 0.9753
    Epoch 17/30
    1313/1313 [==============================] - 11s 8ms/step - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.1152 - val_accuracy: 0.9751
    Epoch 18/30
    1313/1313 [==============================] - 14s 10ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.1155 - val_accuracy: 0.9749
    Epoch 19/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.1164 - val_accuracy: 0.9751
    Epoch 20/30
    1313/1313 [==============================] - 12s 9ms/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.1161 - val_accuracy: 0.9751
    Epoch 21/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.1165 - val_accuracy: 0.9749
    Epoch 22/30
    1313/1313 [==============================] - 14s 11ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.1164 - val_accuracy: 0.9752
    Epoch 23/30
    1313/1313 [==============================] - 14s 10ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.1173 - val_accuracy: 0.9751
    Epoch 24/30
    1313/1313 [==============================] - 17s 13ms/step - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.1175 - val_accuracy: 0.9748
    Epoch 25/30
    1313/1313 [==============================] - 19s 14ms/step - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.1176 - val_accuracy: 0.9752
    Epoch 26/30
    1313/1313 [==============================] - 17s 13ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.1176 - val_accuracy: 0.9751
    Epoch 27/30
    1313/1313 [==============================] - 14s 11ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.1184 - val_accuracy: 0.9752
    Epoch 28/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.1188 - val_accuracy: 0.9752
    Epoch 29/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.1183 - val_accuracy: 0.9748
    Epoch 30/30
    1313/1313 [==============================] - 13s 10ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.1193 - val_accuracy: 0.9751
   




```python
%tensorboard --logdir {log_dir} por 8000
```
![image](https://user-images.githubusercontent.com/100760303/223606802-51b9a612-552f-40b5-8531-e5932d8978d6.png)
