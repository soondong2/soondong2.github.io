---
title: "[Chatbot] Chapter6 챗봇 엔진에 필요한 딥러닝 모델2"
date: 2023-03-18

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Chatbot
---

## 챗봇 문답 데이터 감정 분류 모델 - CNN

### Library Call


```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Embedding, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from tensorflow.keras import preprocessing
```

### Data Load
- `Q` : 질문
- `A` : 답변
- `Label` : 감정
- 질문 데이터를 감정 클래스별로 분류하는 모델한 구현하기 때문에 답변 데이터는 사용하지 않음


```python
train_file = '/content/Chatbot_data.csv'
data = pd.read_csv(train_file, delimiter=',')

print(data.shape)
data.head()
```

    (11823, 3)
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12시 땡!</td>
      <td>하루가 또 가네요.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1지망 학교 떨어졌어</td>
      <td>위로해 드립니다.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3박4일 놀러가고 싶다</td>
      <td>여행은 언제나 좋죠.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3박4일 정도 놀러가고 싶다</td>
      <td>여행은 언제나 좋죠.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PPL 심하네</td>
      <td>눈살이 찌푸려지죠.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

<br> 

```python
features = data['Q'].tolist()
labels = data['label'].tolist()
```

### Data Preprocessing

- 시퀀스 번호로 만든 벡터의 한 가지 문제점
- 문장의 길이가 제각각
- 패딩으로 채움


```python
# 단어 인덱스 시퀀스 벡터

# ex) ['3박4일 놀러가고 싶다] -> ['3박4일', '놀러가고', '싶다]
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features] 

tokenizer = preprocessing.text.Tokenizer()

tokenizer.fit_on_texts(corpus) # 빈도수 기준으로 단어 집합 생성
sequences = tokenizer.texts_to_sequences(corpus) # 코퍼스에 대해서 각 단어를 정해진 인덱스로 변환
word_index = tokenizer.word_index # 각 단어에 인덱스가 어떻게 부여 되었는지 확인 (고유한 인덱스)

MAX_SEQ_LEN = 15 # 단어 시퀀스 벡터 크기

# Padding
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
```


```python
padded_seqs
```




    array([[ 4646,  4647,     0, ...,     0,     0,     0],
           [ 4648,   343,   448, ...,     0,     0,     0],
           [ 2580,   803,    11, ...,     0,     0,     0],
           ...,
           [13395,  2517,    89, ...,     0,     0,     0],
           [  147,    46,    91, ...,     0,     0,     0],
           [  555, 13398,     0, ...,     0,     0,     0]], dtype=int32)



### Data Split
- 학습:검증:테스트 = 7:2:1


```python
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

print(train_size, val_size, test_size)
```

    8276 2364 1182
    


```python
train_data = ds.take(train_size).batch(20)
val_data = ds.take(val_size).batch(20)
test_data = ds.take(test_size).batch(20)
```

### Hyperparameter


```python
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1 # 전체 단어 수
```

### Modeling


```python
# CNN Model
input = Input(shape=(MAX_SEQ_LEN))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu')(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu')(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu')(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3, 4, 5-gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(units=128, activation='relu')(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(units=3, name='logits')(dropout_hidden)
output = Dense(units=3, activation='softmax')(logits)
```


```python
# Model 생성
model = Model(inputs=input, outputs=output)
model.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_4 (InputLayer)           [(None, 15)]         0           []                               
                                                                                                      
     embedding_1 (Embedding)        (None, 15, 128)      1715072     ['input_4[0][0]']                
                                                                                                      
     dropout_2 (Dropout)            (None, 15, 128)      0           ['embedding_1[0][0]']            
                                                                                                      
     conv1d_3 (Conv1D)              (None, 13, 128)      49280       ['dropout_2[0][0]']              
                                                                                                      
     conv1d_4 (Conv1D)              (None, 12, 128)      65664       ['dropout_2[0][0]']              
                                                                                                      
     conv1d_5 (Conv1D)              (None, 11, 128)      82048       ['dropout_2[0][0]']              
                                                                                                      
     global_max_pooling1d_3 (Global  (None, 128)         0           ['conv1d_3[0][0]']               
     MaxPooling1D)                                                                                    
                                                                                                      
     global_max_pooling1d_4 (Global  (None, 128)         0           ['conv1d_4[0][0]']               
     MaxPooling1D)                                                                                    
                                                                                                      
     global_max_pooling1d_5 (Global  (None, 128)         0           ['conv1d_5[0][0]']               
     MaxPooling1D)                                                                                    
                                                                                                      
     concatenate_1 (Concatenate)    (None, 384)          0           ['global_max_pooling1d_3[0][0]', 
                                                                      'global_max_pooling1d_4[0][0]', 
                                                                      'global_max_pooling1d_5[0][0]'] 
                                                                                                      
     dense_8 (Dense)                (None, 128)          49280       ['concatenate_1[0][0]']          
                                                                                                      
     dropout_3 (Dropout)            (None, 128)          0           ['dense_8[0][0]']                
                                                                                                      
     logits (Dense)                 (None, 3)            387         ['dropout_3[0][0]']              
                                                                                                      
     dense_9 (Dense)                (None, 3)            12          ['logits[0][0]']                 
                                                                                                      
    ==================================================================================================
    Total params: 1,961,743
    Trainable params: 1,961,743
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```


```python
# Model 학습
history = model.fit(train_data, validation_data=val_data, epochs=EPOCH, verbose=1)
```

    Epoch 1/5
    414/414 [==============================] - 4s 9ms/step - loss: 0.5011 - accuracy: 0.8134 - val_loss: 0.2919 - val_accuracy: 0.9074
    Epoch 2/5
    414/414 [==============================] - 3s 8ms/step - loss: 0.3081 - accuracy: 0.9003 - val_loss: 0.1549 - val_accuracy: 0.9543
    Epoch 3/5
    414/414 [==============================] - 3s 8ms/step - loss: 0.1855 - accuracy: 0.9379 - val_loss: 0.1013 - val_accuracy: 0.9674
    Epoch 4/5
    414/414 [==============================] - 3s 6ms/step - loss: 0.1366 - accuracy: 0.9589 - val_loss: 0.0685 - val_accuracy: 0.9755
    Epoch 5/5
    414/414 [==============================] - 4s 10ms/step - loss: 0.0961 - accuracy: 0.9706 - val_loss: 0.0520 - val_accuracy: 0.9814
    


```python
# Model evaluate
model.evaluate(test_data, verbose=1)
```

    60/60 [==============================] - 0s 4ms/step - loss: 0.0530 - accuracy: 0.9831

    [0.053025633096694946, 0.9830795526504517]




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
![image](https://user-images.githubusercontent.com/100760303/226100751-4439da53-3fcc-4ec7-8c8e-ec31e25103a9.png)

### Predict


```python
# 10212번째 문장 데이터 확인
print('단어 시퀀스 : ', corpus[10212])
print('단어 인덱스 시퀀스 : ', padded_seqs[10212])
print('문장 분류 : ', labels[10212])
```

    단어 시퀀스 :  ['썸', '타는', '여자가', '남사친', '만나러', '간다는데', '뭐라', '해']
    단어 인덱스 시퀀스 :  [   13    61   127  4320  1333 12162   856    31     0     0     0     0
         0     0     0]
    문장 분류 :  2
    


```python
# 10212번째 문장 예측
pred = model.predict(padded_seqs[[10212]])
pred_class = tf.math.argmax(pred, axis=1)

print('감정 예측 점수 : ', pred)
print('감정 예측 Class : ', pred_class.numpy())
```

    1/1 [==============================] - 0s 72ms/step
    감정 예측 점수 :  [[3.8029467e-07 1.0989698e-08 9.9999964e-01]]
    감정 예측 Class :  [2]
    
