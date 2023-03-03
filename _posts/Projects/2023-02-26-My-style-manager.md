---
title: "Fashion Recommendation System"
date: 2023-02-26

categories:
  - Projects
tags:
    - DL
    - AutoEncoder
    - Recommendation
---

# Fashion Recommendation System
![image](https://user-images.githubusercontent.com/100760303/222645589-81d03215-5bcb-427f-b998-70d8fa406f80.png)
![image](https://user-images.githubusercontent.com/100760303/222645691-66dbd214-6667-4696-99ab-859155ddbff3.png)
![image](https://user-images.githubusercontent.com/100760303/222645800-dab417b3-d930-408c-b647-091bbf10855a.png)


**[Data Information]**<br>
Data Source: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=78   
Raw Data Type : 720x1280x24b

**[Image]**<br>
Item-Image : 16585장   
Model-Image : 18040장

**[Version]**<br>
Augmented Convolutional AE, 128x128x3

## 0. Setting

### Goole Drive Connecting


```python
pwd
```




    '/Users/haesik/AISCHOOL/Final_Project/img'




```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
%cd '/content/drive/MyDrive/Code Lion/Final'
```

    /content/drive/MyDrive/Code Lion/Final
    


```python
!ls
```

     data  'Fashion Recommendation System0810.ipynb'   figure  'low version'
    

### Library Call


```python
pip install tensorflow_addons
```


```python
pip install opencv-python
```
   


```python
pip install pydot
```


```python
pip install graphviz
```


```python
# 상용 라이브러리
from glob import glob
import os
import cv2
import pandas as pd
import numpy as np
import datetime as dt
import time

# 시각화 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go

# 한글 폰트 패치
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False   

# 시각화 포맷 설정
plt.style.use("ggplot")
sns.set(font_scale=2)
sns.set_style("whitegrid")
sns.set_context("talk")

# 경고문 처리
import warnings
warnings.filterwarnings('ignore')

# sckit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances

# Tensorflow 라이브러리
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras import layers, models
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
```

### User Function Definition


```python
# -------------Image Load & Preprocessing ------------- #
# Global Constant Definition
imgR = 128
imgC = 128
channel = 3
crop_y = (250,1000)
crop_x = (40,680)
root_dir = 'D:/Fasion_Images/Train/train_itemimages/Item-Image/'
model_dir = 'D:/Fasion_Images/Train/train_modelimages/Model-Image_deid/'

# Single Image Load
def img_read(file):
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

# Gamma Correction
def adjust_gamma(img, gamma=1.0): # 감마 보정 함수
    invGamma = 1.0 / gamma
    out = img.copy().astype(np.float)
    out = ((out / 255) ** invGamma) * 255
    return out.astype(np.uint8)

# Image Crop & Resize
def img_crop(img):
    img = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    img = cv2.resize(img, (imgR,imgC), cv2.INTER_LINEAR)
    return img

# Load All img from folder
def load_img_folder():
    # 의상의 전방부 사진만 가져오기
    wfiles = sorted(glob(f'{root_dir}/*_F.jpg'))
    img_list = []
    label_list = []
    for file in wfiles:
        img = img_read(file)
        img = img_crop(img)
        img = adjust_gamma(img, 0.8)
        img_list.append(img)
        label_list.append(file.split('/')[-1])
    return np.array(img_list), label_list

# Top10 유사 이미지 시각화
def top10_visualize(img_set,top10_idx):
  fig = plt.figure()
  fig, ax = plt.subplots(2, 5, figsize=(5*3,2*3))
  plt.suptitle('Top10 Similar Images',size=20)
  k=0
  for i in range(2):
    for j in range(5):
      axis = ax[i,j]
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
      axis.imshow(img_set[top10_idx[i+j]])
      plt.axis('off')
    k += 5
  plt.show()

# -------------Image EDA & Visualization ------------- #
# plot_images
def plot_images(nRow, nCol, img_set):
  fig = plt.figure()
  fig, ax = plt.subplots(nRow, nCol, figsize=(nCol*4,nRow*4))
  k=0
  for i in range(nRow):
    for j in range(nCol):
      if nRow <= 1 : axis = ax[j]
      else:          axis = ax[i,j]
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
      axis.imshow(img_set[k+j])
      plt.axis('off')
    k += nCol
  plt.show()

# ------------- Model Function ------------- #
# Reconstruction Error Function Definition
def Reconstruction_Error(X_test,X_pred):
  error_list = []
  for i in range(len(X_test)):
    ele = np.mean(np.power(X_test[i] - X_pred[i], 2),axis=1).mean()
    error_list.append(ele)
  return error_list

# Average Pooling Fuction Definition
def AVGpooling(raw_feature):
  result = []
  for i in range(raw_feature.shape[0]):
    row= []
    for j in range(raw_feature.shape[-1]):
      row.append(raw_feature[i,:,:,j].mean())
    result.append(row)
  return np.array(result)

# ------------- Recommendation System ------------- #
def Fashion_coordination(top10_result, fashion_df):
  result_df = []
  for item in top10_result:
    ele_df = pd.DataFrame(columns=fashion_df.columns)
    for col in fashion_df.columns:
      ele = fashion_df[fashion_df[col] == item]
      ele_df = pd.concat([ele_df,ele])
    result_df.append(ele_df)
  return result_df
```

## 1. Data Load


```python
# Image DataSet Load
img_set, label_set = load_img_folder()
print('img_set.shape :',img_set.shape)
print('label_set.shape :',len(label_set))
```

    img_set.shape : (16585, 128, 128, 3)
    label_set.shape : 16585
    


```python
# Image Information
type(img_set), round(img_set.mean(),4)
```




    (numpy.ndarray, 215.1488)




```python
# Label Information
type(label_set), label_set[:4]
```




    (list,
     ['Item-Image\\0928015_F.jpg',
      'Item-Image\\0929029_F.jpg',
      'Item-Image\\1008001_F.jpg',
      'Item-Image\\1008004_F.jpg'])




```python
# Image Sample
plt.imshow(img_set[200])
```


```python
# Multi Image Samples
_ = plot_images(2,5,img_set)
```


## 2. Data Preprocessing


```python
# Data Normalization
img_scaled = img_set / 255.0
print('Raw Image Format :',img_set.shape, img_set.mean())
print('Scaled Image Format :',img_scaled.shape, img_scaled.mean())
```

    Raw Image Format : (16585, 128, 128, 3) 215.1488384048635
    Scaled Image Format : (16585, 128, 128, 3) 0.8437209349210284
    


```python
# Train, Test Data Split
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(img_scaled, label_set, random_state=42, test_size=0.2, shuffle=True)
print(X_train.shape, len(y_train))
print(X_test.shape, len(y_test))
```

    (13268, 128, 128, 3) 13268
    (3317, 128, 128, 3) 3317
    

## 3. Convolutional Autoencoder Modeling

### Encoder


```python
# Encoder Part Modeling
tf.keras.backend.clear_session()
encoder_input = Input(shape=(imgR,imgC,channel))

# Fisrt ConvPooling Layer : 128
L1 = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_input)
L2 = MaxPooling2D((2, 2))(L1)

# Second ConvPooling Layer : 64
L3 = Conv2D(64, (3, 3), activation='relu', padding='same')(L2)
L4 = Conv2D(64, (3, 3), activation='relu', padding='same')(L3)
L5 = BatchNormalization()(L4)
L6 = MaxPooling2D((2, 2))(L5)

# Third ConvPooling Layer : 32
L7 = Conv2D(32, (3, 3), activation='relu', padding='same')(L6)
L8 = Conv2D(32, (3, 3), activation='relu', padding='same')(L7)
L9 = BatchNormalization()(L8)
L10 = MaxPooling2D((2, 2))(L9)

# Fourth ConvPooling Layer : 16
L11 = Conv2D(16, (3, 3), activation='relu', padding='same')(L10)
L12 = Conv2D(16, (3, 3), activation='relu', padding='same')(L11)
L13 = BatchNormalization()(L12)
L14 = MaxPooling2D((2, 2))(L13)

# Fifth ConvPooling Layer : 8
L15 = Conv2D(8, (3, 3), activation='relu', padding='same')(L14)
L16 = MaxPooling2D((2, 2))(L15)

encoder_output = L16
```


```python
# Encoder Summary()
encoder = tf.keras.Model(encoder_input, encoder_output)
encoder.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 128, 128, 3)]     0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 128, 128, 128)     3584      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 64, 64, 128)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 64, 64)        73792     
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 64, 64, 64)        36928     
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 64)        256       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 32, 32, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 32, 32, 32)        18464     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 32, 32, 32)        9248      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 32, 32, 32)        128       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 16)        4624      
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 16, 16, 16)        2320      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 16, 16, 16)        64        
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 8, 8, 16)          0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 8, 8, 8)           1160      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 4, 4, 8)           0         
    =================================================================
    Total params: 150,568
    Trainable params: 150,344
    Non-trainable params: 224
    _________________________________________________________________
    


```python
# Plot Encoder Diagram
plot_model(encoder, to_file='figure/Eecoder0818.png', show_shapes=True)
```

    ('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')
    

### Decoder


```python
# Decoder Part Modeling
decoder_input = Input(shape=(4,4,8))  # Decoder의 Input Shape는 Hard Coding이 필요함. (개선점)

# First ConvPooling Layer : 8
L17 = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_input)
L18 = UpSampling2D((2, 2))(L17)

# Second ConvPooling Layer : 16
L19 = Conv2D(16, (3, 3), activation='relu', padding='same')(L18)
L20 = Conv2D(16, (3, 3), activation='relu', padding='same')(L19)
L21 = BatchNormalization()(L20)
L22 = UpSampling2D((2, 2))(L21)

# Third ConvPooling Layer : 32
L23 = Conv2D(32, (3, 3), activation='relu', padding='same')(L22)
L24 = Conv2D(32, (3, 3), activation='relu', padding='same')(L23)
L25 = BatchNormalization()(L24)
L26 = UpSampling2D((2, 2))(L25)

# Fourth ConvPooling Layer : 64
L27 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(L26)
L28 = Conv2D(64, (3, 3), activation='relu', padding='same')(L27)
L29 = BatchNormalization()(L28)
L30 = UpSampling2D((2, 2))(L29)

# Fifth ConvPooling Layer : 128
L31 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(L30)
L32 = UpSampling2D((2, 2))(L31)

# Sixth ConvPooling Layer : 3
L33 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(L32)

decoder_output = L33
```


```python
# Decoder Summary()
decoder = tf.keras.Model(decoder_input, decoder_output)
decoder.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 4, 4, 8)]         0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 4, 4, 8)           584       
    _________________________________________________________________
    up_sampling2d (UpSampling2D) (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 8, 8, 16)          1168      
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 8, 8, 16)          2320      
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 8, 8, 16)          64        
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 16, 16, 16)        0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 16, 16, 32)        4640      
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 16, 16, 32)        9248      
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 16, 16, 32)        128       
    _________________________________________________________________
    up_sampling2d_2 (UpSampling2 (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 32, 32, 64)        18496     
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 32, 32, 64)        36928     
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 32, 32, 64)        256       
    _________________________________________________________________
    up_sampling2d_3 (UpSampling2 (None, 64, 64, 64)        0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 64, 64, 128)       73856     
    _________________________________________________________________
    up_sampling2d_4 (UpSampling2 (None, 128, 128, 128)     0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 128, 128, 3)       3459      
    =================================================================
    Total params: 151,147
    Trainable params: 150,923
    Non-trainable params: 224
    _________________________________________________________________
    


```python
# Plot Decoder Diagram
plot_model(decoder, to_file='figure/Decoder0818.png', show_shapes=True)
```

    ('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')
    

### AutoEncoder (Encoder + Decoder)


```python
# Convolutional Autoencoder Modeling
# Connecting Encoder & Decoder Part

init_input = Input(shape=(imgR,imgC,channel))
connect_input = encoder(init_input)
connect_output = decoder(connect_input)

model = tf.keras.Model(init_input, connect_output)
```


```python
# Model Compile
model.compile(optimizer='Adam',loss='binary_crossentropy')
```


```python
# Convolutional Autoencoder Summary
model.summary()
```

    Model: "model_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 128, 128, 3)]     0         
    _________________________________________________________________
    model (Functional)           (None, 4, 4, 8)           150568    
    _________________________________________________________________
    model_1 (Functional)         (None, 128, 128, 3)       151147    
    =================================================================
    Total params: 301,715
    Trainable params: 301,267
    Non-trainable params: 448
    _________________________________________________________________
    


```python
# Plot ConvAE Diagram
plot_model(model, to_file='figure/ConvAE0818.png', show_shapes=True)
```

    ('You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')
    


```python
# Checkpoint Callback Function Definition
checkpoint_dir = 'Training-checkpoint/'
checkpoint_path = checkpoint_dir + 'cp-{epoch:04d}-{val_loss:.2f}.ckpt'

# 10번 에포크씩 val_loss 변화 확인- 변화 없을 시 학습 중단
patience_epoch = 20
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_epoch)
cp = ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                     save_weights_only=True,
                     save_best_only=True)
```


```python
# TQDM Tracking Conv-AE Model Training
nb_epochs = 100
batch_size = 64

start = time.time()
tqdm_callback = tfa.callbacks.TQDMProgressBar()
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, shuffle=True,
                    callbacks=[early_stopping, cp, tqdm_callback], validation_split=0.05).history
end = time.time()
```


```python
# Print Training Time
train_time = end-start
result = dt.timedelta(seconds=train_time)
print('Training Time :',str(result).split('.')[0])
```

    Training Time : 3:25:02
    


```python
# Training History DataFrame
df_hist = pd.DataFrame(history)
df_hist.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>92</th>
      <td>0.208783</td>
      <td>0.220412</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.208756</td>
      <td>0.214018</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.208805</td>
      <td>0.223127</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.208726</td>
      <td>0.212937</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.208711</td>
      <td>0.219324</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Best Estimator
df_hist.sort_values(by='val_loss').head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76</th>
      <td>0.209124</td>
      <td>0.212793</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save Model
# Encoder Part Save
encoder.save('model_save/Encoder_0818/')

# Decoder Part Save
decoder.save('model_save/Decoder_0818/')

# Convolutional Autoencoder Model Save
model.save('model_save/ConvAE_0818/')
```

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: model_save/Encoder_0818/assets
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    INFO:tensorflow:Assets written to: model_save/Decoder_0818/assets
    INFO:tensorflow:Assets written to: model_save/ConvAE_0818/assets
    


```python
# Load Model
encoder = keras.models.load_model('model_save_2/model_save/Encoder_0818/')
decoder = keras.models.load_model('model_save_2/model_save/Decoder_0818/')
model = keras.models.load_model('model_save_2/model_save/ConvAE_0818/')
```

    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    

## 4. Performance Evaluation


```python
# Loss Learning Curve
plt.figure(figsize=(8,6))
plt.title('Loss Learning Curve')
plt.plot(df_hist.loss, label='loss', color='black', linewidth=2.0)
plt.plot(df_hist.val_loss, label='val_loss', color='green', linewidth=2.0)
plt.axvline(x=df_hist.shape[0]-patience_epoch, color='r', linestyle='--',label='best epoch')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(['Training Loss', 'Validation Loss', 'Saturation epoch'], fontsize=14)
plt.show()
```

```python
# Test Data Evaluation
test_loss = model.evaluate(X_test,X_test)
print('test loss :',np.round(test_loss,4))
```

    104/104 [==============================] - 321s 3s/step - loss: 0.2163
    test loss : 0.2163
    


```python
# Test Data Prediction(Reconstruction)
X_pred = model.predict(X_test)
X_pred.shape,round(X_pred.mean(),4)
```

    104/104 [==============================] - 289s 3s/step
    




    ((3317, 128, 128, 3), 0.8249)




```python
# Test Image Samples
_ = plot_images(2,5,X_test)
```


```python
# Prediction Image Samples
_ = plot_images(2,5,X_pred)
```


```python
# Reconstruction Error 
X_error = Reconstruction_Error(X_test,X_pred)
len(X_error), type(X_error)
```




    (3317, list)




```python
# Reconstruction Error Visaulization
X_loop = np.arange(len(X_error))
boundary = 0.05

plt.figure(figsize=(10,6))
sns.scatterplot(X_loop, X_error, color = 'red', alpha=0.5, marker='*', label='Error Points')
sns.lineplot(X_loop, boundary, color='blue', linestyle='--', label='95% CI Boundary', alpha=0.8)
plt.title('Reconstruction Error Variances', size=18)
plt.legend(loc="upper right")
plt.ylim(0.00,0.10)
plt.show()
```

## 5. Latent Space Projection


```python
# Raw Latent Feature
raw_feature = encoder.predict(img_scaled)
raw_feature.shape, type(raw_feature)
```


```python
# Deep Compact Latent 8 Features 
latent_feature = AVGpooling(raw_feature)
latent_feature.shape, type(latent_feature)
```




    ((16585, 8), numpy.ndarray)




```python
# Latent Feature DataFrame
comp_list = ['comp1','comp2','comp3','comp4','comp5','comp6','comp7','comp8','label']
df = pd.DataFrame(latent_feature)
df = pd.concat([df,pd.Series(label_set)],axis=1)
df.columns = comp_list
print('df.shape :',df.shape)
df.head()
```


```python
# Latent Feature DataFrame export to csv
df.to_csv('latent_feature_0818.csv',index=False)
```

## 6. Similarity Calculation & Top10 Item Return


```python
# Latent Feature DataFrame load D:\Fasion_Images
df = pd.read_csv('D:/Fasion_Images/latent_feature_0818.csv')
print(df.shape)
df.head()
```

    (16585, 9)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comp1</th>
      <th>comp2</th>
      <th>comp3</th>
      <th>comp4</th>
      <th>comp5</th>
      <th>comp6</th>
      <th>comp7</th>
      <th>comp8</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.816822</td>
      <td>1.886551</td>
      <td>1.807105</td>
      <td>0.858414</td>
      <td>1.428525</td>
      <td>2.295357</td>
      <td>0.791543</td>
      <td>1.505969</td>
      <td>0928015_F.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.904211</td>
      <td>1.930044</td>
      <td>1.972450</td>
      <td>0.783947</td>
      <td>1.302266</td>
      <td>2.407675</td>
      <td>0.821974</td>
      <td>1.499656</td>
      <td>0929029_F.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.471252</td>
      <td>6.036780</td>
      <td>5.894671</td>
      <td>2.869880</td>
      <td>2.186035</td>
      <td>10.727018</td>
      <td>2.532120</td>
      <td>5.862957</td>
      <td>1008001_F.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.119862</td>
      <td>4.681663</td>
      <td>5.019187</td>
      <td>2.043906</td>
      <td>1.902202</td>
      <td>7.571554</td>
      <td>1.737040</td>
      <td>4.642845</td>
      <td>1008004_F.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.114837</td>
      <td>3.997970</td>
      <td>3.995601</td>
      <td>1.616894</td>
      <td>1.459364</td>
      <td>5.931483</td>
      <td>1.358331</td>
      <td>3.653468</td>
      <td>1008006_F.jpg</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Latent Feature Data & Label Split
data = df.drop('label',axis=1)
label = df['label']
print(data.shape, label.shape)
```

    (16585, 8) (16585,)
    

### Cosine Similarity


```python
# data downcasting
data = data.astype('float32')
```


```python
# Cosine Similarity Computation
from sklearn.metrics.pairwise import cosine_similarity

cosine_matrix = cosine_similarity(data, data)
print('cosine_matrix.shape :',cosine_matrix.shape)
cosine_matrix[:4,:4]
```

    cosine_matrix.shape : (16585, 16585)
    




    array([[1.        , 0.9987278 , 0.97110724, 0.97986597],
           [0.9987278 , 0.99999994, 0.97669864, 0.98550713],
           [0.97110724, 0.97669864, 0.99999994, 0.9974694 ],
           [0.97986597, 0.98550713, 0.9974694 , 1.        ]], dtype=float32)




```python
# Cosine Similarity Visualization
plt.figure(figsize=(10,8))
plt.title('Latent Feature Cosine Similarity')
sns.heatmap(cosine_matrix[:100, :100], cmap='RdBu')
plt.show()
```

```python
# 기존 이미지 중 단일 샘플 추출
sample_idx = np.random.choice(np.arange(df.shape[0]),1)[0]
sample = data.loc[sample_idx,:].values.reshape(1,-1)
print('sample_idx :',sample_idx)
```

    sample_idx : 5390
    


```python
# 기존 이미지 중 단일 샘플에 대한 코사인 유사도 계산식
sample_cosine_sim = cosine_similarity(sample, data)

print('sample_cosine_sim.shape :',sample_cosine_sim.shape)
```

    sample_cosine_sim.shape : (1, 16585)
    


```python
# # 새로운 이미지 중 단일 샘플에 대한 코사인 유사도 계산식
# smp = New_image_preprocessing('sample.jpg')
# smp_emb = encoder.predict(np.reshape(smp,(1,64,64,3)))
# smp_comp = AVGpooling(smp_emb)
# print('smp.shape :',smp.shape)
# plt.imshow(smp)

# sample_cosine_sim = cosine_similarity(smp_comp, data)
# print('sample_cosine_sim.shape :',sample_cosine_sim.shape)
# print()
```


```python
# Define Sample Similarity DataFrame
df_cosine = pd.DataFrame(sample_cosine_sim.T, index=df.index, columns=['sample'])
print('df_cosine.shape :',df_cosine.shape)
df_cosine.head(4)
```

    df_cosine.shape : (16585, 1)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.963485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.972127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.996787</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.997025</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Return Top10 Similar Items
top10_idx_cosine = df_cosine['sample'].nlargest(10).index
top10_label_cosine = label[top10_idx_cosine].values
print('top10_idx_cosine :',top10_idx_cosine)
print(top10_label_cosine)
```

    top10_idx_cosine : Int64Index([5390, 6126, 5178, 4123, 5665, 7182, 7177, 6353, 7232, 7678], dtype='int64')
    ['1103045_F.jpg' '1105054_F.jpg' '1102111_F.jpg' '1028064_F.jpg'
     '1103367_F.jpg' '1109245_F.jpg' '1109239_F.jpg' '1105319_F.jpg'
     '1109301_F.jpg' '1110275_F.jpg']
    


```python
# 기존 이미지 중 단일 샘플 이미지
print(df.loc[sample_idx,'label'])
_ = plt.imshow(img_set[sample_idx])
```

```python
# Top10 유사 이미지 시각화
top10_visualize(img_set,top10_idx_cosine)
```

### Euclidean Distance Calculation


```python
# Euclidean Distance Calculation
from sklearn.metrics.pairwise import euclidean_distances

ec_matrix = 1 / euclidean_distances(data, data)
print('ec_matrix.shape :',ec_matrix.shape)
ec_matrix[:4,:4]
```

    ec_matrix.shape : (16585, 16585)
    




    array([[       inf, 3.7284596 , 0.08076027, 0.12017515],
           [3.7284596 ,        inf, 0.08180973, 0.12260531],
           [0.08076027, 0.08180973,        inf, 0.24063474],
           [0.12017515, 0.12260531, 0.24063474,        inf]], dtype=float32)




```python
# Euclidean Distance Visualization
plt.figure(figsize=(10,8))
plt.title('Latent Feature Euclidean Distance')
sns.heatmap(ec_matrix[:100, :100], cmap='RdBu')
plt.show()
```


```python
# 기존 이미지 중 단일 샘플에 대한 유클리디안 거리 계산식
sample_ec_distance = 1 / euclidean_distances(sample, data)

print('sample_ec_distance.shape :',sample_ec_distance.shape)
```

    sample_ec_distance.shape : (1, 16585)
    


```python
# Define Sample Similarity DataFrame
df_ec = pd.DataFrame(sample_ec_distance.T, index=df.index, columns=['sample'])
print('df_ec.shape :',df_ec.shape)
df_ec.head(4)
```

    df_ec.shape : (16585, 1)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.155935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.159182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.158337</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.394013</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Return Top10 Similar Items
top10_idx_ec = df_ec['sample'].nlargest(10).index
print('top10_idx_ec :',top10_idx_ec)
print(label[top10_idx_ec].values)
```

    top10_idx_ec : Int64Index([860, 15170, 931, 355, 783, 2415, 1082, 1384, 15172, 9433], dtype='int64')
    ['1014162_F.jpg' '1208281_F.jpg' '1014240_F.jpg' '1012304_F.jpg'
     '1014080_F.jpg' '1020250_F.jpg' '1015009_F.jpg' '1015412_F.jpg'
     '1208283_F.jpg' '1116157_F.jpg']
    


```python
# 기존 이미지 중 단일 샘플 이미지
print(df.loc[sample_idx,'label'])
_ = plt.imshow(img_set[sample_idx])
```


```python
# Top10 유사 이미지 시각화
top10_visualize(img_set,top10_idx_ec)
```


### Pearson Similarity


```python
# Pearson Similarity Computation
pearson_sim = np.corrcoef(data.to_numpy())
print('pearson_sim.shape :',pearson_sim.shape)
pearson_sim[:4,:4]
```

    pearson_sim.shape : (16585, 16585)
    




    array([[1.        , 0.99104718, 0.87296293, 0.91160757],
           [0.99104718, 1.        , 0.89450709, 0.93525527],
           [0.87296293, 0.89450709, 1.        , 0.9877586 ],
           [0.91160757, 0.93525527, 0.9877586 , 1.        ]])




```python
# Pearson Similarity Visualization
plt.figure(figsize=(10,8))
plt.title('Latent Feature Pearson Similarity')
sns.heatmap(pearson_sim[:100, :100], cmap='RdBu')
plt.show()
```

```python
# 기존 이미지 중 단일 샘플에 대한 코사인 유사도 계산식
sample_pearson_sim = np.corrcoef(x=data.to_numpy(),y=sample)

print('sample_pearson_sim.shape :',sample_pearson_sim.shape)
```

    sample_pearson_sim.shape : (16586, 16586)
    


```python
# Define Sample Similarity DataFrame
df_pearson = pd.DataFrame(sample_pearson_sim[-1,:-1], index=df.index, columns=['sample'])
print('df_pearson.shape :',df_pearson.shape)
df_pearson.head(4)
```

    df_pearson.shape : (16585, 1)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.854181</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.867406</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.953587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.958042</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Return Top10 Similar Items
top10_idx_pearson = df_pearson['sample'].nlargest(10).index
print('top10_idx_pearson :',top10_idx_pearson)
print(label[top10_idx_pearson].values)
```

    top10_idx_pearson : Int64Index([860, 1441, 732, 1727, 1363, 1724, 1487, 14339, 773, 1141], dtype='int64')
    ['1014162_F.jpg' '1015529_F.jpg' '1014025_F.jpg' '1016276_F.jpg'
     '1015359_F.jpg' '1016273_F.jpg' '1016015_F.jpg' '1203183_F.jpg'
     '1014069_F.jpg' '1015080_F.jpg']
    


```python
# 기존 이미지 중 단일 샘플 이미지
print(df.loc[sample_idx,'label'])
_ = plt.imshow(img_set[sample_idx])
```


```python
# Top10 유사 이미지 시각화
top10_visualize(img_set,top10_idx_pearson)
```


## 7. Fashion Coordination Recommendation

### Fashion Coordination Dataframe Design


```python
# fashion coordination dataframe
fashion_df = pd.read_csv('D:/Fasion_Images/uni_wearing.csv')
print('fashion_df.shape :',fashion_df.shape)
fashion_df.head()
```

    fashion_df.shape : (18040, 6)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wearing</th>
      <th>hat</th>
      <th>main_top</th>
      <th>inner_top</th>
      <th>bottom</th>
      <th>shoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1008_1008_720_A_A001_A001_000.jpg</td>
      <td>1008013.0</td>
      <td>1008011</td>
      <td>NaN</td>
      <td>1008012.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1030_1030_720_A_A002_232_223_222_A002_000.jpg</td>
      <td>1029449.0</td>
      <td>1029157</td>
      <td>NaN</td>
      <td>1029107.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1030_1030_720_A_A003_232_220_222_A003_000.jpg</td>
      <td>1029442.0</td>
      <td>1029411</td>
      <td>NaN</td>
      <td>1029109.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1030_1030_720_B_B002_232_221_223_B002_000.jpg</td>
      <td>1029434.0</td>
      <td>1029073</td>
      <td>NaN</td>
      <td>1029141.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1030_1030_720_B_B003_232_227_223_B003_000.jpg</td>
      <td>1029431.0</td>
      <td>1029255</td>
      <td>NaN</td>
      <td>1029142.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fashion dataframe Information
fashion_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18040 entries, 0 to 18039
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   wearing    18040 non-null  object 
     1   hat        32 non-null     float64
     2   main_top   18040 non-null  int64  
     3   inner_top  2838 non-null   float64
     4   bottom     16224 non-null  float64
     5   shoes      125 non-null    float64
    dtypes: float64(4), int64(1), object(1)
    memory usage: 845.8+ KB
    


```python
# fashion dataframe 
print(fashion_df.isna().sum())
```

    wearing          0
    hat          18008
    main_top         0
    inner_top    15202
    bottom        1816
    shoes        17915
    dtype: int64
    


```python
# fashion dataframe Missing Value Imputation
fashion_df.fillna(0,inplace=True)
print('Remain Missing Value :',fashion_df.isna().sum().sum())
```

    Remain Missing Value : 0
    


```python
# fashion data preprocessing
fashion_df.iloc[:,1:] = fashion_df.iloc[:,1:].astype(int)
fashion_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wearing</th>
      <th>hat</th>
      <th>main_top</th>
      <th>inner_top</th>
      <th>bottom</th>
      <th>shoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1008_1008_720_A_A001_A001_000.jpg</td>
      <td>1008013</td>
      <td>1008011</td>
      <td>0</td>
      <td>1008012</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1030_1030_720_A_A002_232_223_222_A002_000.jpg</td>
      <td>1029449</td>
      <td>1029157</td>
      <td>0</td>
      <td>1029107</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1030_1030_720_A_A003_232_220_222_A003_000.jpg</td>
      <td>1029442</td>
      <td>1029411</td>
      <td>0</td>
      <td>1029109</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1030_1030_720_B_B002_232_221_223_B002_000.jpg</td>
      <td>1029434</td>
      <td>1029073</td>
      <td>0</td>
      <td>1029141</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1030_1030_720_B_B003_232_227_223_B003_000.jpg</td>
      <td>1029431</td>
      <td>1029255</td>
      <td>0</td>
      <td>1029142</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Match the coordination


```python
# top10 label by cosine similarity
top10_label_cosine
```




    array(['1103045_F.jpg', '1105054_F.jpg', '1102111_F.jpg', '1028064_F.jpg',
           '1103367_F.jpg', '1109245_F.jpg', '1109239_F.jpg', '1105319_F.jpg',
           '1109301_F.jpg', '1110275_F.jpg'], dtype=object)




```python
# top10 label preprocessing
top10_result_cosine = list(map(lambda x : int(x[:7]), top10_label_cosine))
top10_result_cosine
```




    [1103045,
     1105054,
     1102111,
     1028064,
     1103367,
     1109245,
     1109239,
     1105319,
     1109301,
     1110275]




```python
# top10 Recommendation Result
recomm_df = Fashion_coordination(top10_result_cosine, fashion_df)
print('recomm_df :',len(recomm_df),'DataFrame')
for i in range(len(recomm_df)):
  print(f"{i+1} item's coordination cases :",recomm_df[i].shape[0])
```

    recomm_df : 10 DataFrame
    1 item's coordination cases : 1
    2 item's coordination cases : 1
    3 item's coordination cases : 3
    4 item's coordination cases : 2
    5 item's coordination cases : 1
    6 item's coordination cases : 1
    7 item's coordination cases : 1
    8 item's coordination cases : 1
    9 item's coordination cases : 3
    10 item's coordination cases : 1
    


```python
# Total Similar Item Coordination DataFrame
recomm_total = pd.concat(recomm_df,axis=0)
recomm_total.reset_index(drop=True,)
print('recomm_total.shape :',recomm_total.shape)
recomm_total.head()
```

    recomm_total.shape : (15, 6)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wearing</th>
      <th>hat</th>
      <th>main_top</th>
      <th>inner_top</th>
      <th>bottom</th>
      <th>shoes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6720</th>
      <td>1104_1104_720_A_A078_172_274_017_A078_000.jpg</td>
      <td>0</td>
      <td>1103045</td>
      <td>1103303</td>
      <td>1103338</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7973</th>
      <td>1106_1106_720_C_C115_298_290_288_C115_000.jpg</td>
      <td>0</td>
      <td>1105335</td>
      <td>1105054</td>
      <td>1105031</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6297</th>
      <td>1103_1103_720_B_B020_263_260_B020_000.jpg</td>
      <td>0</td>
      <td>1102111</td>
      <td>0</td>
      <td>1102055</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6360</th>
      <td>1103_1103_720_B_B112_263_078_B112_000.jpg</td>
      <td>0</td>
      <td>1102111</td>
      <td>0</td>
      <td>1102023</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6456</th>
      <td>1103_1103_720_C_C097_263_263_C097_000.jpg</td>
      <td>0</td>
      <td>1102111</td>
      <td>0</td>
      <td>1102116</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Recommendation Service Output Result


```python
# Recommendation Service Output Result : 2
count = 0
fig = plt.figure()
fig, ax = plt.subplots(2,5,figsize=(5*3,2*3))
plt.suptitle('Fashion Best Fit Recommendatation!!')
for i in range(2):
  for j in range(5):
    axis = ax[i,j]
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    try:
      fashion_img_name = recomm_total.iloc[count,0]
      img = img_read(model_dir + '/' + fashion_img_name)
      axis.imshow(img)
      plt.axis('off')
    except:
      pass
    count+=1
plt.show()
```
