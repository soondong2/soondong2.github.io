---
title: "COVID-19 Voice Detector"
date: 2023-03-03

categories:
  - Projects
tags:
    - DSP
    - ML
---

# COVID-19 Voice Detection
![image](https://user-images.githubusercontent.com/100760303/222647064-a81befdf-a88c-40ea-8e58-a7e21708dbfd.png)
![image](https://user-images.githubusercontent.com/100760303/222647111-b7ada789-1435-4e5a-9f7d-746b6b9ca93b.png)


**[Data Information]**<br>   
Data Source: https://dacon.io/competitions/official/235910/overview/description   
Raw Data Type : wav, csv

**[Voice Data]**<br>   
train_wav : 3805개   
test_wav : 5732개   
unlabeled_wav : 1867개   

**[Voice Data Format]**<br>   
CFG = {   
    'SR' : 16000,     # 주파수(Sample_rate)   
    'N_MFCC' : 15,    # MFCC 벡터를 추출할 개수   
    'SEED' : 41,      # 시드(Random Seed)   
    'N_MELS' : 128    # 고주파 대역 스케일(Mel Scale)   
}   

**[Csv Data]**<br>
train_csv : (3805,6)   
test_csv : (5732,5)   
unlabeled_csv : (1867,5)   

**[Version]**<br>   
1st Advanced MFCC Feature, f1_score, MLP, Semi-Supervised Learning

## 0. Setting

### Path Check


```python
pwd
```




    'C:\\myPyCode\\Code lion\\COVID-19 Voice Detection'




```python
ls
```

     C 드라이브의 볼륨에는 이름이 없습니다.
     볼륨 일련 번호: A494-C8A8
    
     C:\myPyCode\Code lion\COVID-19 Voice Detection 디렉터리
    
    2022-08-23  오후 05:46    <DIR>          .
    2022-08-23  오후 05:46    <DIR>          ..
    2022-08-23  오후 05:02    <DIR>          .ipynb_checkpoints
    2022-08-23  오후 05:46         1,043,358 COVID-19 Voice Detection MLP Semi Supervised Main.ipynb
    2022-08-23  오후 04:04    <DIR>          data
    2022-08-23  오후 01:53    <DIR>          docs
    2022-08-23  오후 04:35    <DIR>          low version
    2022-08-23  오후 04:05    <DIR>          model_save
                   1개 파일           1,043,358 바이트
                   7개 디렉터리  14,035,189,760 바이트 남음
    

### Library Call


```python
# pip install librosa
```


```python
# pip install xgboost
```


```python
# pip install lightgbm
```


```python
# 상용 라이브러리
from glob import glob
import os
import pandas as pd
import numpy as np
import datetime as dt
import time
from tqdm import tqdm
import librosa
import random
import IPython
import pickle

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
# sckit-learn preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# ML Model Library
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC ,LinearSVC
from sklearn.ensemble import  RandomForestClassifier ,AdaBoostClassifier ,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBoostClassifier, plot_importance
from lightgbm import LGBMClassifier, plot_importance

# Clf Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, precision_recall_curve
```

### User Function Definition


```python
# ------------- WAV File Load & Preprocessing ------------- #
# Global Constant Definition
root_dir = 'C:\\myPyCode\\Code lion\\COVID-19 Voice Detection'
CFG = {
    'SR' : 16000,     # 주파수(Sample_rate)
    'N_MFCC' : 15,    # MFCC 벡터를 추출할 개수
    'SEED' : 41,      # 시드(Random Seed)
    'N_MELS' : 128    # 고주파 대역 스케일(Mel Scale)
}

# Fixing Every Random Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED']) # Seed 고정

# MFCC(Mel-Frequency Cepstral Coefficient) : 음성데이터를 다루기 쉬운 형태로 피처화
def get_mfcc_feature(df, data_type, save_path):
    # Data Folder path
    root_folder = './data'
    if os.path.exists(save_path):
        print(f'{save_path} is exist.')
        return
    features = []
    for uid in tqdm(df['id']):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5)+'.wav')

        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(path, sr=CFG['SR'])
        
        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])

        y_feature = []
        # 추출된 MFCC들의 평균을 Feature로 사용
        for e in mfcc:
            y_feature.append(np.mean(e))
        features.append(y_feature)
    
    # 기존의 자가진단 정보를 담은 데이터프레임에 추출된 오디오 Feature를 추가
    mfcc_df = pd.DataFrame(features, columns=['mfcc_'+str(x) for x in range(1,CFG['N_MFCC']+1)])
    df = pd.concat([df, mfcc_df], axis=1)
    df.to_csv(save_path, index=False)
    print('Done.')

# One-Hot Encoding
def onehot_encoding(ohe, x):
    # 학습데이터로 부터 fit된 one-hot encoder (ohe)를 받아 transform 시켜주는 함수
    encoded = ohe.transform(x['gender'].values.reshape(-1,1))
    encoded_df = pd.DataFrame(encoded, columns=ohe.categories_[0])
    x = pd.concat([x.drop(columns=['gender']), encoded_df], axis=1)
    return x

# ------------- WAV File Visualization ------------- #
# WAV Plot
def wav_plot(y, sr):
    time = np.linspace(0, len(y)/sr, len(y))
    fig, ax = plt.subplots()
    ax.plot(time, y, color = 'g')
    ax.set_xlabel('Time')
    ax.set_ylabel('Voice')
    plt.title('WAV Plot')
    plt.show()

# WAV Display
def wav_display(y, sr):
    return IPython.display.Audio(data = y, rate = sr)

# ------------- ML Modeling ------------- #
# ML GridSearchCV
def ML_search(X_train,y_train,X_test,y_test):
    lr = LogisticRegression(random_state=42)
    svm = SVC(random_state=42)
    rf  = RandomForestClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    gus = GaussianNB()
    knn = KNeighborsClassifier()
    lin_svc = LinearSVC(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    grd  = GradientBoostingClassifier(random_state=42)
    lgbm = LGBMClassifier(random_state=42)
    mlp = MLPClassifier(random_state=42)
    
    models = [lr, svm, rf, dt, gus, knn, lin_svc, ada, grd, lgbm, mlp]
    models_result = []
    
    for model in models:
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        pre = precision_score(y_test,pred)
        rec = recall_score(y_test,pred)
        f1 = f1_score(y_test,pred)
        acc = accuracy_score(y_test,pred)
        models_result.append([model,acc,f1,pre,rec])
        
    df = pd.DataFrame(models_result, columns=['name','acc','f1','precision','recall'])
    return df

# Get MLPClassifier Format
def get_mlp_format(model):
    print('n_features :',model.n_features_in_)
    print('n_iter :',model.n_iter_)
    print('n_layers :',model.n_layers_)
    print('n_outputs :',model.n_outputs_)
    print('out_activation :',model.out_activation_)
    return None

# MLP Post Tuning
def mlp_post_tuning(y_pred_proba_1,lower,upper):
    post_result = pd.DataFrame(columns=['acc','pre','rec','f1'])
    proba_grid = np.arange(lower,upper,0.01)
    for proba in proba_grid:
        ele =  np.where(y_pred_proba_1 < proba, 0, 1)
        ele_result = get_clf_eval(y_valid, ele)
        post_result = pd.concat([post_result,pd.DataFrame(ele_result,index=['acc','pre','rec','f1']).T],axis=0)
    post_result.index = proba_grid
    post_result.reset_index(inplace=True)
    post_result.rename(columns={'index':'proba'},inplace=True)
    return post_result

# ------------- Performance Evaluation Metrics ------------- #
# Get Clf Evaluation Metrics
def get_clf_eval(y_test,pred):
    cf = confusion_matrix(y_test,pred)
    acc = accuracy_score(y_test,pred)
    pre = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    
    print(cf)
    print('정확도 :{0:.4f}, 정밀도 :{1:.4f}, 재현율 :{2:.4f}, F1 :{3:.4f}'.format(acc,pre,rec,f1))
    return [acc, pre, rec, f1]        
        
# 정확도 재현율 곡선
def precision_recall_curve_plot(y_test,pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba)
    
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
    plt.axvline(x=0.5, label='Normal Threshold', linestyle=':', alpha=0.8, color='green')
    
    start,end = plt.xlim()
    plt.title('Precision Recall Curve', size=20)
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# ROC 곡선 
def roc_curve_plot(y_test, pred_proba):
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba)
    
    plt.figure(figsize=(8,6))
    plt.plot(fprs, tprs, label='ROC')
    plt.plot([0,1],[0,1],linestyle='--', label='Random', color='k')
    start, end = plt.xlim()
    plt.title('Roc Auc Curve', size=20)
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR(1 - Sensitivity)')
    plt.ylabel('TPR(Recall)')
    plt.legend()
    plt.show()
```

## 1. Data Load


```python
# CSV data load
train_csv = pd.read_csv('data/train_data.csv')
test_csv = pd.read_csv('data/test_data.csv')
unlabeled_csv = pd.read_csv('data/unlabeled_data.csv')

print(train_csv.shape, test_csv.shape, unlabeled_csv.shape)
train_csv.head()
```

    (3805, 6) (5732, 5) (1867, 5)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
      <th>covid19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>51</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>22</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>29</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>23</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_csv.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3806</td>
      <td>48</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3807</td>
      <td>24</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3808</td>
      <td>29</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3809</td>
      <td>39</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3810</td>
      <td>34</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
unlabeled_csv.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9538</td>
      <td>35</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9539</td>
      <td>40</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9540</td>
      <td>33</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9541</td>
      <td>35</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9542</td>
      <td>54</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # MFCC(Mel-Frequency Cepstral Coefficient) Feature 
# get_mfcc_feature(train_csv.copy(), 'train', 'MFCC/train_mfcc_data_added.csv')
# get_mfcc_feature(test_csv.copy(), 'test', 'MFCC/test_mfcc_data_added.csv')
# get_mfcc_feature(unlabeled_csv.copy(), 'unlabeled', 'MFCC/unlabeled_mfcc_data_added.csv')
```


```python
# Train & Test Data Load

# wav 파일의 MFCC Feature와 상태정보를 합친 학습데이터를 불러옵니다.
train_df = pd.read_csv('data/MFCC/train_mfcc_data.csv')
test_df = pd.read_csv('data/MFCC/test_mfcc_data.csv')
unlabeled_df = pd.read_csv('data/MFCC/unlabeled_mfcc_data.csv')

print(train_df.shape, test_df.shape, unlabeled_df.shape)
train_df.head()
```

    (3805, 38) (5732, 37) (1867, 37)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>gender</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
      <th>covid19</th>
      <th>mfcc_1</th>
      <th>mfcc_2</th>
      <th>mfcc_3</th>
      <th>mfcc_4</th>
      <th>...</th>
      <th>mfcc_23</th>
      <th>mfcc_24</th>
      <th>mfcc_25</th>
      <th>mfcc_26</th>
      <th>mfcc_27</th>
      <th>mfcc_28</th>
      <th>mfcc_29</th>
      <th>mfcc_30</th>
      <th>mfcc_31</th>
      <th>mfcc_32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-276.01898</td>
      <td>30.519340</td>
      <td>-20.314617</td>
      <td>-6.689037</td>
      <td>...</td>
      <td>-2.679408</td>
      <td>2.454339</td>
      <td>-1.176285</td>
      <td>2.314315</td>
      <td>-0.339533</td>
      <td>2.514413</td>
      <td>-4.784703</td>
      <td>1.239072</td>
      <td>-1.556883</td>
      <td>-1.548770</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>51</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-312.99362</td>
      <td>54.141323</td>
      <td>-1.748550</td>
      <td>-9.437217</td>
      <td>...</td>
      <td>-7.248304</td>
      <td>1.238725</td>
      <td>-6.894970</td>
      <td>-1.810402</td>
      <td>-7.259594</td>
      <td>0.715029</td>
      <td>-1.372265</td>
      <td>-1.760624</td>
      <td>-2.735181</td>
      <td>1.134190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>22</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-438.60306</td>
      <td>46.675842</td>
      <td>-22.771935</td>
      <td>-3.527922</td>
      <td>...</td>
      <td>-0.136723</td>
      <td>-1.707353</td>
      <td>2.649277</td>
      <td>1.208829</td>
      <td>-0.033701</td>
      <td>-1.008729</td>
      <td>-0.687255</td>
      <td>-0.472232</td>
      <td>0.850565</td>
      <td>0.353839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>29</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-369.26100</td>
      <td>47.762012</td>
      <td>-8.256503</td>
      <td>-2.891349</td>
      <td>...</td>
      <td>-0.389230</td>
      <td>4.033148</td>
      <td>-2.658165</td>
      <td>2.867084</td>
      <td>1.679876</td>
      <td>2.136411</td>
      <td>0.289792</td>
      <td>1.709179</td>
      <td>-0.592465</td>
      <td>1.754549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>23</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-535.68915</td>
      <td>7.509357</td>
      <td>-7.762263</td>
      <td>2.567660</td>
      <td>...</td>
      <td>-0.279360</td>
      <td>-0.292286</td>
      <td>-1.559678</td>
      <td>0.328864</td>
      <td>-1.053423</td>
      <td>0.844060</td>
      <td>-0.788914</td>
      <td>1.182740</td>
      <td>-0.527028</td>
      <td>1.208361</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
# Single Sample
sample_idx = np.random.choice(np.arange(1,train_df.shape[0]),1)[0]
sample_path = 'data/train/' + str(sample_idx).zfill(5) + '.wav'
print(sample_path)
y,sr = librosa.load(sample_path, sr = CFG['SR'])
```

    data/train/01985.wav
    


```python
# WAV Sample Visualization
wav_plot(y, sr)
```
    



```python
# WAV Sample Display
wav_display(y, sr)
```



```python
# Target value Distribution
print(train_df['covid19'].value_counts())

plt.figure(figsize=(8,4))
plt.title('COVID-19 Target Value Distribution')
sns.countplot(train_df['covid19'])
plt.xticks([0,1],['Negative', 'Positive'])
plt.show()
```

    0    3499
    1     306
    Name: covid19, dtype: int64
    




## 2. Data Preprocessing


```python
# Data Split
train_x = train_df.drop(columns=['id', 'covid19'])
train_y = train_df['covid19']
test_x = test_df.drop(columns=['id'])
unlabeled_x = unlabeled_df.drop(columns=['id'])

train_x.shape, train_y.shape, test_x.shape, unlabeled_x.shape
```




    ((3805, 36), (3805,), (5732, 36), (1867, 36))




```python
# OneHotEncoding
ohe = OneHotEncoder(sparse=False)
ohe.fit(train_x['gender'].values.reshape(-1,1))
train_x = onehot_encoding(ohe, train_x)
test_x = onehot_encoding(ohe, test_x)
unlabeled_x = onehot_encoding(ohe, unlabeled_x)

train_x.shape, test_x.shape, unlabeled_x.shape
```




    ((3805, 38), (5732, 38), (1867, 38))




```python
# # StandardScaler
# scaler = StandardScaler()
# train_scaled = scaler.fit_transform(train_x)
# test_scaled = scaler.transform(test_x)
# unlabeled_scaled = scaler.transform(unlabeled_x)
```


```python
# Train Valid Data Split
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, shuffle=True, stratify=train_y, random_state=42)

print('X_train.shape :',X_train.shape)
print('y_train.shape :',y_train.shape)
print('X_test.shape :',X_valid.shape)
print('y_test.shape :',y_valid.shape)
```

    X_train.shape : (3044, 38)
    y_train.shape : (3044,)
    X_test.shape : (761, 38)
    y_test.shape : (761,)
    

## 3. Semi-Supervised Learning


```python
# Pre Semi-Supervised Model : MLPClassifier
semi_model = MLPClassifier(activation = 'relu', solver = 'adam', random_state=42)
```


```python
# Training
start_time = time.time()
semi_model.fit(X_train, y_train)
end_time = time.time()
```


```python
# Training Time Measurements
duration = end_time - start_time
result = dt.timedelta(seconds=duration)
print('Training Time :',str(result).split('.')[0])
```

    Training Time : 0:00:00
    


```python
# Parameter coefficients 
print("Neural Network Parameter's format",semi_model.coefs_[0].shape, semi_model.coefs_[1].shape)
```

    Neural Network Parameter's format (38, 100) (100, 1)
    


```python
# Get MLPClassifier Format
get_mlp_format(semi_model)
```

    n_features : 38
    n_iter : 44
    n_layers : 3
    n_outputs : 1
    out_activation : logistic
    


```python
plt.title('Loss Learning Curve')
plt.plot(semi_model.loss_curve_ )
plt.show()
```




```python
# Test Data Prediction Probability
y_pred_proba = semi_model.predict_proba(X_valid)
y_pred_proba_1 = y_pred_proba[:,1]
```


```python
# Post Tuning
post_result = mlp_post_tuning(y_pred_proba_1,0.01,0.8)
```

    [[ 12 688]
     [  0  61]]
    정확도 :0.0959, 정밀도 :0.0814, 재현율 :1.0000, F1 :0.1506
    [[ 37 663]
     [  3  58]]
    정확도 :0.1248, 정밀도 :0.0804, 재현율 :0.9508, F1 :0.1483
    [[ 73 627]
     [  4  57]]
    정확도 :0.1708, 정밀도 :0.0833, 재현율 :0.9344, F1 :0.1530
    [[131 569]
     [  7  54]]
    정확도 :0.2431, 정밀도 :0.0867, 재현율 :0.8852, F1 :0.1579
    [[191 509]
     [ 10  51]]
    정확도 :0.3180, 정밀도 :0.0911, 재현율 :0.8361, F1 :0.1643
    [[252 448]
     [ 12  49]]
    정확도 :0.3955, 정밀도 :0.0986, 재현율 :0.8033, F1 :0.1756
    [[293 407]
     [ 14  47]]
    정확도 :0.4468, 정밀도 :0.1035, 재현율 :0.7705, F1 :0.1825
    [[341 359]
     [ 18  43]]
    정확도 :0.5046, 정밀도 :0.1070, 재현율 :0.7049, F1 :0.1857
    [[380 320]
     [ 20  41]]
    정확도 :0.5532, 정밀도 :0.1136, 재현율 :0.6721, F1 :0.1943
    [[427 273]
     [ 20  41]]
    정확도 :0.6150, 정밀도 :0.1306, 재현율 :0.6721, F1 :0.2187
    [[453 247]
     [ 21  40]]
    정확도 :0.6478, 정밀도 :0.1394, 재현율 :0.6557, F1 :0.2299
    [[477 223]
     [ 23  38]]
    정확도 :0.6767, 정밀도 :0.1456, 재현율 :0.6230, F1 :0.2360
    [[506 194]
     [ 26  35]]
    정확도 :0.7109, 정밀도 :0.1528, 재현율 :0.5738, F1 :0.2414
    [[530 170]
     [ 29  32]]
    정확도 :0.7385, 정밀도 :0.1584, 재현율 :0.5246, F1 :0.2433
    [[548 152]
     [ 29  32]]
    정확도 :0.7622, 정밀도 :0.1739, 재현율 :0.5246, F1 :0.2612
    [[561 139]
     [ 30  31]]
    정확도 :0.7779, 정밀도 :0.1824, 재현율 :0.5082, F1 :0.2684
    [[579 121]
     [ 31  30]]
    정확도 :0.8003, 정밀도 :0.1987, 재현율 :0.4918, F1 :0.2830
    [[588 112]
     [ 32  29]]
    정확도 :0.8108, 정밀도 :0.2057, 재현율 :0.4754, F1 :0.2871
    [[596 104]
     [ 33  28]]
    정확도 :0.8200, 정밀도 :0.2121, 재현율 :0.4590, F1 :0.2902
    [[609  91]
     [ 34  27]]
    정확도 :0.8357, 정밀도 :0.2288, 재현율 :0.4426, F1 :0.3017
    [[619  81]
     [ 37  24]]
    정확도 :0.8449, 정밀도 :0.2286, 재현율 :0.3934, F1 :0.2892
    [[625  75]
     [ 37  24]]
    정확도 :0.8528, 정밀도 :0.2424, 재현율 :0.3934, F1 :0.3000
    [[631  69]
     [ 39  22]]
    정확도 :0.8581, 정밀도 :0.2418, 재현율 :0.3607, F1 :0.2895
    [[637  63]
     [ 41  20]]
    정확도 :0.8633, 정밀도 :0.2410, 재현율 :0.3279, F1 :0.2778
    [[643  57]
     [ 42  19]]
    정확도 :0.8699, 정밀도 :0.2500, 재현율 :0.3115, F1 :0.2774
    [[648  52]
     [ 43  18]]
    정확도 :0.8752, 정밀도 :0.2571, 재현율 :0.2951, F1 :0.2748
    [[651  49]
     [ 43  18]]
    정확도 :0.8791, 정밀도 :0.2687, 재현율 :0.2951, F1 :0.2812
    [[655  45]
     [ 44  17]]
    정확도 :0.8830, 정밀도 :0.2742, 재현율 :0.2787, F1 :0.2764
    [[655  45]
     [ 45  16]]
    정확도 :0.8817, 정밀도 :0.2623, 재현율 :0.2623, F1 :0.2623
    [[659  41]
     [ 46  15]]
    정확도 :0.8857, 정밀도 :0.2679, 재현율 :0.2459, F1 :0.2564
    [[666  34]
     [ 46  15]]
    정확도 :0.8949, 정밀도 :0.3061, 재현율 :0.2459, F1 :0.2727
    [[667  33]
     [ 49  12]]
    정확도 :0.8922, 정밀도 :0.2667, 재현율 :0.1967, F1 :0.2264
    [[667  33]
     [ 49  12]]
    정확도 :0.8922, 정밀도 :0.2667, 재현율 :0.1967, F1 :0.2264
    [[671  29]
     [ 49  12]]
    정확도 :0.8975, 정밀도 :0.2927, 재현율 :0.1967, F1 :0.2353
    [[671  29]
     [ 49  12]]
    정확도 :0.8975, 정밀도 :0.2927, 재현율 :0.1967, F1 :0.2353
    [[672  28]
     [ 49  12]]
    정확도 :0.8988, 정밀도 :0.3000, 재현율 :0.1967, F1 :0.2376
    [[672  28]
     [ 49  12]]
    정확도 :0.8988, 정밀도 :0.3000, 재현율 :0.1967, F1 :0.2376
    [[673  27]
     [ 49  12]]
    정확도 :0.9001, 정밀도 :0.3077, 재현율 :0.1967, F1 :0.2400
    [[675  25]
     [ 49  12]]
    정확도 :0.9028, 정밀도 :0.3243, 재현율 :0.1967, F1 :0.2449
    [[677  23]
     [ 49  12]]
    정확도 :0.9054, 정밀도 :0.3429, 재현율 :0.1967, F1 :0.2500
    [[678  22]
     [ 49  12]]
    정확도 :0.9067, 정밀도 :0.3529, 재현율 :0.1967, F1 :0.2526
    [[680  20]
     [ 49  12]]
    정확도 :0.9093, 정밀도 :0.3750, 재현율 :0.1967, F1 :0.2581
    [[680  20]
     [ 49  12]]
    정확도 :0.9093, 정밀도 :0.3750, 재현율 :0.1967, F1 :0.2581
    [[680  20]
     [ 51  10]]
    정확도 :0.9067, 정밀도 :0.3333, 재현율 :0.1639, F1 :0.2198
    [[680  20]
     [ 51  10]]
    정확도 :0.9067, 정밀도 :0.3333, 재현율 :0.1639, F1 :0.2198
    [[680  20]
     [ 52   9]]
    정확도 :0.9054, 정밀도 :0.3103, 재현율 :0.1475, F1 :0.2000
    [[681  19]
     [ 52   9]]
    정확도 :0.9067, 정밀도 :0.3214, 재현율 :0.1475, F1 :0.2022
    [[682  18]
     [ 55   6]]
    정확도 :0.9041, 정밀도 :0.2500, 재현율 :0.0984, F1 :0.1412
    [[683  17]
     [ 56   5]]
    정확도 :0.9041, 정밀도 :0.2273, 재현율 :0.0820, F1 :0.1205
    [[685  15]
     [ 57   4]]
    정확도 :0.9054, 정밀도 :0.2105, 재현율 :0.0656, F1 :0.1000
    [[685  15]
     [ 57   4]]
    정확도 :0.9054, 정밀도 :0.2105, 재현율 :0.0656, F1 :0.1000
    [[685  15]
     [ 57   4]]
    정확도 :0.9054, 정밀도 :0.2105, 재현율 :0.0656, F1 :0.1000
    [[685  15]
     [ 57   4]]
    정확도 :0.9054, 정밀도 :0.2105, 재현율 :0.0656, F1 :0.1000
    [[689  11]
     [ 57   4]]
    정확도 :0.9106, 정밀도 :0.2667, 재현율 :0.0656, F1 :0.1053
    [[689  11]
     [ 57   4]]
    정확도 :0.9106, 정밀도 :0.2667, 재현율 :0.0656, F1 :0.1053
    [[689  11]
     [ 57   4]]
    정확도 :0.9106, 정밀도 :0.2667, 재현율 :0.0656, F1 :0.1053
    [[689  11]
     [ 57   4]]
    정확도 :0.9106, 정밀도 :0.2667, 재현율 :0.0656, F1 :0.1053
    [[689  11]
     [ 57   4]]
    정확도 :0.9106, 정밀도 :0.2667, 재현율 :0.0656, F1 :0.1053
    [[690  10]
     [ 57   4]]
    정확도 :0.9120, 정밀도 :0.2857, 재현율 :0.0656, F1 :0.1067
    [[690  10]
     [ 57   4]]
    정확도 :0.9120, 정밀도 :0.2857, 재현율 :0.0656, F1 :0.1067
    [[690  10]
     [ 57   4]]
    정확도 :0.9120, 정밀도 :0.2857, 재현율 :0.0656, F1 :0.1067
    [[690  10]
     [ 57   4]]
    정확도 :0.9120, 정밀도 :0.2857, 재현율 :0.0656, F1 :0.1067
    [[690  10]
     [ 57   4]]
    정확도 :0.9120, 정밀도 :0.2857, 재현율 :0.0656, F1 :0.1067
    [[691   9]
     [ 57   4]]
    정확도 :0.9133, 정밀도 :0.3077, 재현율 :0.0656, F1 :0.1081
    [[691   9]
     [ 57   4]]
    정확도 :0.9133, 정밀도 :0.3077, 재현율 :0.0656, F1 :0.1081
    [[692   8]
     [ 57   4]]
    정확도 :0.9146, 정밀도 :0.3333, 재현율 :0.0656, F1 :0.1096
    [[692   8]
     [ 59   2]]
    정확도 :0.9120, 정밀도 :0.2000, 재현율 :0.0328, F1 :0.0563
    [[692   8]
     [ 59   2]]
    정확도 :0.9120, 정밀도 :0.2000, 재현율 :0.0328, F1 :0.0563
    [[692   8]
     [ 59   2]]
    정확도 :0.9120, 정밀도 :0.2000, 재현율 :0.0328, F1 :0.0563
    [[692   8]
     [ 59   2]]
    정확도 :0.9120, 정밀도 :0.2000, 재현율 :0.0328, F1 :0.0563
    [[694   6]
     [ 59   2]]
    정확도 :0.9146, 정밀도 :0.2500, 재현율 :0.0328, F1 :0.0580
    [[694   6]
     [ 59   2]]
    정확도 :0.9146, 정밀도 :0.2500, 재현율 :0.0328, F1 :0.0580
    [[694   6]
     [ 59   2]]
    정확도 :0.9146, 정밀도 :0.2500, 재현율 :0.0328, F1 :0.0580
    [[694   6]
     [ 59   2]]
    정확도 :0.9146, 정밀도 :0.2500, 재현율 :0.0328, F1 :0.0580
    [[695   5]
     [ 59   2]]
    정확도 :0.9159, 정밀도 :0.2857, 재현율 :0.0328, F1 :0.0588
    [[695   5]
     [ 60   1]]
    정확도 :0.9146, 정밀도 :0.1667, 재현율 :0.0164, F1 :0.0299
    [[695   5]
     [ 60   1]]
    정확도 :0.9146, 정밀도 :0.1667, 재현율 :0.0164, F1 :0.0299
    [[695   5]
     [ 60   1]]
    정확도 :0.9146, 정밀도 :0.1667, 재현율 :0.0164, F1 :0.0299
    [[695   5]
     [ 60   1]]
    정확도 :0.9146, 정밀도 :0.1667, 재현율 :0.0164, F1 :0.0299
    


```python
# Post Tuning Result
post_result.sort_values(by='f1', ascending=False).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>proba</th>
      <th>acc</th>
      <th>pre</th>
      <th>rec</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>0.20</td>
      <td>0.835742</td>
      <td>0.228814</td>
      <td>0.442623</td>
      <td>0.301676</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.22</td>
      <td>0.852825</td>
      <td>0.242424</td>
      <td>0.393443</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.19</td>
      <td>0.819974</td>
      <td>0.212121</td>
      <td>0.459016</td>
      <td>0.290155</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.23</td>
      <td>0.858081</td>
      <td>0.241758</td>
      <td>0.360656</td>
      <td>0.289474</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.21</td>
      <td>0.844941</td>
      <td>0.228571</td>
      <td>0.393443</td>
      <td>0.289157</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Semi-Supervised Model
semi_pred_proba = semi_model.predict_proba(unlabeled_x)
semi_pred_proba_1 = semi_pred_proba[:,1]
```


```python
# Semi-Supervised Model
semi_proba = 0.2
semi_prediction =  np.where(semi_pred_proba_1 < semi_proba, 0, 1)
```


```python
# Unlabeled Data Labeling
unlabeled_x['covid19'] = -1
unlabeled_x['covid19'] = semi_prediction
```


```python
# Unlabeled Data value Distribution
print(unlabeled_x['covid19'].value_counts())

plt.figure(figsize=(8,4))
plt.title('Unlabeled Data Target Value Distribution')
sns.countplot(unlabeled_x['covid19'])
plt.xticks([0,1],['Negative', 'Positive'])
plt.show()
```

    0    1251
    1     616
    Name: covid19, dtype: int64
    



```python
# Unlabeled Data Split
unlabeled_y = unlabeled_x['covid19']
unlabeled_x = unlabeled_x.drop(['covid19'], axis = 1)

unlabeled_x.shape, unlabeled_y.shape
```




    ((1867, 38), (1867,))




```python
# Total DataSet Definition
all_x = pd.concat([train_x, unlabeled_x], axis=0)
all_y = pd.concat([train_y, unlabeled_y], axis=0)

all_x.shape, all_y.shape
```




    ((5672, 38), (5672,))




```python
# Total Data value Distribution
print(all_y.value_counts())

plt.figure(figsize=(8,4))
plt.title('Total Data Target Value Distribution')
sns.countplot(all_y)
plt.xticks([0,1],['Negative', 'Positive'])
plt.show()
```

    0    4750
    1     922
    Name: covid19, dtype: int64


## 4. ML Modeling


```python
# MLPClassifier
model = MLPClassifier(activation = 'relu', solver = 'adam', random_state=42)
```


```python
# Training
start_time = time.time()
model.fit(all_x, all_y)
end_time = time.time()
```


```python
# Training Time Measurements
duration = end_time - start_time
result = dt.timedelta(seconds=duration)
print('Training Time :',str(result).split('.')[0])
```

    Training Time : 0:00:03
    


```python
# Parameter coefficients 
print("Neural Network Parameter's format",model.coefs_[0].shape, model.coefs_[1].shape)
```

    Neural Network Parameter's format (38, 100) (100, 1)
    


```python
# Get MLPClassifier Format
get_mlp_format(model)
```

    n_features : 38
    n_iter : 100
    n_layers : 3
    n_outputs : 1
    out_activation : logistic
    


```python
plt.title('Loss Learning Curve')
plt.plot(model.loss_curve_ )
plt.show()
```



## 5. Performance Evaluation


```python
# Test Data Prediction
y_pred = model.predict(X_valid)
result = y_valid.to_frame().rename(columns={'covid19':'True'})
result['Pred'] = y_pred
print('result.shape :',result.shape)
result.head()
```

    result.shape : (761, 2)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>True</th>
      <th>Pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3497</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2167</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2466</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3744</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>223</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Test Data Prediction Confusion Matrix
plt.figure(figsize=(8,8))
plt.title('MLPClassifier Test Data Performance',y=1.02)
sns.heatmap(confusion_matrix(result['True'],result['Pred'],normalize='all'), cmap='PuBu', annot=True, fmt='.4f')
plt.xticks([0.5,1.5],['True','False'])
plt.yticks([0.5,1.5],['True','False'])
plt.show()
```



```python
# Get Clf Metrics
_ = get_clf_eval(result['True'],result['Pred'])
```

    [[677  23]
     [ 44  17]]
    정확도 :0.9120, 정밀도 :0.4250, 재현율 :0.2787, F1 :0.3366
    


```python
# Test Data Prediction Probability
y_pred_proba = model.predict_proba(X_valid)
y_pred_proba_1 = y_pred_proba[:,1]
```


```python
# ROC CURVE
print('Roc Auc Score :',np.round(roc_auc_score(y_valid,y_pred_proba_1),4))
roc_curve_plot(y_valid, y_pred_proba_1)
```

    Roc Auc Score : 0.7864
    




```python
# Precision Recall Curve
print('Precision :',np.round(precision_score(y_valid,y_pred),4))
print('Recall :',np.round(recall_score(y_valid,y_pred),4))
precision_recall_curve_plot(y_valid,y_pred_proba_1)
```

    Precision : 0.425
    Recall : 0.2787
    




## 6. Post Tuning


```python
# Post Tuning
post_result = mlp_post_tuning(y_pred_proba_1,0.01,0.8)
```

    [[241 459]
     [  7  54]]
    정확도 :0.3876, 정밀도 :0.1053, 재현율 :0.8852, F1 :0.1882
    [[366 334]
     [ 10  51]]
    정확도 :0.5480, 정밀도 :0.1325, 재현율 :0.8361, F1 :0.2287
    [[422 278]
     [ 14  47]]
    정확도 :0.6163, 정밀도 :0.1446, 재현율 :0.7705, F1 :0.2435
    [[473 227]
     [ 15  46]]
    정확도 :0.6820, 정밀도 :0.1685, 재현율 :0.7541, F1 :0.2754
    [[506 194]
     [ 17  44]]
    정확도 :0.7227, 정밀도 :0.1849, 재현율 :0.7213, F1 :0.2943
    [[530 170]
     [ 19  42]]
    정확도 :0.7516, 정밀도 :0.1981, 재현율 :0.6885, F1 :0.3077
    [[543 157]
     [ 19  42]]
    정확도 :0.7687, 정밀도 :0.2111, 재현율 :0.6885, F1 :0.3231
    [[564 136]
     [ 21  40]]
    정확도 :0.7937, 정밀도 :0.2273, 재현율 :0.6557, F1 :0.3376
    [[577 123]
     [ 21  40]]
    정확도 :0.8108, 정밀도 :0.2454, 재현율 :0.6557, F1 :0.3571
    [[589 111]
     [ 24  37]]
    정확도 :0.8226, 정밀도 :0.2500, 재현율 :0.6066, F1 :0.3541
    [[597 103]
     [ 24  37]]
    정확도 :0.8331, 정밀도 :0.2643, 재현율 :0.6066, F1 :0.3682
    [[603  97]
     [ 25  36]]
    정확도 :0.8397, 정밀도 :0.2707, 재현율 :0.5902, F1 :0.3711
    [[609  91]
     [ 26  35]]
    정확도 :0.8463, 정밀도 :0.2778, 재현율 :0.5738, F1 :0.3743
    [[614  86]
     [ 28  33]]
    정확도 :0.8502, 정밀도 :0.2773, 재현율 :0.5410, F1 :0.3667
    [[622  78]
     [ 29  32]]
    정확도 :0.8594, 정밀도 :0.2909, 재현율 :0.5246, F1 :0.3743
    [[626  74]
     [ 29  32]]
    정확도 :0.8647, 정밀도 :0.3019, 재현율 :0.5246, F1 :0.3832
    [[627  73]
     [ 29  32]]
    정확도 :0.8660, 정밀도 :0.3048, 재현율 :0.5246, F1 :0.3855
    [[634  66]
     [ 29  32]]
    정확도 :0.8752, 정밀도 :0.3265, 재현율 :0.5246, F1 :0.4025
    [[635  65]
     [ 29  32]]
    정확도 :0.8765, 정밀도 :0.3299, 재현율 :0.5246, F1 :0.4051
    [[637  63]
     [ 29  32]]
    정확도 :0.8791, 정밀도 :0.3368, 재현율 :0.5246, F1 :0.4103
    [[641  59]
     [ 29  32]]
    정확도 :0.8844, 정밀도 :0.3516, 재현율 :0.5246, F1 :0.4211
    [[643  57]
     [ 29  32]]
    정확도 :0.8870, 정밀도 :0.3596, 재현율 :0.5246, F1 :0.4267
    [[648  52]
     [ 30  31]]
    정확도 :0.8922, 정밀도 :0.3735, 재현율 :0.5082, F1 :0.4306
    [[648  52]
     [ 30  31]]
    정확도 :0.8922, 정밀도 :0.3735, 재현율 :0.5082, F1 :0.4306
    [[651  49]
     [ 31  30]]
    정확도 :0.8949, 정밀도 :0.3797, 재현율 :0.4918, F1 :0.4286
    [[652  48]
     [ 31  30]]
    정확도 :0.8962, 정밀도 :0.3846, 재현율 :0.4918, F1 :0.4317
    [[653  47]
     [ 31  30]]
    정확도 :0.8975, 정밀도 :0.3896, 재현율 :0.4918, F1 :0.4348
    [[656  44]
     [ 31  30]]
    정확도 :0.9014, 정밀도 :0.4054, 재현율 :0.4918, F1 :0.4444
    [[656  44]
     [ 31  30]]
    정확도 :0.9014, 정밀도 :0.4054, 재현율 :0.4918, F1 :0.4444
    [[656  44]
     [ 31  30]]
    정확도 :0.9014, 정밀도 :0.4054, 재현율 :0.4918, F1 :0.4444
    [[656  44]
     [ 32  29]]
    정확도 :0.9001, 정밀도 :0.3973, 재현율 :0.4754, F1 :0.4328
    [[658  42]
     [ 34  27]]
    정확도 :0.9001, 정밀도 :0.3913, 재현율 :0.4426, F1 :0.4154
    [[662  38]
     [ 35  26]]
    정확도 :0.9041, 정밀도 :0.4062, 재현율 :0.4262, F1 :0.4160
    [[663  37]
     [ 36  25]]
    정확도 :0.9041, 정밀도 :0.4032, 재현율 :0.4098, F1 :0.4065
    [[664  36]
     [ 36  25]]
    정확도 :0.9054, 정밀도 :0.4098, 재현율 :0.4098, F1 :0.4098
    [[666  34]
     [ 36  25]]
    정확도 :0.9080, 정밀도 :0.4237, 재현율 :0.4098, F1 :0.4167
    [[667  33]
     [ 36  25]]
    정확도 :0.9093, 정밀도 :0.4310, 재현율 :0.4098, F1 :0.4202
    [[668  32]
     [ 37  24]]
    정확도 :0.9093, 정밀도 :0.4286, 재현율 :0.3934, F1 :0.4103
    [[670  30]
     [ 38  23]]
    정확도 :0.9106, 정밀도 :0.4340, 재현율 :0.3770, F1 :0.4035
    [[672  28]
     [ 38  23]]
    정확도 :0.9133, 정밀도 :0.4510, 재현율 :0.3770, F1 :0.4107
    [[672  28]
     [ 38  23]]
    정확도 :0.9133, 정밀도 :0.4510, 재현율 :0.3770, F1 :0.4107
    [[673  27]
     [ 39  22]]
    정확도 :0.9133, 정밀도 :0.4490, 재현율 :0.3607, F1 :0.4000
    [[673  27]
     [ 41  20]]
    정확도 :0.9106, 정밀도 :0.4255, 재현율 :0.3279, F1 :0.3704
    [[673  27]
     [ 42  19]]
    정확도 :0.9093, 정밀도 :0.4130, 재현율 :0.3115, F1 :0.3551
    [[674  26]
     [ 42  19]]
    정확도 :0.9106, 정밀도 :0.4222, 재현율 :0.3115, F1 :0.3585
    [[674  26]
     [ 42  19]]
    정확도 :0.9106, 정밀도 :0.4222, 재현율 :0.3115, F1 :0.3585
    [[674  26]
     [ 42  19]]
    정확도 :0.9106, 정밀도 :0.4222, 재현율 :0.3115, F1 :0.3585
    [[675  25]
     [ 42  19]]
    정확도 :0.9120, 정밀도 :0.4318, 재현율 :0.3115, F1 :0.3619
    [[676  24]
     [ 43  18]]
    정확도 :0.9120, 정밀도 :0.4286, 재현율 :0.2951, F1 :0.3495
    [[677  23]
     [ 44  17]]
    정확도 :0.9120, 정밀도 :0.4250, 재현율 :0.2787, F1 :0.3366
    [[678  22]
     [ 44  17]]
    정확도 :0.9133, 정밀도 :0.4359, 재현율 :0.2787, F1 :0.3400
    [[679  21]
     [ 44  17]]
    정확도 :0.9146, 정밀도 :0.4474, 재현율 :0.2787, F1 :0.3434
    [[682  18]
     [ 44  17]]
    정확도 :0.9185, 정밀도 :0.4857, 재현율 :0.2787, F1 :0.3542
    [[682  18]
     [ 44  17]]
    정확도 :0.9185, 정밀도 :0.4857, 재현율 :0.2787, F1 :0.3542
    [[683  17]
     [ 44  17]]
    정확도 :0.9198, 정밀도 :0.5000, 재현율 :0.2787, F1 :0.3579
    [[683  17]
     [ 45  16]]
    정확도 :0.9185, 정밀도 :0.4848, 재현율 :0.2623, F1 :0.3404
    [[683  17]
     [ 46  15]]
    정확도 :0.9172, 정밀도 :0.4688, 재현율 :0.2459, F1 :0.3226
    [[683  17]
     [ 46  15]]
    정확도 :0.9172, 정밀도 :0.4688, 재현율 :0.2459, F1 :0.3226
    [[684  16]
     [ 47  14]]
    정확도 :0.9172, 정밀도 :0.4667, 재현율 :0.2295, F1 :0.3077
    [[684  16]
     [ 47  14]]
    정확도 :0.9172, 정밀도 :0.4667, 재현율 :0.2295, F1 :0.3077
    [[684  16]
     [ 47  14]]
    정확도 :0.9172, 정밀도 :0.4667, 재현율 :0.2295, F1 :0.3077
    [[687  13]
     [ 47  14]]
    정확도 :0.9212, 정밀도 :0.5185, 재현율 :0.2295, F1 :0.3182
    [[687  13]
     [ 47  14]]
    정확도 :0.9212, 정밀도 :0.5185, 재현율 :0.2295, F1 :0.3182
    [[688  12]
     [ 47  14]]
    정확도 :0.9225, 정밀도 :0.5385, 재현율 :0.2295, F1 :0.3218
    [[689  11]
     [ 47  14]]
    정확도 :0.9238, 정밀도 :0.5600, 재현율 :0.2295, F1 :0.3256
    [[689  11]
     [ 47  14]]
    정확도 :0.9238, 정밀도 :0.5600, 재현율 :0.2295, F1 :0.3256
    [[689  11]
     [ 48  13]]
    정확도 :0.9225, 정밀도 :0.5417, 재현율 :0.2131, F1 :0.3059
    [[689  11]
     [ 48  13]]
    정확도 :0.9225, 정밀도 :0.5417, 재현율 :0.2131, F1 :0.3059
    [[690  10]
     [ 48  13]]
    정확도 :0.9238, 정밀도 :0.5652, 재현율 :0.2131, F1 :0.3095
    [[690  10]
     [ 48  13]]
    정확도 :0.9238, 정밀도 :0.5652, 재현율 :0.2131, F1 :0.3095
    [[693   7]
     [ 48  13]]
    정확도 :0.9277, 정밀도 :0.6500, 재현율 :0.2131, F1 :0.3210
    [[693   7]
     [ 48  13]]
    정확도 :0.9277, 정밀도 :0.6500, 재현율 :0.2131, F1 :0.3210
    [[694   6]
     [ 48  13]]
    정확도 :0.9290, 정밀도 :0.6842, 재현율 :0.2131, F1 :0.3250
    [[694   6]
     [ 48  13]]
    정확도 :0.9290, 정밀도 :0.6842, 재현율 :0.2131, F1 :0.3250
    [[694   6]
     [ 49  12]]
    정확도 :0.9277, 정밀도 :0.6667, 재현율 :0.1967, F1 :0.3038
    [[694   6]
     [ 49  12]]
    정확도 :0.9277, 정밀도 :0.6667, 재현율 :0.1967, F1 :0.3038
    [[694   6]
     [ 50  11]]
    정확도 :0.9264, 정밀도 :0.6471, 재현율 :0.1803, F1 :0.2821
    [[694   6]
     [ 50  11]]
    정확도 :0.9264, 정밀도 :0.6471, 재현율 :0.1803, F1 :0.2821
    [[694   6]
     [ 50  11]]
    정확도 :0.9264, 정밀도 :0.6471, 재현율 :0.1803, F1 :0.2821
    


```python
# Post Tuning Result
post_result.sort_values(by='f1', ascending=False).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>proba</th>
      <th>acc</th>
      <th>pre</th>
      <th>rec</th>
      <th>f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0.30</td>
      <td>0.901445</td>
      <td>0.405405</td>
      <td>0.491803</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.29</td>
      <td>0.901445</td>
      <td>0.405405</td>
      <td>0.491803</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.28</td>
      <td>0.901445</td>
      <td>0.405405</td>
      <td>0.491803</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.27</td>
      <td>0.897503</td>
      <td>0.38961</td>
      <td>0.491803</td>
      <td>0.434783</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.31</td>
      <td>0.900131</td>
      <td>0.39726</td>
      <td>0.47541</td>
      <td>0.432836</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Post Tuning Result Visualization
plt.figure(figsize=(10,6))
plt.plot(post_result['proba'], post_result['f1'], label='f1_score')
plt.title('Post Tuning Result')
plt.axvline(x=0.3, label='Best Prob', linestyle=':', alpha=0.8, color='red')
plt.xlabel('Probability Threshold')
plt.ylabel('f1_score')
plt.legend()
plt.show()
```



## 7. Test Submission 


```python
# Sample Submission DataFrame
submit = pd.read_csv('data/sample_submission.csv')
print('submit :',submit.shape)
submit.head()
```

    submit : (5732, 2)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>covid19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3806</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3807</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3808</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3809</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3810</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Submission Data Prediction
y_sub_pred = model.predict_proba(test_x)[:,1]
sub_proba = 0.3
sub_prediction = np.where(y_sub_pred < sub_proba, 0, 1)
```


```python
# Final Submission DataFrame
submit['covid19'] = sub_prediction
print('sub_result.shape :',submit.shape)
submit.head()
```

    sub_result.shape : (5732, 2)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>covid19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3806</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3807</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3808</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3809</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3810</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Submission Export
file = 'data/submission/submit_0823.csv'
submit.to_csv(file,index=False)
```


```python
# Submission Read
pd.read_csv(file).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>covid19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3806</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3807</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3808</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3809</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3810</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Simulation

### Model Save & Load


```python
# Model path Definition
filename = 'model_save/mlp_first.sav'
```


```python
# # MLP Model Save
# pickle.dump(model, open(filename, 'wb'))
```


```python
# MLP Model Load
model = pickle.load(open(filename, 'rb'))
```

### Positive Sample Simulation


```python
# Single Sample
# sample_idx = np.random.choice(np.arange(1,train_df.shape[0]),1)[0]
sample_idx = 16
sample_path = 'data/train/' + str(sample_idx).zfill(5) + '.wav'
print(sample_path)
y,sr = librosa.load(sample_path, sr = CFG['SR'])
```

    data/train/00016.wav
    


```python
# Object Profile
pd.DataFrame(train_x.iloc[sample_idx-1,:]).T[['age','respiratory_condition','fever_or_muscle_pain','female','male','other']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
      <th>female</th>
      <th>male</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# WAV Sample Display
wav_display(y, sr)
```






```python
# Sample WAV Prediction Result
sample_proba = 0.3
sample_data = np.array(train_x.iloc[sample_idx-1,:]).reshape(1,-1)
sample_pred = model.predict_proba(sample_data)[:,1][0]
sample_result = np.where(sample_pred < sample_proba, 0, 1)
sample_return = np.array(['Negative','Positive'])[sample_result]
print('sample_return :',sample_return)
```

    sample_return : Positive
    

### Negative Sample Simulation


```python
# Single Sample
# sample_idx = np.random.choice(np.arange(1,train_df.shape[0]),1)[0]
sample_idx = 55
sample_path = 'data/train/' + str(sample_idx).zfill(5) + '.wav'
print(sample_path)
y,sr = librosa.load(sample_path, sr = CFG['SR'])
```

    data/train/00055.wav
    


```python
# Object Profile
pd.DataFrame(train_x.iloc[sample_idx-1,:]).T[['age','respiratory_condition','fever_or_muscle_pain','female','male','other']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
      <th>female</th>
      <th>male</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>53.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# WAV Sample Display
wav_display(y, sr)
```





```python
# Sample WAV Prediction Result
sample_proba = 0.3
sample_data = np.array(train_x.iloc[sample_idx-1,:]).reshape(1,-1)
sample_pred = model.predict_proba(sample_data)[:,1][0]
sample_result = np.where(sample_pred < sample_proba, 0, 1)
sample_return = np.array(['Negative','Positive'])[sample_result]
print('sample_return :',sample_return)
```

    sample_return : Negative
   
