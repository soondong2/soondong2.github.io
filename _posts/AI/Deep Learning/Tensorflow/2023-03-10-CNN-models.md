---
title: "[Tensorflow] CNN 모델의 발전"
date: 2023-03-10

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - Tensorflow
---

## CNN 모델의 발전

* 1998: LeNet – Gradient-based Learning Applied to Document Recognition
* 2012: AlexNet – ImageNet Classification with Deep Convolutional Neural Network
* **2014: VggNet – Very Deep Convolutional Networks for Large-Scale Image Recognition**
* **2014: GooLeNet – Going Deeper with Convolutions**
* 2014: SppNet – Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
* **2015: ResNet – Deep Residual Learning for Image Recognition**
* **2016: Xception – Xception: Deep Learning with Depthwise Separable Convolutions**
* **2017: MobileNet – MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Application**
* **2017: DenseNet – Densely Connected Convolutional Networks**
* 2017: SeNet – Squeeze and Excitation Networks
* 2017: ShuffleNet – ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
* **2018: NasNet – Learning Transferable Architectures for Scalable Image Recognition**
* 2018: Bag of Tricks – Bag of Tricks for Image Classification with Convolutional Neural Networks
* **2019: EfficientNet – EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**


## VGGNet(Visual Geometry Group Net)

- 2014년 ILSVRC에서 2등 차지 (상위-5 오류율: 7.32%), 이 후의 수많은 연구에 영향을 미침

- 특징
  - 활성화 함수로 `ReLU` 사용, Dropout 적용
  - 합성곱과 풀링 계층으로 구성된 블록과 분류를 위한 완전 연결계층으로 결합된 전형적인 구조
  - 이미지 변환, 좌우 반전 등의 변환을 시도하여 인위적으로 데이터셋을 늘림
  - 몇 개의 합성곱 계층과 최대-풀링 계층이 따르는 5개의 블록과, 3개의 완전연결계층(학습 시, 드롭아웃 사용)으로 구성
  - 모든 합성곱과 최대-풀링 계층에 `padding='SAME'` 적용
  - 합성곱 계층에는 `stride=1`, 활성화 함수로 `ReLU` 사용
  - 특징 맵 깊이를 증가시킴
  - 척도 변경을 통한 데이터 보강(Data Augmentation)

- 기여

  - 3x3 커널을 갖는 두 합성곱 계층을 쌓은 스택이 5x5 커널을 갖는 하나의 합성곱 계층과 동일한 수용영역(ERF)을 가짐
  - 11x11 사이즈의 필터 크기를 가지는 AlexNet과 비교하여, 더 작은 합성곱 계층을 더 많이 포함해 더 큰 ERF를 얻음
  - 합성곱 계층의 개수가 많아지면, 매개변수 개수를 줄이고, 비선형성을 증가시킴

- VGG-16 모델(16개 층)
- VGG-19 모델(19개 층)  
- ImageNet에서 훈련이 끝난 후 얻게된 매개변수 값 로딩
- 네트워크를 다시 처음부터 학습하고자 한다면 `weights=None`으로 설정, 케라스에서 무작위로 가중치를 설정함
- `include_top=False`: VGG의 밀집 계층을 제외한다는 뜻
- 해당 네트워크의 출력은 합성곱/최대-풀링 블록의 특징맵이 됨
- `pooling`: 특징맵을 반환하기 전에 적용할 선택적인 연산을 지정


```python
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
```


```python
# weight은 imagenet에서 사용한 가중치를 사용
# 실제 가중치가 적용된 model을 다운로드

vggnet= VGG19(include_top=True, weights='imagenet',
              input_tensor=None, input_shape=None,
              pooling=None, classes=1000)
vggnet.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    574710816/574710816 [==============================] - 36s 0us/step
    Model: "vgg19"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv4 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv4 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv4 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     flatten (Flatten)           (None, 25088)             0         
                                                                     
     fc1 (Dense)                 (None, 4096)              102764544 
                                                                     
     fc2 (Dense)                 (None, 4096)              16781312  
                                                                     
     predictions (Dense)         (None, 1000)              4097000   
                                                                     
    =================================================================
    Total params: 143,667,240
    Trainable params: 143,667,240
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# 웹에 있는 사진 다운
!wget -O dog.jpg https://www.publicdomainpictures.net/pictures/250000/nahled/dog-beagle-portrait.jpg
```

    --2023-03-09 14:20:19--  https://www.publicdomainpictures.net/pictures/250000/nahled/dog-beagle-portrait.jpg
    Resolving www.publicdomainpictures.net (www.publicdomainpictures.net)... 104.20.44.162, 104.20.45.162, 172.67.2.204, ...
    Connecting to www.publicdomainpictures.net (www.publicdomainpictures.net)|104.20.44.162|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 94395 (92K) [image/jpeg]
    Saving to: ‘dog.jpg’
    
    dog.jpg             100%[===================>]  92.18K  --.-KB/s    in 0.003s  
    
    2023-03-09 14:20:20 (29.1 MB/s) - ‘dog.jpg’ saved [94395/94395]
    
    


```python
# 다운로드한 dog.jpg를 terget_size로 줄여줌
# dog의 input은 [(None, 224, 224, 3)] 크기를 가짐
img = image.load_img('dog.jpg', target_size=(224, 224))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = vggnet.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    1/1 [==============================] - 0s 28ms/step
    [[('n02088364', 'beagle', 0.83961403), ('n02089973', 'English_foxhound', 0.08818262), ('n02089867', 'Walker_hound', 0.062327106), ('n02088238', 'basset', 0.004562265), ('n02088632', 'bluetick', 0.0033385658)]]
    


 

## GoogLeNet, Inception

- VGGNet을 제치고 같은 해 분류 과제에서 1등을 차지
- 인셉션 블록이라는 개념을 도입하여, **인셉션 네트워크(Inception Network)**라고도 불림

  <img src="https://miro.medium.com/max/2800/0*rbWRzjKvoGt9W3Mf.png">

- 특징 
  - CNN 계산 용량을 최적화하는 것을 고려
  - 전형적인 합성곱, 풀링 계층으로 시작하고, 이 정보는 9개의 인셉션 모듈 스택을 통과 (해당 모듈을 하위 네트워크라고도 함)
  - 각 모듈에서 입력 특징 맵은 서로 다른 계층으로 구성된 4개의 병렬 하위 블록에 전달되고, 이를 서로 다시 연결
  - 모든 합성곱과 풀링 계층의 `padding`옵션은 `'SAME'`이며 `stride=1` 활성화 함수는 `ReLU` 사용

- 기여
  - 규모가 큰 블록과 병목을 보편화
  - 병목 계층으로 1x1 합성곱 계층 사용
  - 완전 연결 계층 대신 풀링 계층 사용
  - 중간 소실로 경사 소실 문제 해결

  <img src="https://norman3.github.io/papers/images/google_inception/f01.png">


```python
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
```


```python
# include_top=True : top 부분 가져오기
# weights='imagenet' : imgnet의 가중치 그대로 사용
inception = InceptionV3(include_top=True, weights='imagenet',
                        input_tensor=None, input_shape=None,
                        pooling=None, classes=1000)
inception.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
    96112376/96112376 [==============================] - 5s 0us/step
    Model: "inception_v3"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_2 (InputLayer)           [(None, 299, 299, 3  0           []                               
                                    )]                                                                
                                                                                                      
     conv2d (Conv2D)                (None, 149, 149, 32  864         ['input_2[0][0]']                
                                    )                                                                 
                                                                                                      
     batch_normalization (BatchNorm  (None, 149, 149, 32  96         ['conv2d[0][0]']                 
     alization)                     )                                                                 
                                                                                                      
     activation (Activation)        (None, 149, 149, 32  0           ['batch_normalization[0][0]']    
                                    )                                                                 
                                                                                                      
     conv2d_1 (Conv2D)              (None, 147, 147, 32  9216        ['activation[0][0]']             
                                    )                                                                 
                                                                                                      
     batch_normalization_1 (BatchNo  (None, 147, 147, 32  96         ['conv2d_1[0][0]']               
     rmalization)                   )                                                                 
                                                                                                      
     activation_1 (Activation)      (None, 147, 147, 32  0           ['batch_normalization_1[0][0]']  
                                    )                                                                 
                                                                                                      
     conv2d_2 (Conv2D)              (None, 147, 147, 64  18432       ['activation_1[0][0]']           
                                    )                                                                 
                                                                                                      
     batch_normalization_2 (BatchNo  (None, 147, 147, 64  192        ['conv2d_2[0][0]']               
     rmalization)                   )                                                                 
                                                                                                      
     activation_2 (Activation)      (None, 147, 147, 64  0           ['batch_normalization_2[0][0]']  
                                    )                                                                 
                                                                                                      
     max_pooling2d (MaxPooling2D)   (None, 73, 73, 64)   0           ['activation_2[0][0]']           
                                                                                                      
     conv2d_3 (Conv2D)              (None, 73, 73, 80)   5120        ['max_pooling2d[0][0]']          
                                                                                                      
     batch_normalization_3 (BatchNo  (None, 73, 73, 80)  240         ['conv2d_3[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     activation_3 (Activation)      (None, 73, 73, 80)   0           ['batch_normalization_3[0][0]']  
                                                                                                      
     conv2d_4 (Conv2D)              (None, 71, 71, 192)  138240      ['activation_3[0][0]']           
                                                                                                      
     batch_normalization_4 (BatchNo  (None, 71, 71, 192)  576        ['conv2d_4[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     activation_4 (Activation)      (None, 71, 71, 192)  0           ['batch_normalization_4[0][0]']  
                                                                                                      
     max_pooling2d_1 (MaxPooling2D)  (None, 35, 35, 192)  0          ['activation_4[0][0]']           
                                                                                                      
     conv2d_8 (Conv2D)              (None, 35, 35, 64)   12288       ['max_pooling2d_1[0][0]']        
                                                                                                      
     batch_normalization_8 (BatchNo  (None, 35, 35, 64)  192         ['conv2d_8[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     activation_8 (Activation)      (None, 35, 35, 64)   0           ['batch_normalization_8[0][0]']  
                                                                                                      
     conv2d_6 (Conv2D)              (None, 35, 35, 48)   9216        ['max_pooling2d_1[0][0]']        
                                                                                                      
     conv2d_9 (Conv2D)              (None, 35, 35, 96)   55296       ['activation_8[0][0]']           
                                                                                                      
     batch_normalization_6 (BatchNo  (None, 35, 35, 48)  144         ['conv2d_6[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     batch_normalization_9 (BatchNo  (None, 35, 35, 96)  288         ['conv2d_9[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     activation_6 (Activation)      (None, 35, 35, 48)   0           ['batch_normalization_6[0][0]']  
                                                                                                      
     activation_9 (Activation)      (None, 35, 35, 96)   0           ['batch_normalization_9[0][0]']  
                                                                                                      
     average_pooling2d (AveragePool  (None, 35, 35, 192)  0          ['max_pooling2d_1[0][0]']        
     ing2D)                                                                                           
                                                                                                      
     conv2d_5 (Conv2D)              (None, 35, 35, 64)   12288       ['max_pooling2d_1[0][0]']        
                                                                                                      
     conv2d_7 (Conv2D)              (None, 35, 35, 64)   76800       ['activation_6[0][0]']           
                                                                                                      
     conv2d_10 (Conv2D)             (None, 35, 35, 96)   82944       ['activation_9[0][0]']           
                                                                                                      
     conv2d_11 (Conv2D)             (None, 35, 35, 32)   6144        ['average_pooling2d[0][0]']      
                                                                                                      
     batch_normalization_5 (BatchNo  (None, 35, 35, 64)  192         ['conv2d_5[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     batch_normalization_7 (BatchNo  (None, 35, 35, 64)  192         ['conv2d_7[0][0]']               
     rmalization)                                                                                     
                                                                                                      
     batch_normalization_10 (BatchN  (None, 35, 35, 96)  288         ['conv2d_10[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_11 (BatchN  (None, 35, 35, 32)  96          ['conv2d_11[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_5 (Activation)      (None, 35, 35, 64)   0           ['batch_normalization_5[0][0]']  
                                                                                                      
     activation_7 (Activation)      (None, 35, 35, 64)   0           ['batch_normalization_7[0][0]']  
                                                                                                      
     activation_10 (Activation)     (None, 35, 35, 96)   0           ['batch_normalization_10[0][0]'] 
                                                                                                      
     activation_11 (Activation)     (None, 35, 35, 32)   0           ['batch_normalization_11[0][0]'] 
                                                                                                      
     mixed0 (Concatenate)           (None, 35, 35, 256)  0           ['activation_5[0][0]',           
                                                                      'activation_7[0][0]',           
                                                                      'activation_10[0][0]',          
                                                                      'activation_11[0][0]']          
                                                                                                      
     conv2d_15 (Conv2D)             (None, 35, 35, 64)   16384       ['mixed0[0][0]']                 
                                                                                                      
     batch_normalization_15 (BatchN  (None, 35, 35, 64)  192         ['conv2d_15[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_15 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_15[0][0]'] 
                                                                                                      
     conv2d_13 (Conv2D)             (None, 35, 35, 48)   12288       ['mixed0[0][0]']                 
                                                                                                      
     conv2d_16 (Conv2D)             (None, 35, 35, 96)   55296       ['activation_15[0][0]']          
                                                                                                      
     batch_normalization_13 (BatchN  (None, 35, 35, 48)  144         ['conv2d_13[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_16 (BatchN  (None, 35, 35, 96)  288         ['conv2d_16[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_13 (Activation)     (None, 35, 35, 48)   0           ['batch_normalization_13[0][0]'] 
                                                                                                      
     activation_16 (Activation)     (None, 35, 35, 96)   0           ['batch_normalization_16[0][0]'] 
                                                                                                      
     average_pooling2d_1 (AveragePo  (None, 35, 35, 256)  0          ['mixed0[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_12 (Conv2D)             (None, 35, 35, 64)   16384       ['mixed0[0][0]']                 
                                                                                                      
     conv2d_14 (Conv2D)             (None, 35, 35, 64)   76800       ['activation_13[0][0]']          
                                                                                                      
     conv2d_17 (Conv2D)             (None, 35, 35, 96)   82944       ['activation_16[0][0]']          
                                                                                                      
     conv2d_18 (Conv2D)             (None, 35, 35, 64)   16384       ['average_pooling2d_1[0][0]']    
                                                                                                      
     batch_normalization_12 (BatchN  (None, 35, 35, 64)  192         ['conv2d_12[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_14 (BatchN  (None, 35, 35, 64)  192         ['conv2d_14[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_17 (BatchN  (None, 35, 35, 96)  288         ['conv2d_17[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_18 (BatchN  (None, 35, 35, 64)  192         ['conv2d_18[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_12 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_12[0][0]'] 
                                                                                                      
     activation_14 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_14[0][0]'] 
                                                                                                      
     activation_17 (Activation)     (None, 35, 35, 96)   0           ['batch_normalization_17[0][0]'] 
                                                                                                      
     activation_18 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_18[0][0]'] 
                                                                                                      
     mixed1 (Concatenate)           (None, 35, 35, 288)  0           ['activation_12[0][0]',          
                                                                      'activation_14[0][0]',          
                                                                      'activation_17[0][0]',          
                                                                      'activation_18[0][0]']          
                                                                                                      
     conv2d_22 (Conv2D)             (None, 35, 35, 64)   18432       ['mixed1[0][0]']                 
                                                                                                      
     batch_normalization_22 (BatchN  (None, 35, 35, 64)  192         ['conv2d_22[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_22 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_22[0][0]'] 
                                                                                                      
     conv2d_20 (Conv2D)             (None, 35, 35, 48)   13824       ['mixed1[0][0]']                 
                                                                                                      
     conv2d_23 (Conv2D)             (None, 35, 35, 96)   55296       ['activation_22[0][0]']          
                                                                                                      
     batch_normalization_20 (BatchN  (None, 35, 35, 48)  144         ['conv2d_20[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_23 (BatchN  (None, 35, 35, 96)  288         ['conv2d_23[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_20 (Activation)     (None, 35, 35, 48)   0           ['batch_normalization_20[0][0]'] 
                                                                                                      
     activation_23 (Activation)     (None, 35, 35, 96)   0           ['batch_normalization_23[0][0]'] 
                                                                                                      
     average_pooling2d_2 (AveragePo  (None, 35, 35, 288)  0          ['mixed1[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_19 (Conv2D)             (None, 35, 35, 64)   18432       ['mixed1[0][0]']                 
                                                                                                      
     conv2d_21 (Conv2D)             (None, 35, 35, 64)   76800       ['activation_20[0][0]']          
                                                                                                      
     conv2d_24 (Conv2D)             (None, 35, 35, 96)   82944       ['activation_23[0][0]']          
                                                                                                      
     conv2d_25 (Conv2D)             (None, 35, 35, 64)   18432       ['average_pooling2d_2[0][0]']    
                                                                                                      
     batch_normalization_19 (BatchN  (None, 35, 35, 64)  192         ['conv2d_19[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_21 (BatchN  (None, 35, 35, 64)  192         ['conv2d_21[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_24 (BatchN  (None, 35, 35, 96)  288         ['conv2d_24[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_25 (BatchN  (None, 35, 35, 64)  192         ['conv2d_25[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_19 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_19[0][0]'] 
                                                                                                      
     activation_21 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_21[0][0]'] 
                                                                                                      
     activation_24 (Activation)     (None, 35, 35, 96)   0           ['batch_normalization_24[0][0]'] 
                                                                                                      
     activation_25 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_25[0][0]'] 
                                                                                                      
     mixed2 (Concatenate)           (None, 35, 35, 288)  0           ['activation_19[0][0]',          
                                                                      'activation_21[0][0]',          
                                                                      'activation_24[0][0]',          
                                                                      'activation_25[0][0]']          
                                                                                                      
     conv2d_27 (Conv2D)             (None, 35, 35, 64)   18432       ['mixed2[0][0]']                 
                                                                                                      
     batch_normalization_27 (BatchN  (None, 35, 35, 64)  192         ['conv2d_27[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_27 (Activation)     (None, 35, 35, 64)   0           ['batch_normalization_27[0][0]'] 
                                                                                                      
     conv2d_28 (Conv2D)             (None, 35, 35, 96)   55296       ['activation_27[0][0]']          
                                                                                                      
     batch_normalization_28 (BatchN  (None, 35, 35, 96)  288         ['conv2d_28[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_28 (Activation)     (None, 35, 35, 96)   0           ['batch_normalization_28[0][0]'] 
                                                                                                      
     conv2d_26 (Conv2D)             (None, 17, 17, 384)  995328      ['mixed2[0][0]']                 
                                                                                                      
     conv2d_29 (Conv2D)             (None, 17, 17, 96)   82944       ['activation_28[0][0]']          
                                                                                                      
     batch_normalization_26 (BatchN  (None, 17, 17, 384)  1152       ['conv2d_26[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_29 (BatchN  (None, 17, 17, 96)  288         ['conv2d_29[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_26 (Activation)     (None, 17, 17, 384)  0           ['batch_normalization_26[0][0]'] 
                                                                                                      
     activation_29 (Activation)     (None, 17, 17, 96)   0           ['batch_normalization_29[0][0]'] 
                                                                                                      
     max_pooling2d_2 (MaxPooling2D)  (None, 17, 17, 288)  0          ['mixed2[0][0]']                 
                                                                                                      
     mixed3 (Concatenate)           (None, 17, 17, 768)  0           ['activation_26[0][0]',          
                                                                      'activation_29[0][0]',          
                                                                      'max_pooling2d_2[0][0]']        
                                                                                                      
     conv2d_34 (Conv2D)             (None, 17, 17, 128)  98304       ['mixed3[0][0]']                 
                                                                                                      
     batch_normalization_34 (BatchN  (None, 17, 17, 128)  384        ['conv2d_34[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_34 (Activation)     (None, 17, 17, 128)  0           ['batch_normalization_34[0][0]'] 
                                                                                                      
     conv2d_35 (Conv2D)             (None, 17, 17, 128)  114688      ['activation_34[0][0]']          
                                                                                                      
     batch_normalization_35 (BatchN  (None, 17, 17, 128)  384        ['conv2d_35[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_35 (Activation)     (None, 17, 17, 128)  0           ['batch_normalization_35[0][0]'] 
                                                                                                      
     conv2d_31 (Conv2D)             (None, 17, 17, 128)  98304       ['mixed3[0][0]']                 
                                                                                                      
     conv2d_36 (Conv2D)             (None, 17, 17, 128)  114688      ['activation_35[0][0]']          
                                                                                                      
     batch_normalization_31 (BatchN  (None, 17, 17, 128)  384        ['conv2d_31[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_36 (BatchN  (None, 17, 17, 128)  384        ['conv2d_36[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_31 (Activation)     (None, 17, 17, 128)  0           ['batch_normalization_31[0][0]'] 
                                                                                                      
     activation_36 (Activation)     (None, 17, 17, 128)  0           ['batch_normalization_36[0][0]'] 
                                                                                                      
     conv2d_32 (Conv2D)             (None, 17, 17, 128)  114688      ['activation_31[0][0]']          
                                                                                                      
     conv2d_37 (Conv2D)             (None, 17, 17, 128)  114688      ['activation_36[0][0]']          
                                                                                                      
     batch_normalization_32 (BatchN  (None, 17, 17, 128)  384        ['conv2d_32[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_37 (BatchN  (None, 17, 17, 128)  384        ['conv2d_37[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_32 (Activation)     (None, 17, 17, 128)  0           ['batch_normalization_32[0][0]'] 
                                                                                                      
     activation_37 (Activation)     (None, 17, 17, 128)  0           ['batch_normalization_37[0][0]'] 
                                                                                                      
     average_pooling2d_3 (AveragePo  (None, 17, 17, 768)  0          ['mixed3[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_30 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed3[0][0]']                 
                                                                                                      
     conv2d_33 (Conv2D)             (None, 17, 17, 192)  172032      ['activation_32[0][0]']          
                                                                                                      
     conv2d_38 (Conv2D)             (None, 17, 17, 192)  172032      ['activation_37[0][0]']          
                                                                                                      
     conv2d_39 (Conv2D)             (None, 17, 17, 192)  147456      ['average_pooling2d_3[0][0]']    
                                                                                                      
     batch_normalization_30 (BatchN  (None, 17, 17, 192)  576        ['conv2d_30[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_33 (BatchN  (None, 17, 17, 192)  576        ['conv2d_33[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_38 (BatchN  (None, 17, 17, 192)  576        ['conv2d_38[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_39 (BatchN  (None, 17, 17, 192)  576        ['conv2d_39[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_30 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_30[0][0]'] 
                                                                                                      
     activation_33 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_33[0][0]'] 
                                                                                                      
     activation_38 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_38[0][0]'] 
                                                                                                      
     activation_39 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_39[0][0]'] 
                                                                                                      
     mixed4 (Concatenate)           (None, 17, 17, 768)  0           ['activation_30[0][0]',          
                                                                      'activation_33[0][0]',          
                                                                      'activation_38[0][0]',          
                                                                      'activation_39[0][0]']          
                                                                                                      
     conv2d_44 (Conv2D)             (None, 17, 17, 160)  122880      ['mixed4[0][0]']                 
                                                                                                      
     batch_normalization_44 (BatchN  (None, 17, 17, 160)  480        ['conv2d_44[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_44 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_44[0][0]'] 
                                                                                                      
     conv2d_45 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_44[0][0]']          
                                                                                                      
     batch_normalization_45 (BatchN  (None, 17, 17, 160)  480        ['conv2d_45[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_45 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_45[0][0]'] 
                                                                                                      
     conv2d_41 (Conv2D)             (None, 17, 17, 160)  122880      ['mixed4[0][0]']                 
                                                                                                      
     conv2d_46 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_45[0][0]']          
                                                                                                      
     batch_normalization_41 (BatchN  (None, 17, 17, 160)  480        ['conv2d_41[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_46 (BatchN  (None, 17, 17, 160)  480        ['conv2d_46[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_41 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_41[0][0]'] 
                                                                                                      
     activation_46 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_46[0][0]'] 
                                                                                                      
     conv2d_42 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_41[0][0]']          
                                                                                                      
     conv2d_47 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_46[0][0]']          
                                                                                                      
     batch_normalization_42 (BatchN  (None, 17, 17, 160)  480        ['conv2d_42[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_47 (BatchN  (None, 17, 17, 160)  480        ['conv2d_47[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_42 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_42[0][0]'] 
                                                                                                      
     activation_47 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_47[0][0]'] 
                                                                                                      
     average_pooling2d_4 (AveragePo  (None, 17, 17, 768)  0          ['mixed4[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_40 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed4[0][0]']                 
                                                                                                      
     conv2d_43 (Conv2D)             (None, 17, 17, 192)  215040      ['activation_42[0][0]']          
                                                                                                      
     conv2d_48 (Conv2D)             (None, 17, 17, 192)  215040      ['activation_47[0][0]']          
                                                                                                      
     conv2d_49 (Conv2D)             (None, 17, 17, 192)  147456      ['average_pooling2d_4[0][0]']    
                                                                                                      
     batch_normalization_40 (BatchN  (None, 17, 17, 192)  576        ['conv2d_40[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_43 (BatchN  (None, 17, 17, 192)  576        ['conv2d_43[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_48 (BatchN  (None, 17, 17, 192)  576        ['conv2d_48[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_49 (BatchN  (None, 17, 17, 192)  576        ['conv2d_49[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_40 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_40[0][0]'] 
                                                                                                      
     activation_43 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_43[0][0]'] 
                                                                                                      
     activation_48 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_48[0][0]'] 
                                                                                                      
     activation_49 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_49[0][0]'] 
                                                                                                      
     mixed5 (Concatenate)           (None, 17, 17, 768)  0           ['activation_40[0][0]',          
                                                                      'activation_43[0][0]',          
                                                                      'activation_48[0][0]',          
                                                                      'activation_49[0][0]']          
                                                                                                      
     conv2d_54 (Conv2D)             (None, 17, 17, 160)  122880      ['mixed5[0][0]']                 
                                                                                                      
     batch_normalization_54 (BatchN  (None, 17, 17, 160)  480        ['conv2d_54[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_54 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_54[0][0]'] 
                                                                                                      
     conv2d_55 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_54[0][0]']          
                                                                                                      
     batch_normalization_55 (BatchN  (None, 17, 17, 160)  480        ['conv2d_55[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_55 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_55[0][0]'] 
                                                                                                      
     conv2d_51 (Conv2D)             (None, 17, 17, 160)  122880      ['mixed5[0][0]']                 
                                                                                                      
     conv2d_56 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_55[0][0]']          
                                                                                                      
     batch_normalization_51 (BatchN  (None, 17, 17, 160)  480        ['conv2d_51[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_56 (BatchN  (None, 17, 17, 160)  480        ['conv2d_56[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_51 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_51[0][0]'] 
                                                                                                      
     activation_56 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_56[0][0]'] 
                                                                                                      
     conv2d_52 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_51[0][0]']          
                                                                                                      
     conv2d_57 (Conv2D)             (None, 17, 17, 160)  179200      ['activation_56[0][0]']          
                                                                                                      
     batch_normalization_52 (BatchN  (None, 17, 17, 160)  480        ['conv2d_52[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_57 (BatchN  (None, 17, 17, 160)  480        ['conv2d_57[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_52 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_52[0][0]'] 
                                                                                                      
     activation_57 (Activation)     (None, 17, 17, 160)  0           ['batch_normalization_57[0][0]'] 
                                                                                                      
     average_pooling2d_5 (AveragePo  (None, 17, 17, 768)  0          ['mixed5[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_50 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed5[0][0]']                 
                                                                                                      
     conv2d_53 (Conv2D)             (None, 17, 17, 192)  215040      ['activation_52[0][0]']          
                                                                                                      
     conv2d_58 (Conv2D)             (None, 17, 17, 192)  215040      ['activation_57[0][0]']          
                                                                                                      
     conv2d_59 (Conv2D)             (None, 17, 17, 192)  147456      ['average_pooling2d_5[0][0]']    
                                                                                                      
     batch_normalization_50 (BatchN  (None, 17, 17, 192)  576        ['conv2d_50[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_53 (BatchN  (None, 17, 17, 192)  576        ['conv2d_53[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_58 (BatchN  (None, 17, 17, 192)  576        ['conv2d_58[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_59 (BatchN  (None, 17, 17, 192)  576        ['conv2d_59[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_50 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_50[0][0]'] 
                                                                                                      
     activation_53 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_53[0][0]'] 
                                                                                                      
     activation_58 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_58[0][0]'] 
                                                                                                      
     activation_59 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_59[0][0]'] 
                                                                                                      
     mixed6 (Concatenate)           (None, 17, 17, 768)  0           ['activation_50[0][0]',          
                                                                      'activation_53[0][0]',          
                                                                      'activation_58[0][0]',          
                                                                      'activation_59[0][0]']          
                                                                                                      
     conv2d_64 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed6[0][0]']                 
                                                                                                      
     batch_normalization_64 (BatchN  (None, 17, 17, 192)  576        ['conv2d_64[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_64 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_64[0][0]'] 
                                                                                                      
     conv2d_65 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_64[0][0]']          
                                                                                                      
     batch_normalization_65 (BatchN  (None, 17, 17, 192)  576        ['conv2d_65[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_65 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_65[0][0]'] 
                                                                                                      
     conv2d_61 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed6[0][0]']                 
                                                                                                      
     conv2d_66 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_65[0][0]']          
                                                                                                      
     batch_normalization_61 (BatchN  (None, 17, 17, 192)  576        ['conv2d_61[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_66 (BatchN  (None, 17, 17, 192)  576        ['conv2d_66[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_61 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_61[0][0]'] 
                                                                                                      
     activation_66 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_66[0][0]'] 
                                                                                                      
     conv2d_62 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_61[0][0]']          
                                                                                                      
     conv2d_67 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_66[0][0]']          
                                                                                                      
     batch_normalization_62 (BatchN  (None, 17, 17, 192)  576        ['conv2d_62[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_67 (BatchN  (None, 17, 17, 192)  576        ['conv2d_67[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_62 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_62[0][0]'] 
                                                                                                      
     activation_67 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_67[0][0]'] 
                                                                                                      
     average_pooling2d_6 (AveragePo  (None, 17, 17, 768)  0          ['mixed6[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_60 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed6[0][0]']                 
                                                                                                      
     conv2d_63 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_62[0][0]']          
                                                                                                      
     conv2d_68 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_67[0][0]']          
                                                                                                      
     conv2d_69 (Conv2D)             (None, 17, 17, 192)  147456      ['average_pooling2d_6[0][0]']    
                                                                                                      
     batch_normalization_60 (BatchN  (None, 17, 17, 192)  576        ['conv2d_60[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_63 (BatchN  (None, 17, 17, 192)  576        ['conv2d_63[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_68 (BatchN  (None, 17, 17, 192)  576        ['conv2d_68[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_69 (BatchN  (None, 17, 17, 192)  576        ['conv2d_69[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_60 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_60[0][0]'] 
                                                                                                      
     activation_63 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_63[0][0]'] 
                                                                                                      
     activation_68 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_68[0][0]'] 
                                                                                                      
     activation_69 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_69[0][0]'] 
                                                                                                      
     mixed7 (Concatenate)           (None, 17, 17, 768)  0           ['activation_60[0][0]',          
                                                                      'activation_63[0][0]',          
                                                                      'activation_68[0][0]',          
                                                                      'activation_69[0][0]']          
                                                                                                      
     conv2d_72 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed7[0][0]']                 
                                                                                                      
     batch_normalization_72 (BatchN  (None, 17, 17, 192)  576        ['conv2d_72[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_72 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_72[0][0]'] 
                                                                                                      
     conv2d_73 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_72[0][0]']          
                                                                                                      
     batch_normalization_73 (BatchN  (None, 17, 17, 192)  576        ['conv2d_73[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_73 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_73[0][0]'] 
                                                                                                      
     conv2d_70 (Conv2D)             (None, 17, 17, 192)  147456      ['mixed7[0][0]']                 
                                                                                                      
     conv2d_74 (Conv2D)             (None, 17, 17, 192)  258048      ['activation_73[0][0]']          
                                                                                                      
     batch_normalization_70 (BatchN  (None, 17, 17, 192)  576        ['conv2d_70[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_74 (BatchN  (None, 17, 17, 192)  576        ['conv2d_74[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_70 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_70[0][0]'] 
                                                                                                      
     activation_74 (Activation)     (None, 17, 17, 192)  0           ['batch_normalization_74[0][0]'] 
                                                                                                      
     conv2d_71 (Conv2D)             (None, 8, 8, 320)    552960      ['activation_70[0][0]']          
                                                                                                      
     conv2d_75 (Conv2D)             (None, 8, 8, 192)    331776      ['activation_74[0][0]']          
                                                                                                      
     batch_normalization_71 (BatchN  (None, 8, 8, 320)   960         ['conv2d_71[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_75 (BatchN  (None, 8, 8, 192)   576         ['conv2d_75[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_71 (Activation)     (None, 8, 8, 320)    0           ['batch_normalization_71[0][0]'] 
                                                                                                      
     activation_75 (Activation)     (None, 8, 8, 192)    0           ['batch_normalization_75[0][0]'] 
                                                                                                      
     max_pooling2d_3 (MaxPooling2D)  (None, 8, 8, 768)   0           ['mixed7[0][0]']                 
                                                                                                      
     mixed8 (Concatenate)           (None, 8, 8, 1280)   0           ['activation_71[0][0]',          
                                                                      'activation_75[0][0]',          
                                                                      'max_pooling2d_3[0][0]']        
                                                                                                      
     conv2d_80 (Conv2D)             (None, 8, 8, 448)    573440      ['mixed8[0][0]']                 
                                                                                                      
     batch_normalization_80 (BatchN  (None, 8, 8, 448)   1344        ['conv2d_80[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_80 (Activation)     (None, 8, 8, 448)    0           ['batch_normalization_80[0][0]'] 
                                                                                                      
     conv2d_77 (Conv2D)             (None, 8, 8, 384)    491520      ['mixed8[0][0]']                 
                                                                                                      
     conv2d_81 (Conv2D)             (None, 8, 8, 384)    1548288     ['activation_80[0][0]']          
                                                                                                      
     batch_normalization_77 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_77[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_81 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_81[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_77 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_77[0][0]'] 
                                                                                                      
     activation_81 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_81[0][0]'] 
                                                                                                      
     conv2d_78 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_77[0][0]']          
                                                                                                      
     conv2d_79 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_77[0][0]']          
                                                                                                      
     conv2d_82 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_81[0][0]']          
                                                                                                      
     conv2d_83 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_81[0][0]']          
                                                                                                      
     average_pooling2d_7 (AveragePo  (None, 8, 8, 1280)  0           ['mixed8[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_76 (Conv2D)             (None, 8, 8, 320)    409600      ['mixed8[0][0]']                 
                                                                                                      
     batch_normalization_78 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_78[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_79 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_79[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_82 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_82[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_83 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_83[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     conv2d_84 (Conv2D)             (None, 8, 8, 192)    245760      ['average_pooling2d_7[0][0]']    
                                                                                                      
     batch_normalization_76 (BatchN  (None, 8, 8, 320)   960         ['conv2d_76[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_78 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_78[0][0]'] 
                                                                                                      
     activation_79 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_79[0][0]'] 
                                                                                                      
     activation_82 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_82[0][0]'] 
                                                                                                      
     activation_83 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_83[0][0]'] 
                                                                                                      
     batch_normalization_84 (BatchN  (None, 8, 8, 192)   576         ['conv2d_84[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_76 (Activation)     (None, 8, 8, 320)    0           ['batch_normalization_76[0][0]'] 
                                                                                                      
     mixed9_0 (Concatenate)         (None, 8, 8, 768)    0           ['activation_78[0][0]',          
                                                                      'activation_79[0][0]']          
                                                                                                      
     concatenate (Concatenate)      (None, 8, 8, 768)    0           ['activation_82[0][0]',          
                                                                      'activation_83[0][0]']          
                                                                                                      
     activation_84 (Activation)     (None, 8, 8, 192)    0           ['batch_normalization_84[0][0]'] 
                                                                                                      
     mixed9 (Concatenate)           (None, 8, 8, 2048)   0           ['activation_76[0][0]',          
                                                                      'mixed9_0[0][0]',               
                                                                      'concatenate[0][0]',            
                                                                      'activation_84[0][0]']          
                                                                                                      
     conv2d_89 (Conv2D)             (None, 8, 8, 448)    917504      ['mixed9[0][0]']                 
                                                                                                      
     batch_normalization_89 (BatchN  (None, 8, 8, 448)   1344        ['conv2d_89[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_89 (Activation)     (None, 8, 8, 448)    0           ['batch_normalization_89[0][0]'] 
                                                                                                      
     conv2d_86 (Conv2D)             (None, 8, 8, 384)    786432      ['mixed9[0][0]']                 
                                                                                                      
     conv2d_90 (Conv2D)             (None, 8, 8, 384)    1548288     ['activation_89[0][0]']          
                                                                                                      
     batch_normalization_86 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_86[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_90 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_90[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_86 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_86[0][0]'] 
                                                                                                      
     activation_90 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_90[0][0]'] 
                                                                                                      
     conv2d_87 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_86[0][0]']          
                                                                                                      
     conv2d_88 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_86[0][0]']          
                                                                                                      
     conv2d_91 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_90[0][0]']          
                                                                                                      
     conv2d_92 (Conv2D)             (None, 8, 8, 384)    442368      ['activation_90[0][0]']          
                                                                                                      
     average_pooling2d_8 (AveragePo  (None, 8, 8, 2048)  0           ['mixed9[0][0]']                 
     oling2D)                                                                                         
                                                                                                      
     conv2d_85 (Conv2D)             (None, 8, 8, 320)    655360      ['mixed9[0][0]']                 
                                                                                                      
     batch_normalization_87 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_87[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_88 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_88[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_91 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_91[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     batch_normalization_92 (BatchN  (None, 8, 8, 384)   1152        ['conv2d_92[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     conv2d_93 (Conv2D)             (None, 8, 8, 192)    393216      ['average_pooling2d_8[0][0]']    
                                                                                                      
     batch_normalization_85 (BatchN  (None, 8, 8, 320)   960         ['conv2d_85[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_87 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_87[0][0]'] 
                                                                                                      
     activation_88 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_88[0][0]'] 
                                                                                                      
     activation_91 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_91[0][0]'] 
                                                                                                      
     activation_92 (Activation)     (None, 8, 8, 384)    0           ['batch_normalization_92[0][0]'] 
                                                                                                      
     batch_normalization_93 (BatchN  (None, 8, 8, 192)   576         ['conv2d_93[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     activation_85 (Activation)     (None, 8, 8, 320)    0           ['batch_normalization_85[0][0]'] 
                                                                                                      
     mixed9_1 (Concatenate)         (None, 8, 8, 768)    0           ['activation_87[0][0]',          
                                                                      'activation_88[0][0]']          
                                                                                                      
     concatenate_1 (Concatenate)    (None, 8, 8, 768)    0           ['activation_91[0][0]',          
                                                                      'activation_92[0][0]']          
                                                                                                      
     activation_93 (Activation)     (None, 8, 8, 192)    0           ['batch_normalization_93[0][0]'] 
                                                                                                      
     mixed10 (Concatenate)          (None, 8, 8, 2048)   0           ['activation_85[0][0]',          
                                                                      'mixed9_1[0][0]',               
                                                                      'concatenate_1[0][0]',          
                                                                      'activation_93[0][0]']          
                                                                                                      
     avg_pool (GlobalAveragePooling  (None, 2048)        0           ['mixed10[0][0]']                
     2D)                                                                                              
                                                                                                      
     predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 23,851,784
    Trainable params: 23,817,352
    Non-trainable params: 34,432
    __________________________________________________________________________________________________
    


```python
!wget -O fish.jpg https://upload.wikimedia.org/wikipedia/commons/7/7a/Goldfish_1.jpg
```

    --2023-03-09 14:37:13--  https://upload.wikimedia.org/wikipedia/commons/7/7a/Goldfish_1.jpg
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 103.102.166.240, 2001:df2:e500:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|103.102.166.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4648040 (4.4M) [image/jpeg]
    Saving to: ‘fish.jpg’
    
    fish.jpg            100%[===================>]   4.43M  2.86MB/s    in 1.5s    
    
    2023-03-09 14:37:16 (2.86 MB/s) - ‘fish.jpg’ saved [4648040/4648040]
    
    


```python
# 다운로드한 fish.jpg를 terget_size로 줄여줌
# fish의 input은 [(None, 299, 299, 3  0)] 크기를 가짐
img = image.load_img('fish.jpg', target_size=(299, 299))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = inception.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    1/1 [==============================] - 2s 2s/step
    [[('n01443537', 'goldfish', 0.9748526), ('n02701002', 'ambulance', 0.0023246657), ('n02606052', 'rock_beauty', 0.0019062766), ('n02607072', 'anemone_fish', 0.0006647558), ('n09256479', 'coral_reef', 0.0004319898)]]
    





## ResNet(Residual Net)

- 네트워크의 깊이가 깊어질수록 경사가 소실되거나 폭발하는 문제를 해결하고자 함
- 병목 합성곱 계층을 추가하거나 크기가 작은 커널을 사용
- 152개의 훈련가능한 계층을 수직으로 연결하여 구성
- 모든 합성곱과 풀링 계층에서 패딩옵셥으로 `'SAME'`, `stride=1` 사용
- 3x3 합성곱 계층 다음마다 배치 정규화 적용, 1x1 합성곱 계층에는 활성화 함수가 존재하지 않음

  <img src="https://miro.medium.com/max/1200/1*6hF97Upuqg_LdsqWY6n_wg.png">


```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
```


```python
resnet = ResNet50(include_top=True, weights='imagenet',
                  input_tensor=None, input_shape=None,
                  pooling=None, classes=1000)
resnet.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5
    102967424/102967424 [==============================] - 29s 0us/step
    Model: "resnet50"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_3 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                    )]                                                                
                                                                                                      
     conv1_pad (ZeroPadding2D)      (None, 230, 230, 3)  0           ['input_3[0][0]']                
                                                                                                      
     conv1_conv (Conv2D)            (None, 112, 112, 64  9472        ['conv1_pad[0][0]']              
                                    )                                                                 
                                                                                                      
     conv1_bn (BatchNormalization)  (None, 112, 112, 64  256         ['conv1_conv[0][0]']             
                                    )                                                                 
                                                                                                      
     conv1_relu (Activation)        (None, 112, 112, 64  0           ['conv1_bn[0][0]']               
                                    )                                                                 
                                                                                                      
     pool1_pad (ZeroPadding2D)      (None, 114, 114, 64  0           ['conv1_relu[0][0]']             
                                    )                                                                 
                                                                                                      
     pool1_pool (MaxPooling2D)      (None, 56, 56, 64)   0           ['pool1_pad[0][0]']              
                                                                                                      
     conv2_block1_1_conv (Conv2D)   (None, 56, 56, 64)   4160        ['pool1_pool[0][0]']             
                                                                                                      
     conv2_block1_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block1_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block1_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block1_1_relu[0][0]']    
                                                                                                      
     conv2_block1_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block1_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block1_0_conv (Conv2D)   (None, 56, 56, 256)  16640       ['pool1_pool[0][0]']             
                                                                                                      
     conv2_block1_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block1_2_relu[0][0]']    
                                                                                                      
     conv2_block1_0_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block1_0_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block1_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block1_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block1_add (Add)         (None, 56, 56, 256)  0           ['conv2_block1_0_bn[0][0]',      
                                                                      'conv2_block1_3_bn[0][0]']      
                                                                                                      
     conv2_block1_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block1_add[0][0]']       
                                                                                                      
     conv2_block2_1_conv (Conv2D)   (None, 56, 56, 64)   16448       ['conv2_block1_out[0][0]']       
                                                                                                      
     conv2_block2_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block2_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block2_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block2_1_relu[0][0]']    
                                                                                                      
     conv2_block2_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block2_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block2_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block2_2_relu[0][0]']    
                                                                                                      
     conv2_block2_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block2_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block2_add (Add)         (None, 56, 56, 256)  0           ['conv2_block1_out[0][0]',       
                                                                      'conv2_block2_3_bn[0][0]']      
                                                                                                      
     conv2_block2_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block2_add[0][0]']       
                                                                                                      
     conv2_block3_1_conv (Conv2D)   (None, 56, 56, 64)   16448       ['conv2_block2_out[0][0]']       
                                                                                                      
     conv2_block3_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block3_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block3_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block3_1_relu[0][0]']    
                                                                                                      
     conv2_block3_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block3_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block3_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block3_2_relu[0][0]']    
                                                                                                      
     conv2_block3_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block3_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block3_add (Add)         (None, 56, 56, 256)  0           ['conv2_block2_out[0][0]',       
                                                                      'conv2_block3_3_bn[0][0]']      
                                                                                                      
     conv2_block3_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block3_add[0][0]']       
                                                                                                      
     conv3_block1_1_conv (Conv2D)   (None, 28, 28, 128)  32896       ['conv2_block3_out[0][0]']       
                                                                                                      
     conv3_block1_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block1_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block1_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block1_1_relu[0][0]']    
                                                                                                      
     conv3_block1_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block1_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block1_0_conv (Conv2D)   (None, 28, 28, 512)  131584      ['conv2_block3_out[0][0]']       
                                                                                                      
     conv3_block1_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block1_2_relu[0][0]']    
                                                                                                      
     conv3_block1_0_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block1_0_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block1_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block1_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block1_add (Add)         (None, 28, 28, 512)  0           ['conv3_block1_0_bn[0][0]',      
                                                                      'conv3_block1_3_bn[0][0]']      
                                                                                                      
     conv3_block1_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block1_add[0][0]']       
                                                                                                      
     conv3_block2_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block1_out[0][0]']       
                                                                                                      
     conv3_block2_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block2_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block2_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block2_1_relu[0][0]']    
                                                                                                      
     conv3_block2_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block2_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block2_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block2_2_relu[0][0]']    
                                                                                                      
     conv3_block2_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block2_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block2_add (Add)         (None, 28, 28, 512)  0           ['conv3_block1_out[0][0]',       
                                                                      'conv3_block2_3_bn[0][0]']      
                                                                                                      
     conv3_block2_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block2_add[0][0]']       
                                                                                                      
     conv3_block3_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block2_out[0][0]']       
                                                                                                      
     conv3_block3_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block3_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block3_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block3_1_relu[0][0]']    
                                                                                                      
     conv3_block3_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block3_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block3_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block3_2_relu[0][0]']    
                                                                                                      
     conv3_block3_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block3_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block3_add (Add)         (None, 28, 28, 512)  0           ['conv3_block2_out[0][0]',       
                                                                      'conv3_block3_3_bn[0][0]']      
                                                                                                      
     conv3_block3_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block3_add[0][0]']       
                                                                                                      
     conv3_block4_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block3_out[0][0]']       
                                                                                                      
     conv3_block4_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block4_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block4_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block4_1_relu[0][0]']    
                                                                                                      
     conv3_block4_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block4_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block4_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block4_2_relu[0][0]']    
                                                                                                      
     conv3_block4_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block4_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block4_add (Add)         (None, 28, 28, 512)  0           ['conv3_block3_out[0][0]',       
                                                                      'conv3_block4_3_bn[0][0]']      
                                                                                                      
     conv3_block4_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block4_add[0][0]']       
                                                                                                      
     conv4_block1_1_conv (Conv2D)   (None, 14, 14, 256)  131328      ['conv3_block4_out[0][0]']       
                                                                                                      
     conv4_block1_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block1_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block1_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block1_1_relu[0][0]']    
                                                                                                      
     conv4_block1_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block1_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block1_0_conv (Conv2D)   (None, 14, 14, 1024  525312      ['conv3_block4_out[0][0]']       
                                    )                                                                 
                                                                                                      
     conv4_block1_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block1_2_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     conv4_block1_0_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block1_0_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block1_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block1_3_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block1_add (Add)         (None, 14, 14, 1024  0           ['conv4_block1_0_bn[0][0]',      
                                    )                                 'conv4_block1_3_bn[0][0]']      
                                                                                                      
     conv4_block1_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block1_add[0][0]']       
                                    )                                                                 
                                                                                                      
     conv4_block2_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block1_out[0][0]']       
                                                                                                      
     conv4_block2_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block2_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block2_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block2_1_relu[0][0]']    
                                                                                                      
     conv4_block2_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block2_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block2_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block2_2_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     conv4_block2_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block2_3_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block2_add (Add)         (None, 14, 14, 1024  0           ['conv4_block1_out[0][0]',       
                                    )                                 'conv4_block2_3_bn[0][0]']      
                                                                                                      
     conv4_block2_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block2_add[0][0]']       
                                    )                                                                 
                                                                                                      
     conv4_block3_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block2_out[0][0]']       
                                                                                                      
     conv4_block3_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block3_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block3_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block3_1_relu[0][0]']    
                                                                                                      
     conv4_block3_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block3_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block3_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block3_2_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     conv4_block3_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block3_3_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block3_add (Add)         (None, 14, 14, 1024  0           ['conv4_block2_out[0][0]',       
                                    )                                 'conv4_block3_3_bn[0][0]']      
                                                                                                      
     conv4_block3_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block3_add[0][0]']       
                                    )                                                                 
                                                                                                      
     conv4_block4_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block3_out[0][0]']       
                                                                                                      
     conv4_block4_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block4_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block4_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block4_1_relu[0][0]']    
                                                                                                      
     conv4_block4_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block4_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block4_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block4_2_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     conv4_block4_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block4_3_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block4_add (Add)         (None, 14, 14, 1024  0           ['conv4_block3_out[0][0]',       
                                    )                                 'conv4_block4_3_bn[0][0]']      
                                                                                                      
     conv4_block4_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block4_add[0][0]']       
                                    )                                                                 
                                                                                                      
     conv4_block5_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block4_out[0][0]']       
                                                                                                      
     conv4_block5_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block5_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block5_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block5_1_relu[0][0]']    
                                                                                                      
     conv4_block5_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block5_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block5_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block5_2_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     conv4_block5_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block5_3_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block5_add (Add)         (None, 14, 14, 1024  0           ['conv4_block4_out[0][0]',       
                                    )                                 'conv4_block5_3_bn[0][0]']      
                                                                                                      
     conv4_block5_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block5_add[0][0]']       
                                    )                                                                 
                                                                                                      
     conv4_block6_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block5_out[0][0]']       
                                                                                                      
     conv4_block6_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block6_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block6_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block6_1_relu[0][0]']    
                                                                                                      
     conv4_block6_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block6_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block6_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block6_2_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     conv4_block6_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block6_3_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     conv4_block6_add (Add)         (None, 14, 14, 1024  0           ['conv4_block5_out[0][0]',       
                                    )                                 'conv4_block6_3_bn[0][0]']      
                                                                                                      
     conv4_block6_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block6_add[0][0]']       
                                    )                                                                 
                                                                                                      
     conv5_block1_1_conv (Conv2D)   (None, 7, 7, 512)    524800      ['conv4_block6_out[0][0]']       
                                                                                                      
     conv5_block1_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block1_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block1_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block1_1_relu[0][0]']    
                                                                                                      
     conv5_block1_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block1_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block1_0_conv (Conv2D)   (None, 7, 7, 2048)   2099200     ['conv4_block6_out[0][0]']       
                                                                                                      
     conv5_block1_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block1_2_relu[0][0]']    
                                                                                                      
     conv5_block1_0_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block1_0_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block1_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block1_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block1_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_0_bn[0][0]',      
                                                                      'conv5_block1_3_bn[0][0]']      
                                                                                                      
     conv5_block1_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block1_add[0][0]']       
                                                                                                      
     conv5_block2_1_conv (Conv2D)   (None, 7, 7, 512)    1049088     ['conv5_block1_out[0][0]']       
                                                                                                      
     conv5_block2_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block2_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block2_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block2_1_relu[0][0]']    
                                                                                                      
     conv5_block2_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block2_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block2_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block2_2_relu[0][0]']    
                                                                                                      
     conv5_block2_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block2_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block2_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_out[0][0]',       
                                                                      'conv5_block2_3_bn[0][0]']      
                                                                                                      
     conv5_block2_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block2_add[0][0]']       
                                                                                                      
     conv5_block3_1_conv (Conv2D)   (None, 7, 7, 512)    1049088     ['conv5_block2_out[0][0]']       
                                                                                                      
     conv5_block3_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block3_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block3_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block3_1_relu[0][0]']    
                                                                                                      
     conv5_block3_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_2_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block3_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_2_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block3_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block3_2_relu[0][0]']    
                                                                                                      
     conv5_block3_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block3_3_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block3_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block2_out[0][0]',       
                                                                      'conv5_block3_3_bn[0][0]']      
                                                                                                      
     conv5_block3_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block3_add[0][0]']       
                                                                                                      
     avg_pool (GlobalAveragePooling  (None, 2048)        0           ['conv5_block3_out[0][0]']       
     2D)                                                                                              
                                                                                                      
     predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 25,636,712
    Trainable params: 25,583,592
    Non-trainable params: 53,120
    __________________________________________________________________________________________________
    


```python
!wget -O bee.jog https://upload.wikimedia.org/wikipedia/commons/4/4d/Apis_mellifera_Western_honey_bee.jpg
```

    --2023-03-09 14:46:48--  https://upload.wikimedia.org/wikipedia/commons/4/4d/Apis_mellifera_Western_honey_bee.jpg
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 103.102.166.240, 2620:0:862:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|103.102.166.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2421052 (2.3M) [image/jpeg]
    Saving to: ‘bee.jog’
    
    bee.jog             100%[===================>]   2.31M  --.-KB/s    in 0.08s   
    
    2023-03-09 14:46:48 (29.2 MB/s) - ‘bee.jog’ saved [2421052/2421052]
    
    


```python
# 다운로드한 bee.jpg를 terget_size로 줄여줌
# bee의 input은 [(None, 224, 224, 3  0)] 크기를 가짐
img = image.load_img('bee.jog', target_size=(224, 224))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = resnet.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    1/1 [==============================] - 1s 1s/step
    [[('n02206856', 'bee', 0.9990982), ('n03530642', 'honeycomb', 0.000561837), ('n02190166', 'fly', 0.00014348973), ('n02727426', 'apiary', 0.0001017911), ('n02219486', 'ant', 5.7517515e-05)]]
    



## Xception

* Inception module을 이용하여 depthwise convolution 적용

* 기존의 Conv layer에서 얻은 feature map을 각 채널별로 다른 Conv layer에 적용하여 feature map을 얻음


```python
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
```


```python
xception = Xception(include_top=True, weights='imagenet',
                    input_tensor=None, input_shape=None,
                    pooling=None, classes=1000)
xception.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5
    91884032/91884032 [==============================] - 6s 0us/step
    Model: "xception"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_4 (InputLayer)           [(None, 299, 299, 3  0           []                               
                                    )]                                                                
                                                                                                      
     block1_conv1 (Conv2D)          (None, 149, 149, 32  864         ['input_4[0][0]']                
                                    )                                                                 
                                                                                                      
     block1_conv1_bn (BatchNormaliz  (None, 149, 149, 32  128        ['block1_conv1[0][0]']           
     ation)                         )                                                                 
                                                                                                      
     block1_conv1_act (Activation)  (None, 149, 149, 32  0           ['block1_conv1_bn[0][0]']        
                                    )                                                                 
                                                                                                      
     block1_conv2 (Conv2D)          (None, 147, 147, 64  18432       ['block1_conv1_act[0][0]']       
                                    )                                                                 
                                                                                                      
     block1_conv2_bn (BatchNormaliz  (None, 147, 147, 64  256        ['block1_conv2[0][0]']           
     ation)                         )                                                                 
                                                                                                      
     block1_conv2_act (Activation)  (None, 147, 147, 64  0           ['block1_conv2_bn[0][0]']        
                                    )                                                                 
                                                                                                      
     block2_sepconv1 (SeparableConv  (None, 147, 147, 12  8768       ['block1_conv2_act[0][0]']       
     2D)                            8)                                                                
                                                                                                      
     block2_sepconv1_bn (BatchNorma  (None, 147, 147, 12  512        ['block2_sepconv1[0][0]']        
     lization)                      8)                                                                
                                                                                                      
     block2_sepconv2_act (Activatio  (None, 147, 147, 12  0          ['block2_sepconv1_bn[0][0]']     
     n)                             8)                                                                
                                                                                                      
     block2_sepconv2 (SeparableConv  (None, 147, 147, 12  17536      ['block2_sepconv2_act[0][0]']    
     2D)                            8)                                                                
                                                                                                      
     block2_sepconv2_bn (BatchNorma  (None, 147, 147, 12  512        ['block2_sepconv2[0][0]']        
     lization)                      8)                                                                
                                                                                                      
     conv2d_94 (Conv2D)             (None, 74, 74, 128)  8192        ['block1_conv2_act[0][0]']       
                                                                                                      
     block2_pool (MaxPooling2D)     (None, 74, 74, 128)  0           ['block2_sepconv2_bn[0][0]']     
                                                                                                      
     batch_normalization_94 (BatchN  (None, 74, 74, 128)  512        ['conv2d_94[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     add (Add)                      (None, 74, 74, 128)  0           ['block2_pool[0][0]',            
                                                                      'batch_normalization_94[0][0]'] 
                                                                                                      
     block3_sepconv1_act (Activatio  (None, 74, 74, 128)  0          ['add[0][0]']                    
     n)                                                                                               
                                                                                                      
     block3_sepconv1 (SeparableConv  (None, 74, 74, 256)  33920      ['block3_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block3_sepconv1_bn (BatchNorma  (None, 74, 74, 256)  1024       ['block3_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block3_sepconv2_act (Activatio  (None, 74, 74, 256)  0          ['block3_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block3_sepconv2 (SeparableConv  (None, 74, 74, 256)  67840      ['block3_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block3_sepconv2_bn (BatchNorma  (None, 74, 74, 256)  1024       ['block3_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     conv2d_95 (Conv2D)             (None, 37, 37, 256)  32768       ['add[0][0]']                    
                                                                                                      
     block3_pool (MaxPooling2D)     (None, 37, 37, 256)  0           ['block3_sepconv2_bn[0][0]']     
                                                                                                      
     batch_normalization_95 (BatchN  (None, 37, 37, 256)  1024       ['conv2d_95[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     add_1 (Add)                    (None, 37, 37, 256)  0           ['block3_pool[0][0]',            
                                                                      'batch_normalization_95[0][0]'] 
                                                                                                      
     block4_sepconv1_act (Activatio  (None, 37, 37, 256)  0          ['add_1[0][0]']                  
     n)                                                                                               
                                                                                                      
     block4_sepconv1 (SeparableConv  (None, 37, 37, 728)  188672     ['block4_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block4_sepconv1_bn (BatchNorma  (None, 37, 37, 728)  2912       ['block4_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block4_sepconv2_act (Activatio  (None, 37, 37, 728)  0          ['block4_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block4_sepconv2 (SeparableConv  (None, 37, 37, 728)  536536     ['block4_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block4_sepconv2_bn (BatchNorma  (None, 37, 37, 728)  2912       ['block4_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     conv2d_96 (Conv2D)             (None, 19, 19, 728)  186368      ['add_1[0][0]']                  
                                                                                                      
     block4_pool (MaxPooling2D)     (None, 19, 19, 728)  0           ['block4_sepconv2_bn[0][0]']     
                                                                                                      
     batch_normalization_96 (BatchN  (None, 19, 19, 728)  2912       ['conv2d_96[0][0]']              
     ormalization)                                                                                    
                                                                                                      
     add_2 (Add)                    (None, 19, 19, 728)  0           ['block4_pool[0][0]',            
                                                                      'batch_normalization_96[0][0]'] 
                                                                                                      
     block5_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_2[0][0]']                  
     n)                                                                                               
                                                                                                      
     block5_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block5_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block5_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block5_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block5_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block5_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block5_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block5_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block5_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block5_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     block5_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block5_sepconv2_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block5_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block5_sepconv3_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block5_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block5_sepconv3[0][0]']        
     lization)                                                                                        
                                                                                                      
     add_3 (Add)                    (None, 19, 19, 728)  0           ['block5_sepconv3_bn[0][0]',     
                                                                      'add_2[0][0]']                  
                                                                                                      
     block6_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_3[0][0]']                  
     n)                                                                                               
                                                                                                      
     block6_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block6_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block6_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block6_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block6_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block6_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block6_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block6_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block6_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block6_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     block6_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block6_sepconv2_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block6_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block6_sepconv3_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block6_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block6_sepconv3[0][0]']        
     lization)                                                                                        
                                                                                                      
     add_4 (Add)                    (None, 19, 19, 728)  0           ['block6_sepconv3_bn[0][0]',     
                                                                      'add_3[0][0]']                  
                                                                                                      
     block7_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_4[0][0]']                  
     n)                                                                                               
                                                                                                      
     block7_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block7_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block7_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block7_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block7_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block7_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block7_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block7_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block7_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block7_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     block7_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block7_sepconv2_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block7_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block7_sepconv3_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block7_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block7_sepconv3[0][0]']        
     lization)                                                                                        
                                                                                                      
     add_5 (Add)                    (None, 19, 19, 728)  0           ['block7_sepconv3_bn[0][0]',     
                                                                      'add_4[0][0]']                  
                                                                                                      
     block8_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_5[0][0]']                  
     n)                                                                                               
                                                                                                      
     block8_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block8_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block8_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block8_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block8_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block8_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block8_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block8_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block8_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block8_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     block8_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block8_sepconv2_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block8_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block8_sepconv3_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block8_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block8_sepconv3[0][0]']        
     lization)                                                                                        
                                                                                                      
     add_6 (Add)                    (None, 19, 19, 728)  0           ['block8_sepconv3_bn[0][0]',     
                                                                      'add_5[0][0]']                  
                                                                                                      
     block9_sepconv1_act (Activatio  (None, 19, 19, 728)  0          ['add_6[0][0]']                  
     n)                                                                                               
                                                                                                      
     block9_sepconv1 (SeparableConv  (None, 19, 19, 728)  536536     ['block9_sepconv1_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block9_sepconv1_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block9_sepconv1[0][0]']        
     lization)                                                                                        
                                                                                                      
     block9_sepconv2_act (Activatio  (None, 19, 19, 728)  0          ['block9_sepconv1_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block9_sepconv2 (SeparableConv  (None, 19, 19, 728)  536536     ['block9_sepconv2_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block9_sepconv2_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block9_sepconv2[0][0]']        
     lization)                                                                                        
                                                                                                      
     block9_sepconv3_act (Activatio  (None, 19, 19, 728)  0          ['block9_sepconv2_bn[0][0]']     
     n)                                                                                               
                                                                                                      
     block9_sepconv3 (SeparableConv  (None, 19, 19, 728)  536536     ['block9_sepconv3_act[0][0]']    
     2D)                                                                                              
                                                                                                      
     block9_sepconv3_bn (BatchNorma  (None, 19, 19, 728)  2912       ['block9_sepconv3[0][0]']        
     lization)                                                                                        
                                                                                                      
     add_7 (Add)                    (None, 19, 19, 728)  0           ['block9_sepconv3_bn[0][0]',     
                                                                      'add_6[0][0]']                  
                                                                                                      
     block10_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_7[0][0]']                  
     on)                                                                                              
                                                                                                      
     block10_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block10_sepconv1_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block10_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block10_sepconv1[0][0]']       
     alization)                                                                                       
                                                                                                      
     block10_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block10_sepconv1_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block10_sepconv2 (SeparableCon  (None, 19, 19, 728)  536536     ['block10_sepconv2_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block10_sepconv2_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block10_sepconv2[0][0]']       
     alization)                                                                                       
                                                                                                      
     block10_sepconv3_act (Activati  (None, 19, 19, 728)  0          ['block10_sepconv2_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block10_sepconv3 (SeparableCon  (None, 19, 19, 728)  536536     ['block10_sepconv3_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block10_sepconv3_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block10_sepconv3[0][0]']       
     alization)                                                                                       
                                                                                                      
     add_8 (Add)                    (None, 19, 19, 728)  0           ['block10_sepconv3_bn[0][0]',    
                                                                      'add_7[0][0]']                  
                                                                                                      
     block11_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_8[0][0]']                  
     on)                                                                                              
                                                                                                      
     block11_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block11_sepconv1_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block11_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block11_sepconv1[0][0]']       
     alization)                                                                                       
                                                                                                      
     block11_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block11_sepconv1_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block11_sepconv2 (SeparableCon  (None, 19, 19, 728)  536536     ['block11_sepconv2_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block11_sepconv2_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block11_sepconv2[0][0]']       
     alization)                                                                                       
                                                                                                      
     block11_sepconv3_act (Activati  (None, 19, 19, 728)  0          ['block11_sepconv2_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block11_sepconv3 (SeparableCon  (None, 19, 19, 728)  536536     ['block11_sepconv3_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block11_sepconv3_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block11_sepconv3[0][0]']       
     alization)                                                                                       
                                                                                                      
     add_9 (Add)                    (None, 19, 19, 728)  0           ['block11_sepconv3_bn[0][0]',    
                                                                      'add_8[0][0]']                  
                                                                                                      
     block12_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_9[0][0]']                  
     on)                                                                                              
                                                                                                      
     block12_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block12_sepconv1_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block12_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block12_sepconv1[0][0]']       
     alization)                                                                                       
                                                                                                      
     block12_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block12_sepconv1_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block12_sepconv2 (SeparableCon  (None, 19, 19, 728)  536536     ['block12_sepconv2_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block12_sepconv2_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block12_sepconv2[0][0]']       
     alization)                                                                                       
                                                                                                      
     block12_sepconv3_act (Activati  (None, 19, 19, 728)  0          ['block12_sepconv2_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block12_sepconv3 (SeparableCon  (None, 19, 19, 728)  536536     ['block12_sepconv3_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block12_sepconv3_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block12_sepconv3[0][0]']       
     alization)                                                                                       
                                                                                                      
     add_10 (Add)                   (None, 19, 19, 728)  0           ['block12_sepconv3_bn[0][0]',    
                                                                      'add_9[0][0]']                  
                                                                                                      
     block13_sepconv1_act (Activati  (None, 19, 19, 728)  0          ['add_10[0][0]']                 
     on)                                                                                              
                                                                                                      
     block13_sepconv1 (SeparableCon  (None, 19, 19, 728)  536536     ['block13_sepconv1_act[0][0]']   
     v2D)                                                                                             
                                                                                                      
     block13_sepconv1_bn (BatchNorm  (None, 19, 19, 728)  2912       ['block13_sepconv1[0][0]']       
     alization)                                                                                       
                                                                                                      
     block13_sepconv2_act (Activati  (None, 19, 19, 728)  0          ['block13_sepconv1_bn[0][0]']    
     on)                                                                                              
                                                                                                      
     block13_sepconv2 (SeparableCon  (None, 19, 19, 1024  752024     ['block13_sepconv2_act[0][0]']   
     v2D)                           )                                                                 
                                                                                                      
     block13_sepconv2_bn (BatchNorm  (None, 19, 19, 1024  4096       ['block13_sepconv2[0][0]']       
     alization)                     )                                                                 
                                                                                                      
     conv2d_97 (Conv2D)             (None, 10, 10, 1024  745472      ['add_10[0][0]']                 
                                    )                                                                 
                                                                                                      
     block13_pool (MaxPooling2D)    (None, 10, 10, 1024  0           ['block13_sepconv2_bn[0][0]']    
                                    )                                                                 
                                                                                                      
     batch_normalization_97 (BatchN  (None, 10, 10, 1024  4096       ['conv2d_97[0][0]']              
     ormalization)                  )                                                                 
                                                                                                      
     add_11 (Add)                   (None, 10, 10, 1024  0           ['block13_pool[0][0]',           
                                    )                                 'batch_normalization_97[0][0]'] 
                                                                                                      
     block14_sepconv1 (SeparableCon  (None, 10, 10, 1536  1582080    ['add_11[0][0]']                 
     v2D)                           )                                                                 
                                                                                                      
     block14_sepconv1_bn (BatchNorm  (None, 10, 10, 1536  6144       ['block14_sepconv1[0][0]']       
     alization)                     )                                                                 
                                                                                                      
     block14_sepconv1_act (Activati  (None, 10, 10, 1536  0          ['block14_sepconv1_bn[0][0]']    
     on)                            )                                                                 
                                                                                                      
     block14_sepconv2 (SeparableCon  (None, 10, 10, 2048  3159552    ['block14_sepconv1_act[0][0]']   
     v2D)                           )                                                                 
                                                                                                      
     block14_sepconv2_bn (BatchNorm  (None, 10, 10, 2048  8192       ['block14_sepconv2[0][0]']       
     alization)                     )                                                                 
                                                                                                      
     block14_sepconv2_act (Activati  (None, 10, 10, 2048  0          ['block14_sepconv2_bn[0][0]']    
     on)                            )                                                                 
                                                                                                      
     avg_pool (GlobalAveragePooling  (None, 2048)        0           ['block14_sepconv2_act[0][0]']   
     2D)                                                                                              
                                                                                                      
     predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 22,910,480
    Trainable params: 22,855,952
    Non-trainable params: 54,528
    __________________________________________________________________________________________________
    


```
!wget -O beaver.jpg https://upload.wikimedia.org/wikipedia/commons/6/6b/American_Beaver.jpg
```

    --2023-03-09 14:52:46--  https://upload.wikimedia.org/wikipedia/commons/6/6b/American_Beaver.jpg
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 103.102.166.240, 2001:df2:e500:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|103.102.166.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 186747 (182K) [image/jpeg]
    Saving to: ‘beaver.jpg’
    
    beaver.jpg          100%[===================>] 182.37K  --.-KB/s    in 0.006s  
    
    2023-03-09 14:52:46 (28.0 MB/s) - ‘beaver.jpg’ saved [186747/186747]
    
    


```python
# 다운로드한 beaver.jpg를 terget_size로 줄여줌
# beaver input은 [(None, 299, 299, 3  0)] 크기를 가짐
img = image.load_img('beaver.jpg', target_size=(299, 299))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = xception.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    1/1 [==============================] - 1s 1s/step
    [[('n02363005', 'beaver', 0.8280816), ('n02361337', 'marmot', 0.059717532), ('n02493509', 'titi', 0.004413159), ('n02442845', 'mink', 0.0024041226), ('n01883070', 'wombat', 0.0019850638)]]
    




## MobileNet

* 성능보다 모델의 크기 또는 연산 속도 감소
* Depthwise conv와 Pointwise conv 사이에도 batch normalization과 ReLU를 삽입
* Conv layer를 활용한 모델과 정확도는 비슷하면서 계산량은 9배, 파라미터 수는 7배 줄임


```python
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
```


```python
mobilenet = MobileNet(include_top=True, weights='imagenet',
                      input_tensor=None, input_shape=None,
                      pooling=None, classes=1000)
mobilenet.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf.h5
    17225924/17225924 [==============================] - 2s 0us/step
    Model: "mobilenet_1.00_224"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_5 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     conv1 (Conv2D)              (None, 112, 112, 32)      864       
                                                                     
     conv1_bn (BatchNormalizatio  (None, 112, 112, 32)     128       
     n)                                                              
                                                                     
     conv1_relu (ReLU)           (None, 112, 112, 32)      0         
                                                                     
     conv_dw_1 (DepthwiseConv2D)  (None, 112, 112, 32)     288       
                                                                     
     conv_dw_1_bn (BatchNormaliz  (None, 112, 112, 32)     128       
     ation)                                                          
                                                                     
     conv_dw_1_relu (ReLU)       (None, 112, 112, 32)      0         
                                                                     
     conv_pw_1 (Conv2D)          (None, 112, 112, 64)      2048      
                                                                     
     conv_pw_1_bn (BatchNormaliz  (None, 112, 112, 64)     256       
     ation)                                                          
                                                                     
     conv_pw_1_relu (ReLU)       (None, 112, 112, 64)      0         
                                                                     
     conv_pad_2 (ZeroPadding2D)  (None, 113, 113, 64)      0         
                                                                     
     conv_dw_2 (DepthwiseConv2D)  (None, 56, 56, 64)       576       
                                                                     
     conv_dw_2_bn (BatchNormaliz  (None, 56, 56, 64)       256       
     ation)                                                          
                                                                     
     conv_dw_2_relu (ReLU)       (None, 56, 56, 64)        0         
                                                                     
     conv_pw_2 (Conv2D)          (None, 56, 56, 128)       8192      
                                                                     
     conv_pw_2_bn (BatchNormaliz  (None, 56, 56, 128)      512       
     ation)                                                          
                                                                     
     conv_pw_2_relu (ReLU)       (None, 56, 56, 128)       0         
                                                                     
     conv_dw_3 (DepthwiseConv2D)  (None, 56, 56, 128)      1152      
                                                                     
     conv_dw_3_bn (BatchNormaliz  (None, 56, 56, 128)      512       
     ation)                                                          
                                                                     
     conv_dw_3_relu (ReLU)       (None, 56, 56, 128)       0         
                                                                     
     conv_pw_3 (Conv2D)          (None, 56, 56, 128)       16384     
                                                                     
     conv_pw_3_bn (BatchNormaliz  (None, 56, 56, 128)      512       
     ation)                                                          
                                                                     
     conv_pw_3_relu (ReLU)       (None, 56, 56, 128)       0         
                                                                     
     conv_pad_4 (ZeroPadding2D)  (None, 57, 57, 128)       0         
                                                                     
     conv_dw_4 (DepthwiseConv2D)  (None, 28, 28, 128)      1152      
                                                                     
     conv_dw_4_bn (BatchNormaliz  (None, 28, 28, 128)      512       
     ation)                                                          
                                                                     
     conv_dw_4_relu (ReLU)       (None, 28, 28, 128)       0         
                                                                     
     conv_pw_4 (Conv2D)          (None, 28, 28, 256)       32768     
                                                                     
     conv_pw_4_bn (BatchNormaliz  (None, 28, 28, 256)      1024      
     ation)                                                          
                                                                     
     conv_pw_4_relu (ReLU)       (None, 28, 28, 256)       0         
                                                                     
     conv_dw_5 (DepthwiseConv2D)  (None, 28, 28, 256)      2304      
                                                                     
     conv_dw_5_bn (BatchNormaliz  (None, 28, 28, 256)      1024      
     ation)                                                          
                                                                     
     conv_dw_5_relu (ReLU)       (None, 28, 28, 256)       0         
                                                                     
     conv_pw_5 (Conv2D)          (None, 28, 28, 256)       65536     
                                                                     
     conv_pw_5_bn (BatchNormaliz  (None, 28, 28, 256)      1024      
     ation)                                                          
                                                                     
     conv_pw_5_relu (ReLU)       (None, 28, 28, 256)       0         
                                                                     
     conv_pad_6 (ZeroPadding2D)  (None, 29, 29, 256)       0         
                                                                     
     conv_dw_6 (DepthwiseConv2D)  (None, 14, 14, 256)      2304      
                                                                     
     conv_dw_6_bn (BatchNormaliz  (None, 14, 14, 256)      1024      
     ation)                                                          
                                                                     
     conv_dw_6_relu (ReLU)       (None, 14, 14, 256)       0         
                                                                     
     conv_pw_6 (Conv2D)          (None, 14, 14, 512)       131072    
                                                                     
     conv_pw_6_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_pw_6_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_dw_7 (DepthwiseConv2D)  (None, 14, 14, 512)      4608      
                                                                     
     conv_dw_7_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_dw_7_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_pw_7 (Conv2D)          (None, 14, 14, 512)       262144    
                                                                     
     conv_pw_7_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_pw_7_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_dw_8 (DepthwiseConv2D)  (None, 14, 14, 512)      4608      
                                                                     
     conv_dw_8_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_dw_8_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_pw_8 (Conv2D)          (None, 14, 14, 512)       262144    
                                                                     
     conv_pw_8_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_pw_8_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_dw_9 (DepthwiseConv2D)  (None, 14, 14, 512)      4608      
                                                                     
     conv_dw_9_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_dw_9_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_pw_9 (Conv2D)          (None, 14, 14, 512)       262144    
                                                                     
     conv_pw_9_bn (BatchNormaliz  (None, 14, 14, 512)      2048      
     ation)                                                          
                                                                     
     conv_pw_9_relu (ReLU)       (None, 14, 14, 512)       0         
                                                                     
     conv_dw_10 (DepthwiseConv2D  (None, 14, 14, 512)      4608      
     )                                                               
                                                                     
     conv_dw_10_bn (BatchNormali  (None, 14, 14, 512)      2048      
     zation)                                                         
                                                                     
     conv_dw_10_relu (ReLU)      (None, 14, 14, 512)       0         
                                                                     
     conv_pw_10 (Conv2D)         (None, 14, 14, 512)       262144    
                                                                     
     conv_pw_10_bn (BatchNormali  (None, 14, 14, 512)      2048      
     zation)                                                         
                                                                     
     conv_pw_10_relu (ReLU)      (None, 14, 14, 512)       0         
                                                                     
     conv_dw_11 (DepthwiseConv2D  (None, 14, 14, 512)      4608      
     )                                                               
                                                                     
     conv_dw_11_bn (BatchNormali  (None, 14, 14, 512)      2048      
     zation)                                                         
                                                                     
     conv_dw_11_relu (ReLU)      (None, 14, 14, 512)       0         
                                                                     
     conv_pw_11 (Conv2D)         (None, 14, 14, 512)       262144    
                                                                     
     conv_pw_11_bn (BatchNormali  (None, 14, 14, 512)      2048      
     zation)                                                         
                                                                     
     conv_pw_11_relu (ReLU)      (None, 14, 14, 512)       0         
                                                                     
     conv_pad_12 (ZeroPadding2D)  (None, 15, 15, 512)      0         
                                                                     
     conv_dw_12 (DepthwiseConv2D  (None, 7, 7, 512)        4608      
     )                                                               
                                                                     
     conv_dw_12_bn (BatchNormali  (None, 7, 7, 512)        2048      
     zation)                                                         
                                                                     
     conv_dw_12_relu (ReLU)      (None, 7, 7, 512)         0         
                                                                     
     conv_pw_12 (Conv2D)         (None, 7, 7, 1024)        524288    
                                                                     
     conv_pw_12_bn (BatchNormali  (None, 7, 7, 1024)       4096      
     zation)                                                         
                                                                     
     conv_pw_12_relu (ReLU)      (None, 7, 7, 1024)        0         
                                                                     
     conv_dw_13 (DepthwiseConv2D  (None, 7, 7, 1024)       9216      
     )                                                               
                                                                     
     conv_dw_13_bn (BatchNormali  (None, 7, 7, 1024)       4096      
     zation)                                                         
                                                                     
     conv_dw_13_relu (ReLU)      (None, 7, 7, 1024)        0         
                                                                     
     conv_pw_13 (Conv2D)         (None, 7, 7, 1024)        1048576   
                                                                     
     conv_pw_13_bn (BatchNormali  (None, 7, 7, 1024)       4096      
     zation)                                                         
                                                                     
     conv_pw_13_relu (ReLU)      (None, 7, 7, 1024)        0         
                                                                     
     global_average_pooling2d (G  (None, 1, 1, 1024)       0         
     lobalAveragePooling2D)                                          
                                                                     
     dropout (Dropout)           (None, 1, 1, 1024)        0         
                                                                     
     conv_preds (Conv2D)         (None, 1, 1, 1000)        1025000   
                                                                     
     reshape_2 (Reshape)         (None, 1000)              0         
                                                                     
     predictions (Activation)    (None, 1000)              0         
                                                                     
    =================================================================
    Total params: 4,253,864
    Trainable params: 4,231,976
    Non-trainable params: 21,888
    _________________________________________________________________
    


```python
!wget -O crane.jpg https://p1.pxfuel.com/preview/42/50/534/europe-channel-crane-harbour-crane-harbour-cranes-cranes-transport.jpg
```

    --2023-03-09 14:58:51--  https://p1.pxfuel.com/preview/42/50/534/europe-channel-crane-harbour-crane-harbour-cranes-cranes-transport.jpg
    Resolving p1.pxfuel.com (p1.pxfuel.com)... 172.64.201.22, 172.64.200.22, 2606:4700:e6::ac40:c916, ...
    Connecting to p1.pxfuel.com (p1.pxfuel.com)|172.64.201.22|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 86911 (85K) [image/jpeg]
    Saving to: ‘crane.jpg’
    
    crane.jpg           100%[===================>]  84.87K  --.-KB/s    in 0.004s  
    
    2023-03-09 14:58:52 (19.6 MB/s) - ‘crane.jpg’ saved [86911/86911]
    
    


```python
# 다운로드한 crane.jpg를 terget_size로 줄여줌
# crane input은 [(None, 224, 224, 3))] 크기를 가짐
img = image.load_img('crane.jpg', target_size=(224, 224))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = mobilenet.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    WARNING:tensorflow:5 out of the last 6 calls to <function Model.make_predict_function.<locals>.predict_function at 0x7f7aa7d3c8b0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    1/1 [==============================] - 1s 809ms/step
    [[('n03126707', 'crane', 0.95963544), ('n03216828', 'dock', 0.029736752), ('n03240683', 'drilling_platform', 0.0051688235), ('n03344393', 'fireboat', 0.0026515478), ('n04366367', 'suspension_bridge', 0.000502884)]]
    



## DenseNet

* 각 층은 모든 앞 단계에서 올 수 있는 지름질 연결 구성
* 특징지도의 크기를 줄이기 위해 풀링 연산 적용 필요
* 밀집 블록(dense block)과 전이층(transition layer)으로 구성 
* 전이층 : 1x1 컨볼루션과 평균값 풀링(APool)으로 구성   

<img src="https://oi.readthedocs.io/en/latest/_images/cnn_vs_resnet_vs_densenet.png" width="700">


```python
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions
```


```python
densenet = DenseNet201(include_top=True, weights='imagenet',
                      input_tensor=None, input_shape=None,
                      pooling=None, classes=1000)
densenet.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet201_weights_tf_dim_ordering_tf_kernels.h5
    82524592/82524592 [==============================] - 5s 0us/step
    Model: "densenet201"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_6 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                    )]                                                                
                                                                                                      
     zero_padding2d (ZeroPadding2D)  (None, 230, 230, 3)  0          ['input_6[0][0]']                
                                                                                                      
     conv1/conv (Conv2D)            (None, 112, 112, 64  9408        ['zero_padding2d[0][0]']         
                                    )                                                                 
                                                                                                      
     conv1/bn (BatchNormalization)  (None, 112, 112, 64  256         ['conv1/conv[0][0]']             
                                    )                                                                 
                                                                                                      
     conv1/relu (Activation)        (None, 112, 112, 64  0           ['conv1/bn[0][0]']               
                                    )                                                                 
                                                                                                      
     zero_padding2d_1 (ZeroPadding2  (None, 114, 114, 64  0          ['conv1/relu[0][0]']             
     D)                             )                                                                 
                                                                                                      
     pool1 (MaxPooling2D)           (None, 56, 56, 64)   0           ['zero_padding2d_1[0][0]']       
                                                                                                      
     conv2_block1_0_bn (BatchNormal  (None, 56, 56, 64)  256         ['pool1[0][0]']                  
     ization)                                                                                         
                                                                                                      
     conv2_block1_0_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block1_1_conv (Conv2D)   (None, 56, 56, 128)  8192        ['conv2_block1_0_relu[0][0]']    
                                                                                                      
     conv2_block1_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block1_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block1_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block1_1_relu[0][0]']    
                                                                                                      
     conv2_block1_concat (Concatena  (None, 56, 56, 96)  0           ['pool1[0][0]',                  
     te)                                                              'conv2_block1_2_conv[0][0]']    
                                                                                                      
     conv2_block2_0_bn (BatchNormal  (None, 56, 56, 96)  384         ['conv2_block1_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block2_0_relu (Activatio  (None, 56, 56, 96)  0           ['conv2_block2_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block2_1_conv (Conv2D)   (None, 56, 56, 128)  12288       ['conv2_block2_0_relu[0][0]']    
                                                                                                      
     conv2_block2_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block2_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block2_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block2_1_relu[0][0]']    
                                                                                                      
     conv2_block2_concat (Concatena  (None, 56, 56, 128)  0          ['conv2_block1_concat[0][0]',    
     te)                                                              'conv2_block2_2_conv[0][0]']    
                                                                                                      
     conv2_block3_0_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block2_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block3_0_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block3_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block3_1_conv (Conv2D)   (None, 56, 56, 128)  16384       ['conv2_block3_0_relu[0][0]']    
                                                                                                      
     conv2_block3_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block3_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block3_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block3_1_relu[0][0]']    
                                                                                                      
     conv2_block3_concat (Concatena  (None, 56, 56, 160)  0          ['conv2_block2_concat[0][0]',    
     te)                                                              'conv2_block3_2_conv[0][0]']    
                                                                                                      
     conv2_block4_0_bn (BatchNormal  (None, 56, 56, 160)  640        ['conv2_block3_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block4_0_relu (Activatio  (None, 56, 56, 160)  0          ['conv2_block4_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block4_1_conv (Conv2D)   (None, 56, 56, 128)  20480       ['conv2_block4_0_relu[0][0]']    
                                                                                                      
     conv2_block4_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block4_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block4_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block4_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block4_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block4_1_relu[0][0]']    
                                                                                                      
     conv2_block4_concat (Concatena  (None, 56, 56, 192)  0          ['conv2_block3_concat[0][0]',    
     te)                                                              'conv2_block4_2_conv[0][0]']    
                                                                                                      
     conv2_block5_0_bn (BatchNormal  (None, 56, 56, 192)  768        ['conv2_block4_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block5_0_relu (Activatio  (None, 56, 56, 192)  0          ['conv2_block5_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block5_1_conv (Conv2D)   (None, 56, 56, 128)  24576       ['conv2_block5_0_relu[0][0]']    
                                                                                                      
     conv2_block5_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block5_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block5_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block5_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block5_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block5_1_relu[0][0]']    
                                                                                                      
     conv2_block5_concat (Concatena  (None, 56, 56, 224)  0          ['conv2_block4_concat[0][0]',    
     te)                                                              'conv2_block5_2_conv[0][0]']    
                                                                                                      
     conv2_block6_0_bn (BatchNormal  (None, 56, 56, 224)  896        ['conv2_block5_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block6_0_relu (Activatio  (None, 56, 56, 224)  0          ['conv2_block6_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block6_1_conv (Conv2D)   (None, 56, 56, 128)  28672       ['conv2_block6_0_relu[0][0]']    
                                                                                                      
     conv2_block6_1_bn (BatchNormal  (None, 56, 56, 128)  512        ['conv2_block6_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv2_block6_1_relu (Activatio  (None, 56, 56, 128)  0          ['conv2_block6_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv2_block6_2_conv (Conv2D)   (None, 56, 56, 32)   36864       ['conv2_block6_1_relu[0][0]']    
                                                                                                      
     conv2_block6_concat (Concatena  (None, 56, 56, 256)  0          ['conv2_block5_concat[0][0]',    
     te)                                                              'conv2_block6_2_conv[0][0]']    
                                                                                                      
     pool2_bn (BatchNormalization)  (None, 56, 56, 256)  1024        ['conv2_block6_concat[0][0]']    
                                                                                                      
     pool2_relu (Activation)        (None, 56, 56, 256)  0           ['pool2_bn[0][0]']               
                                                                                                      
     pool2_conv (Conv2D)            (None, 56, 56, 128)  32768       ['pool2_relu[0][0]']             
                                                                                                      
     pool2_pool (AveragePooling2D)  (None, 28, 28, 128)  0           ['pool2_conv[0][0]']             
                                                                                                      
     conv3_block1_0_bn (BatchNormal  (None, 28, 28, 128)  512        ['pool2_pool[0][0]']             
     ization)                                                                                         
                                                                                                      
     conv3_block1_0_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block1_1_conv (Conv2D)   (None, 28, 28, 128)  16384       ['conv3_block1_0_relu[0][0]']    
                                                                                                      
     conv3_block1_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block1_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block1_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block1_1_relu[0][0]']    
                                                                                                      
     conv3_block1_concat (Concatena  (None, 28, 28, 160)  0          ['pool2_pool[0][0]',             
     te)                                                              'conv3_block1_2_conv[0][0]']    
                                                                                                      
     conv3_block2_0_bn (BatchNormal  (None, 28, 28, 160)  640        ['conv3_block1_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block2_0_relu (Activatio  (None, 28, 28, 160)  0          ['conv3_block2_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block2_1_conv (Conv2D)   (None, 28, 28, 128)  20480       ['conv3_block2_0_relu[0][0]']    
                                                                                                      
     conv3_block2_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block2_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block2_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block2_1_relu[0][0]']    
                                                                                                      
     conv3_block2_concat (Concatena  (None, 28, 28, 192)  0          ['conv3_block1_concat[0][0]',    
     te)                                                              'conv3_block2_2_conv[0][0]']    
                                                                                                      
     conv3_block3_0_bn (BatchNormal  (None, 28, 28, 192)  768        ['conv3_block2_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block3_0_relu (Activatio  (None, 28, 28, 192)  0          ['conv3_block3_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block3_1_conv (Conv2D)   (None, 28, 28, 128)  24576       ['conv3_block3_0_relu[0][0]']    
                                                                                                      
     conv3_block3_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block3_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block3_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block3_1_relu[0][0]']    
                                                                                                      
     conv3_block3_concat (Concatena  (None, 28, 28, 224)  0          ['conv3_block2_concat[0][0]',    
     te)                                                              'conv3_block3_2_conv[0][0]']    
                                                                                                      
     conv3_block4_0_bn (BatchNormal  (None, 28, 28, 224)  896        ['conv3_block3_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block4_0_relu (Activatio  (None, 28, 28, 224)  0          ['conv3_block4_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block4_1_conv (Conv2D)   (None, 28, 28, 128)  28672       ['conv3_block4_0_relu[0][0]']    
                                                                                                      
     conv3_block4_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block4_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block4_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block4_1_relu[0][0]']    
                                                                                                      
     conv3_block4_concat (Concatena  (None, 28, 28, 256)  0          ['conv3_block3_concat[0][0]',    
     te)                                                              'conv3_block4_2_conv[0][0]']    
                                                                                                      
     conv3_block5_0_bn (BatchNormal  (None, 28, 28, 256)  1024       ['conv3_block4_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block5_0_relu (Activatio  (None, 28, 28, 256)  0          ['conv3_block5_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block5_1_conv (Conv2D)   (None, 28, 28, 128)  32768       ['conv3_block5_0_relu[0][0]']    
                                                                                                      
     conv3_block5_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block5_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block5_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block5_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block5_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block5_1_relu[0][0]']    
                                                                                                      
     conv3_block5_concat (Concatena  (None, 28, 28, 288)  0          ['conv3_block4_concat[0][0]',    
     te)                                                              'conv3_block5_2_conv[0][0]']    
                                                                                                      
     conv3_block6_0_bn (BatchNormal  (None, 28, 28, 288)  1152       ['conv3_block5_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block6_0_relu (Activatio  (None, 28, 28, 288)  0          ['conv3_block6_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block6_1_conv (Conv2D)   (None, 28, 28, 128)  36864       ['conv3_block6_0_relu[0][0]']    
                                                                                                      
     conv3_block6_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block6_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block6_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block6_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block6_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block6_1_relu[0][0]']    
                                                                                                      
     conv3_block6_concat (Concatena  (None, 28, 28, 320)  0          ['conv3_block5_concat[0][0]',    
     te)                                                              'conv3_block6_2_conv[0][0]']    
                                                                                                      
     conv3_block7_0_bn (BatchNormal  (None, 28, 28, 320)  1280       ['conv3_block6_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block7_0_relu (Activatio  (None, 28, 28, 320)  0          ['conv3_block7_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block7_1_conv (Conv2D)   (None, 28, 28, 128)  40960       ['conv3_block7_0_relu[0][0]']    
                                                                                                      
     conv3_block7_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block7_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block7_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block7_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block7_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block7_1_relu[0][0]']    
                                                                                                      
     conv3_block7_concat (Concatena  (None, 28, 28, 352)  0          ['conv3_block6_concat[0][0]',    
     te)                                                              'conv3_block7_2_conv[0][0]']    
                                                                                                      
     conv3_block8_0_bn (BatchNormal  (None, 28, 28, 352)  1408       ['conv3_block7_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block8_0_relu (Activatio  (None, 28, 28, 352)  0          ['conv3_block8_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block8_1_conv (Conv2D)   (None, 28, 28, 128)  45056       ['conv3_block8_0_relu[0][0]']    
                                                                                                      
     conv3_block8_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block8_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block8_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block8_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block8_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block8_1_relu[0][0]']    
                                                                                                      
     conv3_block8_concat (Concatena  (None, 28, 28, 384)  0          ['conv3_block7_concat[0][0]',    
     te)                                                              'conv3_block8_2_conv[0][0]']    
                                                                                                      
     conv3_block9_0_bn (BatchNormal  (None, 28, 28, 384)  1536       ['conv3_block8_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block9_0_relu (Activatio  (None, 28, 28, 384)  0          ['conv3_block9_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block9_1_conv (Conv2D)   (None, 28, 28, 128)  49152       ['conv3_block9_0_relu[0][0]']    
                                                                                                      
     conv3_block9_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block9_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv3_block9_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block9_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv3_block9_2_conv (Conv2D)   (None, 28, 28, 32)   36864       ['conv3_block9_1_relu[0][0]']    
                                                                                                      
     conv3_block9_concat (Concatena  (None, 28, 28, 416)  0          ['conv3_block8_concat[0][0]',    
     te)                                                              'conv3_block9_2_conv[0][0]']    
                                                                                                      
     conv3_block10_0_bn (BatchNorma  (None, 28, 28, 416)  1664       ['conv3_block9_concat[0][0]']    
     lization)                                                                                        
                                                                                                      
     conv3_block10_0_relu (Activati  (None, 28, 28, 416)  0          ['conv3_block10_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv3_block10_1_conv (Conv2D)  (None, 28, 28, 128)  53248       ['conv3_block10_0_relu[0][0]']   
                                                                                                      
     conv3_block10_1_bn (BatchNorma  (None, 28, 28, 128)  512        ['conv3_block10_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv3_block10_1_relu (Activati  (None, 28, 28, 128)  0          ['conv3_block10_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv3_block10_2_conv (Conv2D)  (None, 28, 28, 32)   36864       ['conv3_block10_1_relu[0][0]']   
                                                                                                      
     conv3_block10_concat (Concaten  (None, 28, 28, 448)  0          ['conv3_block9_concat[0][0]',    
     ate)                                                             'conv3_block10_2_conv[0][0]']   
                                                                                                      
     conv3_block11_0_bn (BatchNorma  (None, 28, 28, 448)  1792       ['conv3_block10_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv3_block11_0_relu (Activati  (None, 28, 28, 448)  0          ['conv3_block11_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv3_block11_1_conv (Conv2D)  (None, 28, 28, 128)  57344       ['conv3_block11_0_relu[0][0]']   
                                                                                                      
     conv3_block11_1_bn (BatchNorma  (None, 28, 28, 128)  512        ['conv3_block11_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv3_block11_1_relu (Activati  (None, 28, 28, 128)  0          ['conv3_block11_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv3_block11_2_conv (Conv2D)  (None, 28, 28, 32)   36864       ['conv3_block11_1_relu[0][0]']   
                                                                                                      
     conv3_block11_concat (Concaten  (None, 28, 28, 480)  0          ['conv3_block10_concat[0][0]',   
     ate)                                                             'conv3_block11_2_conv[0][0]']   
                                                                                                      
     conv3_block12_0_bn (BatchNorma  (None, 28, 28, 480)  1920       ['conv3_block11_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv3_block12_0_relu (Activati  (None, 28, 28, 480)  0          ['conv3_block12_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv3_block12_1_conv (Conv2D)  (None, 28, 28, 128)  61440       ['conv3_block12_0_relu[0][0]']   
                                                                                                      
     conv3_block12_1_bn (BatchNorma  (None, 28, 28, 128)  512        ['conv3_block12_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv3_block12_1_relu (Activati  (None, 28, 28, 128)  0          ['conv3_block12_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv3_block12_2_conv (Conv2D)  (None, 28, 28, 32)   36864       ['conv3_block12_1_relu[0][0]']   
                                                                                                      
     conv3_block12_concat (Concaten  (None, 28, 28, 512)  0          ['conv3_block11_concat[0][0]',   
     ate)                                                             'conv3_block12_2_conv[0][0]']   
                                                                                                      
     pool3_bn (BatchNormalization)  (None, 28, 28, 512)  2048        ['conv3_block12_concat[0][0]']   
                                                                                                      
     pool3_relu (Activation)        (None, 28, 28, 512)  0           ['pool3_bn[0][0]']               
                                                                                                      
     pool3_conv (Conv2D)            (None, 28, 28, 256)  131072      ['pool3_relu[0][0]']             
                                                                                                      
     pool3_pool (AveragePooling2D)  (None, 14, 14, 256)  0           ['pool3_conv[0][0]']             
                                                                                                      
     conv4_block1_0_bn (BatchNormal  (None, 14, 14, 256)  1024       ['pool3_pool[0][0]']             
     ization)                                                                                         
                                                                                                      
     conv4_block1_0_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block1_1_conv (Conv2D)   (None, 14, 14, 128)  32768       ['conv4_block1_0_relu[0][0]']    
                                                                                                      
     conv4_block1_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block1_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block1_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block1_1_relu[0][0]']    
                                                                                                      
     conv4_block1_concat (Concatena  (None, 14, 14, 288)  0          ['pool3_pool[0][0]',             
     te)                                                              'conv4_block1_2_conv[0][0]']    
                                                                                                      
     conv4_block2_0_bn (BatchNormal  (None, 14, 14, 288)  1152       ['conv4_block1_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block2_0_relu (Activatio  (None, 14, 14, 288)  0          ['conv4_block2_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block2_1_conv (Conv2D)   (None, 14, 14, 128)  36864       ['conv4_block2_0_relu[0][0]']    
                                                                                                      
     conv4_block2_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block2_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block2_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block2_1_relu[0][0]']    
                                                                                                      
     conv4_block2_concat (Concatena  (None, 14, 14, 320)  0          ['conv4_block1_concat[0][0]',    
     te)                                                              'conv4_block2_2_conv[0][0]']    
                                                                                                      
     conv4_block3_0_bn (BatchNormal  (None, 14, 14, 320)  1280       ['conv4_block2_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block3_0_relu (Activatio  (None, 14, 14, 320)  0          ['conv4_block3_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block3_1_conv (Conv2D)   (None, 14, 14, 128)  40960       ['conv4_block3_0_relu[0][0]']    
                                                                                                      
     conv4_block3_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block3_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block3_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block3_1_relu[0][0]']    
                                                                                                      
     conv4_block3_concat (Concatena  (None, 14, 14, 352)  0          ['conv4_block2_concat[0][0]',    
     te)                                                              'conv4_block3_2_conv[0][0]']    
                                                                                                      
     conv4_block4_0_bn (BatchNormal  (None, 14, 14, 352)  1408       ['conv4_block3_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block4_0_relu (Activatio  (None, 14, 14, 352)  0          ['conv4_block4_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block4_1_conv (Conv2D)   (None, 14, 14, 128)  45056       ['conv4_block4_0_relu[0][0]']    
                                                                                                      
     conv4_block4_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block4_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block4_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block4_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block4_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block4_1_relu[0][0]']    
                                                                                                      
     conv4_block4_concat (Concatena  (None, 14, 14, 384)  0          ['conv4_block3_concat[0][0]',    
     te)                                                              'conv4_block4_2_conv[0][0]']    
                                                                                                      
     conv4_block5_0_bn (BatchNormal  (None, 14, 14, 384)  1536       ['conv4_block4_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block5_0_relu (Activatio  (None, 14, 14, 384)  0          ['conv4_block5_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block5_1_conv (Conv2D)   (None, 14, 14, 128)  49152       ['conv4_block5_0_relu[0][0]']    
                                                                                                      
     conv4_block5_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block5_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block5_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block5_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block5_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block5_1_relu[0][0]']    
                                                                                                      
     conv4_block5_concat (Concatena  (None, 14, 14, 416)  0          ['conv4_block4_concat[0][0]',    
     te)                                                              'conv4_block5_2_conv[0][0]']    
                                                                                                      
     conv4_block6_0_bn (BatchNormal  (None, 14, 14, 416)  1664       ['conv4_block5_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block6_0_relu (Activatio  (None, 14, 14, 416)  0          ['conv4_block6_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block6_1_conv (Conv2D)   (None, 14, 14, 128)  53248       ['conv4_block6_0_relu[0][0]']    
                                                                                                      
     conv4_block6_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block6_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block6_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block6_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block6_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block6_1_relu[0][0]']    
                                                                                                      
     conv4_block6_concat (Concatena  (None, 14, 14, 448)  0          ['conv4_block5_concat[0][0]',    
     te)                                                              'conv4_block6_2_conv[0][0]']    
                                                                                                      
     conv4_block7_0_bn (BatchNormal  (None, 14, 14, 448)  1792       ['conv4_block6_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block7_0_relu (Activatio  (None, 14, 14, 448)  0          ['conv4_block7_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block7_1_conv (Conv2D)   (None, 14, 14, 128)  57344       ['conv4_block7_0_relu[0][0]']    
                                                                                                      
     conv4_block7_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block7_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block7_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block7_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block7_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block7_1_relu[0][0]']    
                                                                                                      
     conv4_block7_concat (Concatena  (None, 14, 14, 480)  0          ['conv4_block6_concat[0][0]',    
     te)                                                              'conv4_block7_2_conv[0][0]']    
                                                                                                      
     conv4_block8_0_bn (BatchNormal  (None, 14, 14, 480)  1920       ['conv4_block7_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block8_0_relu (Activatio  (None, 14, 14, 480)  0          ['conv4_block8_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block8_1_conv (Conv2D)   (None, 14, 14, 128)  61440       ['conv4_block8_0_relu[0][0]']    
                                                                                                      
     conv4_block8_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block8_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block8_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block8_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block8_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block8_1_relu[0][0]']    
                                                                                                      
     conv4_block8_concat (Concatena  (None, 14, 14, 512)  0          ['conv4_block7_concat[0][0]',    
     te)                                                              'conv4_block8_2_conv[0][0]']    
                                                                                                      
     conv4_block9_0_bn (BatchNormal  (None, 14, 14, 512)  2048       ['conv4_block8_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block9_0_relu (Activatio  (None, 14, 14, 512)  0          ['conv4_block9_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block9_1_conv (Conv2D)   (None, 14, 14, 128)  65536       ['conv4_block9_0_relu[0][0]']    
                                                                                                      
     conv4_block9_1_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv4_block9_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv4_block9_1_relu (Activatio  (None, 14, 14, 128)  0          ['conv4_block9_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv4_block9_2_conv (Conv2D)   (None, 14, 14, 32)   36864       ['conv4_block9_1_relu[0][0]']    
                                                                                                      
     conv4_block9_concat (Concatena  (None, 14, 14, 544)  0          ['conv4_block8_concat[0][0]',    
     te)                                                              'conv4_block9_2_conv[0][0]']    
                                                                                                      
     conv4_block10_0_bn (BatchNorma  (None, 14, 14, 544)  2176       ['conv4_block9_concat[0][0]']    
     lization)                                                                                        
                                                                                                      
     conv4_block10_0_relu (Activati  (None, 14, 14, 544)  0          ['conv4_block10_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block10_1_conv (Conv2D)  (None, 14, 14, 128)  69632       ['conv4_block10_0_relu[0][0]']   
                                                                                                      
     conv4_block10_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block10_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block10_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block10_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block10_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block10_1_relu[0][0]']   
                                                                                                      
     conv4_block10_concat (Concaten  (None, 14, 14, 576)  0          ['conv4_block9_concat[0][0]',    
     ate)                                                             'conv4_block10_2_conv[0][0]']   
                                                                                                      
     conv4_block11_0_bn (BatchNorma  (None, 14, 14, 576)  2304       ['conv4_block10_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block11_0_relu (Activati  (None, 14, 14, 576)  0          ['conv4_block11_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block11_1_conv (Conv2D)  (None, 14, 14, 128)  73728       ['conv4_block11_0_relu[0][0]']   
                                                                                                      
     conv4_block11_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block11_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block11_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block11_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block11_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block11_1_relu[0][0]']   
                                                                                                      
     conv4_block11_concat (Concaten  (None, 14, 14, 608)  0          ['conv4_block10_concat[0][0]',   
     ate)                                                             'conv4_block11_2_conv[0][0]']   
                                                                                                      
     conv4_block12_0_bn (BatchNorma  (None, 14, 14, 608)  2432       ['conv4_block11_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block12_0_relu (Activati  (None, 14, 14, 608)  0          ['conv4_block12_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block12_1_conv (Conv2D)  (None, 14, 14, 128)  77824       ['conv4_block12_0_relu[0][0]']   
                                                                                                      
     conv4_block12_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block12_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block12_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block12_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block12_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block12_1_relu[0][0]']   
                                                                                                      
     conv4_block12_concat (Concaten  (None, 14, 14, 640)  0          ['conv4_block11_concat[0][0]',   
     ate)                                                             'conv4_block12_2_conv[0][0]']   
                                                                                                      
     conv4_block13_0_bn (BatchNorma  (None, 14, 14, 640)  2560       ['conv4_block12_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block13_0_relu (Activati  (None, 14, 14, 640)  0          ['conv4_block13_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block13_1_conv (Conv2D)  (None, 14, 14, 128)  81920       ['conv4_block13_0_relu[0][0]']   
                                                                                                      
     conv4_block13_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block13_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block13_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block13_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block13_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block13_1_relu[0][0]']   
                                                                                                      
     conv4_block13_concat (Concaten  (None, 14, 14, 672)  0          ['conv4_block12_concat[0][0]',   
     ate)                                                             'conv4_block13_2_conv[0][0]']   
                                                                                                      
     conv4_block14_0_bn (BatchNorma  (None, 14, 14, 672)  2688       ['conv4_block13_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block14_0_relu (Activati  (None, 14, 14, 672)  0          ['conv4_block14_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block14_1_conv (Conv2D)  (None, 14, 14, 128)  86016       ['conv4_block14_0_relu[0][0]']   
                                                                                                      
     conv4_block14_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block14_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block14_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block14_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block14_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block14_1_relu[0][0]']   
                                                                                                      
     conv4_block14_concat (Concaten  (None, 14, 14, 704)  0          ['conv4_block13_concat[0][0]',   
     ate)                                                             'conv4_block14_2_conv[0][0]']   
                                                                                                      
     conv4_block15_0_bn (BatchNorma  (None, 14, 14, 704)  2816       ['conv4_block14_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block15_0_relu (Activati  (None, 14, 14, 704)  0          ['conv4_block15_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block15_1_conv (Conv2D)  (None, 14, 14, 128)  90112       ['conv4_block15_0_relu[0][0]']   
                                                                                                      
     conv4_block15_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block15_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block15_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block15_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block15_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block15_1_relu[0][0]']   
                                                                                                      
     conv4_block15_concat (Concaten  (None, 14, 14, 736)  0          ['conv4_block14_concat[0][0]',   
     ate)                                                             'conv4_block15_2_conv[0][0]']   
                                                                                                      
     conv4_block16_0_bn (BatchNorma  (None, 14, 14, 736)  2944       ['conv4_block15_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block16_0_relu (Activati  (None, 14, 14, 736)  0          ['conv4_block16_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block16_1_conv (Conv2D)  (None, 14, 14, 128)  94208       ['conv4_block16_0_relu[0][0]']   
                                                                                                      
     conv4_block16_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block16_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block16_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block16_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block16_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block16_1_relu[0][0]']   
                                                                                                      
     conv4_block16_concat (Concaten  (None, 14, 14, 768)  0          ['conv4_block15_concat[0][0]',   
     ate)                                                             'conv4_block16_2_conv[0][0]']   
                                                                                                      
     conv4_block17_0_bn (BatchNorma  (None, 14, 14, 768)  3072       ['conv4_block16_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block17_0_relu (Activati  (None, 14, 14, 768)  0          ['conv4_block17_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block17_1_conv (Conv2D)  (None, 14, 14, 128)  98304       ['conv4_block17_0_relu[0][0]']   
                                                                                                      
     conv4_block17_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block17_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block17_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block17_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block17_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block17_1_relu[0][0]']   
                                                                                                      
     conv4_block17_concat (Concaten  (None, 14, 14, 800)  0          ['conv4_block16_concat[0][0]',   
     ate)                                                             'conv4_block17_2_conv[0][0]']   
                                                                                                      
     conv4_block18_0_bn (BatchNorma  (None, 14, 14, 800)  3200       ['conv4_block17_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block18_0_relu (Activati  (None, 14, 14, 800)  0          ['conv4_block18_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block18_1_conv (Conv2D)  (None, 14, 14, 128)  102400      ['conv4_block18_0_relu[0][0]']   
                                                                                                      
     conv4_block18_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block18_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block18_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block18_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block18_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block18_1_relu[0][0]']   
                                                                                                      
     conv4_block18_concat (Concaten  (None, 14, 14, 832)  0          ['conv4_block17_concat[0][0]',   
     ate)                                                             'conv4_block18_2_conv[0][0]']   
                                                                                                      
     conv4_block19_0_bn (BatchNorma  (None, 14, 14, 832)  3328       ['conv4_block18_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block19_0_relu (Activati  (None, 14, 14, 832)  0          ['conv4_block19_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block19_1_conv (Conv2D)  (None, 14, 14, 128)  106496      ['conv4_block19_0_relu[0][0]']   
                                                                                                      
     conv4_block19_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block19_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block19_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block19_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block19_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block19_1_relu[0][0]']   
                                                                                                      
     conv4_block19_concat (Concaten  (None, 14, 14, 864)  0          ['conv4_block18_concat[0][0]',   
     ate)                                                             'conv4_block19_2_conv[0][0]']   
                                                                                                      
     conv4_block20_0_bn (BatchNorma  (None, 14, 14, 864)  3456       ['conv4_block19_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block20_0_relu (Activati  (None, 14, 14, 864)  0          ['conv4_block20_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block20_1_conv (Conv2D)  (None, 14, 14, 128)  110592      ['conv4_block20_0_relu[0][0]']   
                                                                                                      
     conv4_block20_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block20_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block20_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block20_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block20_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block20_1_relu[0][0]']   
                                                                                                      
     conv4_block20_concat (Concaten  (None, 14, 14, 896)  0          ['conv4_block19_concat[0][0]',   
     ate)                                                             'conv4_block20_2_conv[0][0]']   
                                                                                                      
     conv4_block21_0_bn (BatchNorma  (None, 14, 14, 896)  3584       ['conv4_block20_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block21_0_relu (Activati  (None, 14, 14, 896)  0          ['conv4_block21_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block21_1_conv (Conv2D)  (None, 14, 14, 128)  114688      ['conv4_block21_0_relu[0][0]']   
                                                                                                      
     conv4_block21_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block21_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block21_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block21_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block21_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block21_1_relu[0][0]']   
                                                                                                      
     conv4_block21_concat (Concaten  (None, 14, 14, 928)  0          ['conv4_block20_concat[0][0]',   
     ate)                                                             'conv4_block21_2_conv[0][0]']   
                                                                                                      
     conv4_block22_0_bn (BatchNorma  (None, 14, 14, 928)  3712       ['conv4_block21_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block22_0_relu (Activati  (None, 14, 14, 928)  0          ['conv4_block22_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block22_1_conv (Conv2D)  (None, 14, 14, 128)  118784      ['conv4_block22_0_relu[0][0]']   
                                                                                                      
     conv4_block22_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block22_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block22_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block22_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block22_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block22_1_relu[0][0]']   
                                                                                                      
     conv4_block22_concat (Concaten  (None, 14, 14, 960)  0          ['conv4_block21_concat[0][0]',   
     ate)                                                             'conv4_block22_2_conv[0][0]']   
                                                                                                      
     conv4_block23_0_bn (BatchNorma  (None, 14, 14, 960)  3840       ['conv4_block22_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block23_0_relu (Activati  (None, 14, 14, 960)  0          ['conv4_block23_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block23_1_conv (Conv2D)  (None, 14, 14, 128)  122880      ['conv4_block23_0_relu[0][0]']   
                                                                                                      
     conv4_block23_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block23_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block23_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block23_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block23_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block23_1_relu[0][0]']   
                                                                                                      
     conv4_block23_concat (Concaten  (None, 14, 14, 992)  0          ['conv4_block22_concat[0][0]',   
     ate)                                                             'conv4_block23_2_conv[0][0]']   
                                                                                                      
     conv4_block24_0_bn (BatchNorma  (None, 14, 14, 992)  3968       ['conv4_block23_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block24_0_relu (Activati  (None, 14, 14, 992)  0          ['conv4_block24_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block24_1_conv (Conv2D)  (None, 14, 14, 128)  126976      ['conv4_block24_0_relu[0][0]']   
                                                                                                      
     conv4_block24_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block24_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block24_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block24_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block24_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block24_1_relu[0][0]']   
                                                                                                      
     conv4_block24_concat (Concaten  (None, 14, 14, 1024  0          ['conv4_block23_concat[0][0]',   
     ate)                           )                                 'conv4_block24_2_conv[0][0]']   
                                                                                                      
     conv4_block25_0_bn (BatchNorma  (None, 14, 14, 1024  4096       ['conv4_block24_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block25_0_relu (Activati  (None, 14, 14, 1024  0          ['conv4_block25_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block25_1_conv (Conv2D)  (None, 14, 14, 128)  131072      ['conv4_block25_0_relu[0][0]']   
                                                                                                      
     conv4_block25_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block25_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block25_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block25_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block25_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block25_1_relu[0][0]']   
                                                                                                      
     conv4_block25_concat (Concaten  (None, 14, 14, 1056  0          ['conv4_block24_concat[0][0]',   
     ate)                           )                                 'conv4_block25_2_conv[0][0]']   
                                                                                                      
     conv4_block26_0_bn (BatchNorma  (None, 14, 14, 1056  4224       ['conv4_block25_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block26_0_relu (Activati  (None, 14, 14, 1056  0          ['conv4_block26_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block26_1_conv (Conv2D)  (None, 14, 14, 128)  135168      ['conv4_block26_0_relu[0][0]']   
                                                                                                      
     conv4_block26_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block26_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block26_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block26_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block26_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block26_1_relu[0][0]']   
                                                                                                      
     conv4_block26_concat (Concaten  (None, 14, 14, 1088  0          ['conv4_block25_concat[0][0]',   
     ate)                           )                                 'conv4_block26_2_conv[0][0]']   
                                                                                                      
     conv4_block27_0_bn (BatchNorma  (None, 14, 14, 1088  4352       ['conv4_block26_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block27_0_relu (Activati  (None, 14, 14, 1088  0          ['conv4_block27_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block27_1_conv (Conv2D)  (None, 14, 14, 128)  139264      ['conv4_block27_0_relu[0][0]']   
                                                                                                      
     conv4_block27_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block27_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block27_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block27_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block27_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block27_1_relu[0][0]']   
                                                                                                      
     conv4_block27_concat (Concaten  (None, 14, 14, 1120  0          ['conv4_block26_concat[0][0]',   
     ate)                           )                                 'conv4_block27_2_conv[0][0]']   
                                                                                                      
     conv4_block28_0_bn (BatchNorma  (None, 14, 14, 1120  4480       ['conv4_block27_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block28_0_relu (Activati  (None, 14, 14, 1120  0          ['conv4_block28_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block28_1_conv (Conv2D)  (None, 14, 14, 128)  143360      ['conv4_block28_0_relu[0][0]']   
                                                                                                      
     conv4_block28_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block28_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block28_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block28_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block28_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block28_1_relu[0][0]']   
                                                                                                      
     conv4_block28_concat (Concaten  (None, 14, 14, 1152  0          ['conv4_block27_concat[0][0]',   
     ate)                           )                                 'conv4_block28_2_conv[0][0]']   
                                                                                                      
     conv4_block29_0_bn (BatchNorma  (None, 14, 14, 1152  4608       ['conv4_block28_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block29_0_relu (Activati  (None, 14, 14, 1152  0          ['conv4_block29_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block29_1_conv (Conv2D)  (None, 14, 14, 128)  147456      ['conv4_block29_0_relu[0][0]']   
                                                                                                      
     conv4_block29_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block29_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block29_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block29_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block29_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block29_1_relu[0][0]']   
                                                                                                      
     conv4_block29_concat (Concaten  (None, 14, 14, 1184  0          ['conv4_block28_concat[0][0]',   
     ate)                           )                                 'conv4_block29_2_conv[0][0]']   
                                                                                                      
     conv4_block30_0_bn (BatchNorma  (None, 14, 14, 1184  4736       ['conv4_block29_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block30_0_relu (Activati  (None, 14, 14, 1184  0          ['conv4_block30_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block30_1_conv (Conv2D)  (None, 14, 14, 128)  151552      ['conv4_block30_0_relu[0][0]']   
                                                                                                      
     conv4_block30_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block30_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block30_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block30_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block30_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block30_1_relu[0][0]']   
                                                                                                      
     conv4_block30_concat (Concaten  (None, 14, 14, 1216  0          ['conv4_block29_concat[0][0]',   
     ate)                           )                                 'conv4_block30_2_conv[0][0]']   
                                                                                                      
     conv4_block31_0_bn (BatchNorma  (None, 14, 14, 1216  4864       ['conv4_block30_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block31_0_relu (Activati  (None, 14, 14, 1216  0          ['conv4_block31_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block31_1_conv (Conv2D)  (None, 14, 14, 128)  155648      ['conv4_block31_0_relu[0][0]']   
                                                                                                      
     conv4_block31_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block31_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block31_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block31_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block31_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block31_1_relu[0][0]']   
                                                                                                      
     conv4_block31_concat (Concaten  (None, 14, 14, 1248  0          ['conv4_block30_concat[0][0]',   
     ate)                           )                                 'conv4_block31_2_conv[0][0]']   
                                                                                                      
     conv4_block32_0_bn (BatchNorma  (None, 14, 14, 1248  4992       ['conv4_block31_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block32_0_relu (Activati  (None, 14, 14, 1248  0          ['conv4_block32_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block32_1_conv (Conv2D)  (None, 14, 14, 128)  159744      ['conv4_block32_0_relu[0][0]']   
                                                                                                      
     conv4_block32_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block32_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block32_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block32_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block32_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block32_1_relu[0][0]']   
                                                                                                      
     conv4_block32_concat (Concaten  (None, 14, 14, 1280  0          ['conv4_block31_concat[0][0]',   
     ate)                           )                                 'conv4_block32_2_conv[0][0]']   
                                                                                                      
     conv4_block33_0_bn (BatchNorma  (None, 14, 14, 1280  5120       ['conv4_block32_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block33_0_relu (Activati  (None, 14, 14, 1280  0          ['conv4_block33_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block33_1_conv (Conv2D)  (None, 14, 14, 128)  163840      ['conv4_block33_0_relu[0][0]']   
                                                                                                      
     conv4_block33_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block33_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block33_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block33_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block33_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block33_1_relu[0][0]']   
                                                                                                      
     conv4_block33_concat (Concaten  (None, 14, 14, 1312  0          ['conv4_block32_concat[0][0]',   
     ate)                           )                                 'conv4_block33_2_conv[0][0]']   
                                                                                                      
     conv4_block34_0_bn (BatchNorma  (None, 14, 14, 1312  5248       ['conv4_block33_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block34_0_relu (Activati  (None, 14, 14, 1312  0          ['conv4_block34_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block34_1_conv (Conv2D)  (None, 14, 14, 128)  167936      ['conv4_block34_0_relu[0][0]']   
                                                                                                      
     conv4_block34_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block34_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block34_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block34_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block34_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block34_1_relu[0][0]']   
                                                                                                      
     conv4_block34_concat (Concaten  (None, 14, 14, 1344  0          ['conv4_block33_concat[0][0]',   
     ate)                           )                                 'conv4_block34_2_conv[0][0]']   
                                                                                                      
     conv4_block35_0_bn (BatchNorma  (None, 14, 14, 1344  5376       ['conv4_block34_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block35_0_relu (Activati  (None, 14, 14, 1344  0          ['conv4_block35_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block35_1_conv (Conv2D)  (None, 14, 14, 128)  172032      ['conv4_block35_0_relu[0][0]']   
                                                                                                      
     conv4_block35_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block35_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block35_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block35_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block35_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block35_1_relu[0][0]']   
                                                                                                      
     conv4_block35_concat (Concaten  (None, 14, 14, 1376  0          ['conv4_block34_concat[0][0]',   
     ate)                           )                                 'conv4_block35_2_conv[0][0]']   
                                                                                                      
     conv4_block36_0_bn (BatchNorma  (None, 14, 14, 1376  5504       ['conv4_block35_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block36_0_relu (Activati  (None, 14, 14, 1376  0          ['conv4_block36_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block36_1_conv (Conv2D)  (None, 14, 14, 128)  176128      ['conv4_block36_0_relu[0][0]']   
                                                                                                      
     conv4_block36_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block36_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block36_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block36_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block36_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block36_1_relu[0][0]']   
                                                                                                      
     conv4_block36_concat (Concaten  (None, 14, 14, 1408  0          ['conv4_block35_concat[0][0]',   
     ate)                           )                                 'conv4_block36_2_conv[0][0]']   
                                                                                                      
     conv4_block37_0_bn (BatchNorma  (None, 14, 14, 1408  5632       ['conv4_block36_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block37_0_relu (Activati  (None, 14, 14, 1408  0          ['conv4_block37_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block37_1_conv (Conv2D)  (None, 14, 14, 128)  180224      ['conv4_block37_0_relu[0][0]']   
                                                                                                      
     conv4_block37_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block37_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block37_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block37_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block37_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block37_1_relu[0][0]']   
                                                                                                      
     conv4_block37_concat (Concaten  (None, 14, 14, 1440  0          ['conv4_block36_concat[0][0]',   
     ate)                           )                                 'conv4_block37_2_conv[0][0]']   
                                                                                                      
     conv4_block38_0_bn (BatchNorma  (None, 14, 14, 1440  5760       ['conv4_block37_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block38_0_relu (Activati  (None, 14, 14, 1440  0          ['conv4_block38_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block38_1_conv (Conv2D)  (None, 14, 14, 128)  184320      ['conv4_block38_0_relu[0][0]']   
                                                                                                      
     conv4_block38_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block38_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block38_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block38_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block38_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block38_1_relu[0][0]']   
                                                                                                      
     conv4_block38_concat (Concaten  (None, 14, 14, 1472  0          ['conv4_block37_concat[0][0]',   
     ate)                           )                                 'conv4_block38_2_conv[0][0]']   
                                                                                                      
     conv4_block39_0_bn (BatchNorma  (None, 14, 14, 1472  5888       ['conv4_block38_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block39_0_relu (Activati  (None, 14, 14, 1472  0          ['conv4_block39_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block39_1_conv (Conv2D)  (None, 14, 14, 128)  188416      ['conv4_block39_0_relu[0][0]']   
                                                                                                      
     conv4_block39_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block39_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block39_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block39_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block39_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block39_1_relu[0][0]']   
                                                                                                      
     conv4_block39_concat (Concaten  (None, 14, 14, 1504  0          ['conv4_block38_concat[0][0]',   
     ate)                           )                                 'conv4_block39_2_conv[0][0]']   
                                                                                                      
     conv4_block40_0_bn (BatchNorma  (None, 14, 14, 1504  6016       ['conv4_block39_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block40_0_relu (Activati  (None, 14, 14, 1504  0          ['conv4_block40_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block40_1_conv (Conv2D)  (None, 14, 14, 128)  192512      ['conv4_block40_0_relu[0][0]']   
                                                                                                      
     conv4_block40_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block40_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block40_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block40_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block40_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block40_1_relu[0][0]']   
                                                                                                      
     conv4_block40_concat (Concaten  (None, 14, 14, 1536  0          ['conv4_block39_concat[0][0]',   
     ate)                           )                                 'conv4_block40_2_conv[0][0]']   
                                                                                                      
     conv4_block41_0_bn (BatchNorma  (None, 14, 14, 1536  6144       ['conv4_block40_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block41_0_relu (Activati  (None, 14, 14, 1536  0          ['conv4_block41_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block41_1_conv (Conv2D)  (None, 14, 14, 128)  196608      ['conv4_block41_0_relu[0][0]']   
                                                                                                      
     conv4_block41_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block41_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block41_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block41_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block41_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block41_1_relu[0][0]']   
                                                                                                      
     conv4_block41_concat (Concaten  (None, 14, 14, 1568  0          ['conv4_block40_concat[0][0]',   
     ate)                           )                                 'conv4_block41_2_conv[0][0]']   
                                                                                                      
     conv4_block42_0_bn (BatchNorma  (None, 14, 14, 1568  6272       ['conv4_block41_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block42_0_relu (Activati  (None, 14, 14, 1568  0          ['conv4_block42_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block42_1_conv (Conv2D)  (None, 14, 14, 128)  200704      ['conv4_block42_0_relu[0][0]']   
                                                                                                      
     conv4_block42_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block42_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block42_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block42_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block42_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block42_1_relu[0][0]']   
                                                                                                      
     conv4_block42_concat (Concaten  (None, 14, 14, 1600  0          ['conv4_block41_concat[0][0]',   
     ate)                           )                                 'conv4_block42_2_conv[0][0]']   
                                                                                                      
     conv4_block43_0_bn (BatchNorma  (None, 14, 14, 1600  6400       ['conv4_block42_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block43_0_relu (Activati  (None, 14, 14, 1600  0          ['conv4_block43_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block43_1_conv (Conv2D)  (None, 14, 14, 128)  204800      ['conv4_block43_0_relu[0][0]']   
                                                                                                      
     conv4_block43_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block43_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block43_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block43_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block43_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block43_1_relu[0][0]']   
                                                                                                      
     conv4_block43_concat (Concaten  (None, 14, 14, 1632  0          ['conv4_block42_concat[0][0]',   
     ate)                           )                                 'conv4_block43_2_conv[0][0]']   
                                                                                                      
     conv4_block44_0_bn (BatchNorma  (None, 14, 14, 1632  6528       ['conv4_block43_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block44_0_relu (Activati  (None, 14, 14, 1632  0          ['conv4_block44_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block44_1_conv (Conv2D)  (None, 14, 14, 128)  208896      ['conv4_block44_0_relu[0][0]']   
                                                                                                      
     conv4_block44_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block44_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block44_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block44_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block44_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block44_1_relu[0][0]']   
                                                                                                      
     conv4_block44_concat (Concaten  (None, 14, 14, 1664  0          ['conv4_block43_concat[0][0]',   
     ate)                           )                                 'conv4_block44_2_conv[0][0]']   
                                                                                                      
     conv4_block45_0_bn (BatchNorma  (None, 14, 14, 1664  6656       ['conv4_block44_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block45_0_relu (Activati  (None, 14, 14, 1664  0          ['conv4_block45_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block45_1_conv (Conv2D)  (None, 14, 14, 128)  212992      ['conv4_block45_0_relu[0][0]']   
                                                                                                      
     conv4_block45_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block45_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block45_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block45_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block45_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block45_1_relu[0][0]']   
                                                                                                      
     conv4_block45_concat (Concaten  (None, 14, 14, 1696  0          ['conv4_block44_concat[0][0]',   
     ate)                           )                                 'conv4_block45_2_conv[0][0]']   
                                                                                                      
     conv4_block46_0_bn (BatchNorma  (None, 14, 14, 1696  6784       ['conv4_block45_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block46_0_relu (Activati  (None, 14, 14, 1696  0          ['conv4_block46_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block46_1_conv (Conv2D)  (None, 14, 14, 128)  217088      ['conv4_block46_0_relu[0][0]']   
                                                                                                      
     conv4_block46_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block46_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block46_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block46_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block46_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block46_1_relu[0][0]']   
                                                                                                      
     conv4_block46_concat (Concaten  (None, 14, 14, 1728  0          ['conv4_block45_concat[0][0]',   
     ate)                           )                                 'conv4_block46_2_conv[0][0]']   
                                                                                                      
     conv4_block47_0_bn (BatchNorma  (None, 14, 14, 1728  6912       ['conv4_block46_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block47_0_relu (Activati  (None, 14, 14, 1728  0          ['conv4_block47_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block47_1_conv (Conv2D)  (None, 14, 14, 128)  221184      ['conv4_block47_0_relu[0][0]']   
                                                                                                      
     conv4_block47_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block47_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block47_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block47_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block47_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block47_1_relu[0][0]']   
                                                                                                      
     conv4_block47_concat (Concaten  (None, 14, 14, 1760  0          ['conv4_block46_concat[0][0]',   
     ate)                           )                                 'conv4_block47_2_conv[0][0]']   
                                                                                                      
     conv4_block48_0_bn (BatchNorma  (None, 14, 14, 1760  7040       ['conv4_block47_concat[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     conv4_block48_0_relu (Activati  (None, 14, 14, 1760  0          ['conv4_block48_0_bn[0][0]']     
     on)                            )                                                                 
                                                                                                      
     conv4_block48_1_conv (Conv2D)  (None, 14, 14, 128)  225280      ['conv4_block48_0_relu[0][0]']   
                                                                                                      
     conv4_block48_1_bn (BatchNorma  (None, 14, 14, 128)  512        ['conv4_block48_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv4_block48_1_relu (Activati  (None, 14, 14, 128)  0          ['conv4_block48_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv4_block48_2_conv (Conv2D)  (None, 14, 14, 32)   36864       ['conv4_block48_1_relu[0][0]']   
                                                                                                      
     conv4_block48_concat (Concaten  (None, 14, 14, 1792  0          ['conv4_block47_concat[0][0]',   
     ate)                           )                                 'conv4_block48_2_conv[0][0]']   
                                                                                                      
     pool4_bn (BatchNormalization)  (None, 14, 14, 1792  7168        ['conv4_block48_concat[0][0]']   
                                    )                                                                 
                                                                                                      
     pool4_relu (Activation)        (None, 14, 14, 1792  0           ['pool4_bn[0][0]']               
                                    )                                                                 
                                                                                                      
     pool4_conv (Conv2D)            (None, 14, 14, 896)  1605632     ['pool4_relu[0][0]']             
                                                                                                      
     pool4_pool (AveragePooling2D)  (None, 7, 7, 896)    0           ['pool4_conv[0][0]']             
                                                                                                      
     conv5_block1_0_bn (BatchNormal  (None, 7, 7, 896)   3584        ['pool4_pool[0][0]']             
     ization)                                                                                         
                                                                                                      
     conv5_block1_0_relu (Activatio  (None, 7, 7, 896)   0           ['conv5_block1_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block1_1_conv (Conv2D)   (None, 7, 7, 128)    114688      ['conv5_block1_0_relu[0][0]']    
                                                                                                      
     conv5_block1_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block1_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block1_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block1_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block1_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block1_1_relu[0][0]']    
                                                                                                      
     conv5_block1_concat (Concatena  (None, 7, 7, 928)   0           ['pool4_pool[0][0]',             
     te)                                                              'conv5_block1_2_conv[0][0]']    
                                                                                                      
     conv5_block2_0_bn (BatchNormal  (None, 7, 7, 928)   3712        ['conv5_block1_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block2_0_relu (Activatio  (None, 7, 7, 928)   0           ['conv5_block2_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block2_1_conv (Conv2D)   (None, 7, 7, 128)    118784      ['conv5_block2_0_relu[0][0]']    
                                                                                                      
     conv5_block2_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block2_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block2_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block2_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block2_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block2_1_relu[0][0]']    
                                                                                                      
     conv5_block2_concat (Concatena  (None, 7, 7, 960)   0           ['conv5_block1_concat[0][0]',    
     te)                                                              'conv5_block2_2_conv[0][0]']    
                                                                                                      
     conv5_block3_0_bn (BatchNormal  (None, 7, 7, 960)   3840        ['conv5_block2_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block3_0_relu (Activatio  (None, 7, 7, 960)   0           ['conv5_block3_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block3_1_conv (Conv2D)   (None, 7, 7, 128)    122880      ['conv5_block3_0_relu[0][0]']    
                                                                                                      
     conv5_block3_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block3_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block3_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block3_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block3_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block3_1_relu[0][0]']    
                                                                                                      
     conv5_block3_concat (Concatena  (None, 7, 7, 992)   0           ['conv5_block2_concat[0][0]',    
     te)                                                              'conv5_block3_2_conv[0][0]']    
                                                                                                      
     conv5_block4_0_bn (BatchNormal  (None, 7, 7, 992)   3968        ['conv5_block3_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block4_0_relu (Activatio  (None, 7, 7, 992)   0           ['conv5_block4_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block4_1_conv (Conv2D)   (None, 7, 7, 128)    126976      ['conv5_block4_0_relu[0][0]']    
                                                                                                      
     conv5_block4_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block4_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block4_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block4_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block4_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block4_1_relu[0][0]']    
                                                                                                      
     conv5_block4_concat (Concatena  (None, 7, 7, 1024)  0           ['conv5_block3_concat[0][0]',    
     te)                                                              'conv5_block4_2_conv[0][0]']    
                                                                                                      
     conv5_block5_0_bn (BatchNormal  (None, 7, 7, 1024)  4096        ['conv5_block4_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block5_0_relu (Activatio  (None, 7, 7, 1024)  0           ['conv5_block5_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block5_1_conv (Conv2D)   (None, 7, 7, 128)    131072      ['conv5_block5_0_relu[0][0]']    
                                                                                                      
     conv5_block5_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block5_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block5_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block5_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block5_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block5_1_relu[0][0]']    
                                                                                                      
     conv5_block5_concat (Concatena  (None, 7, 7, 1056)  0           ['conv5_block4_concat[0][0]',    
     te)                                                              'conv5_block5_2_conv[0][0]']    
                                                                                                      
     conv5_block6_0_bn (BatchNormal  (None, 7, 7, 1056)  4224        ['conv5_block5_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block6_0_relu (Activatio  (None, 7, 7, 1056)  0           ['conv5_block6_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block6_1_conv (Conv2D)   (None, 7, 7, 128)    135168      ['conv5_block6_0_relu[0][0]']    
                                                                                                      
     conv5_block6_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block6_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block6_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block6_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block6_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block6_1_relu[0][0]']    
                                                                                                      
     conv5_block6_concat (Concatena  (None, 7, 7, 1088)  0           ['conv5_block5_concat[0][0]',    
     te)                                                              'conv5_block6_2_conv[0][0]']    
                                                                                                      
     conv5_block7_0_bn (BatchNormal  (None, 7, 7, 1088)  4352        ['conv5_block6_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block7_0_relu (Activatio  (None, 7, 7, 1088)  0           ['conv5_block7_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block7_1_conv (Conv2D)   (None, 7, 7, 128)    139264      ['conv5_block7_0_relu[0][0]']    
                                                                                                      
     conv5_block7_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block7_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block7_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block7_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block7_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block7_1_relu[0][0]']    
                                                                                                      
     conv5_block7_concat (Concatena  (None, 7, 7, 1120)  0           ['conv5_block6_concat[0][0]',    
     te)                                                              'conv5_block7_2_conv[0][0]']    
                                                                                                      
     conv5_block8_0_bn (BatchNormal  (None, 7, 7, 1120)  4480        ['conv5_block7_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block8_0_relu (Activatio  (None, 7, 7, 1120)  0           ['conv5_block8_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block8_1_conv (Conv2D)   (None, 7, 7, 128)    143360      ['conv5_block8_0_relu[0][0]']    
                                                                                                      
     conv5_block8_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block8_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block8_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block8_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block8_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block8_1_relu[0][0]']    
                                                                                                      
     conv5_block8_concat (Concatena  (None, 7, 7, 1152)  0           ['conv5_block7_concat[0][0]',    
     te)                                                              'conv5_block8_2_conv[0][0]']    
                                                                                                      
     conv5_block9_0_bn (BatchNormal  (None, 7, 7, 1152)  4608        ['conv5_block8_concat[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block9_0_relu (Activatio  (None, 7, 7, 1152)  0           ['conv5_block9_0_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block9_1_conv (Conv2D)   (None, 7, 7, 128)    147456      ['conv5_block9_0_relu[0][0]']    
                                                                                                      
     conv5_block9_1_bn (BatchNormal  (None, 7, 7, 128)   512         ['conv5_block9_1_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     conv5_block9_1_relu (Activatio  (None, 7, 7, 128)   0           ['conv5_block9_1_bn[0][0]']      
     n)                                                                                               
                                                                                                      
     conv5_block9_2_conv (Conv2D)   (None, 7, 7, 32)     36864       ['conv5_block9_1_relu[0][0]']    
                                                                                                      
     conv5_block9_concat (Concatena  (None, 7, 7, 1184)  0           ['conv5_block8_concat[0][0]',    
     te)                                                              'conv5_block9_2_conv[0][0]']    
                                                                                                      
     conv5_block10_0_bn (BatchNorma  (None, 7, 7, 1184)  4736        ['conv5_block9_concat[0][0]']    
     lization)                                                                                        
                                                                                                      
     conv5_block10_0_relu (Activati  (None, 7, 7, 1184)  0           ['conv5_block10_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block10_1_conv (Conv2D)  (None, 7, 7, 128)    151552      ['conv5_block10_0_relu[0][0]']   
                                                                                                      
     conv5_block10_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block10_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block10_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block10_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block10_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block10_1_relu[0][0]']   
                                                                                                      
     conv5_block10_concat (Concaten  (None, 7, 7, 1216)  0           ['conv5_block9_concat[0][0]',    
     ate)                                                             'conv5_block10_2_conv[0][0]']   
                                                                                                      
     conv5_block11_0_bn (BatchNorma  (None, 7, 7, 1216)  4864        ['conv5_block10_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block11_0_relu (Activati  (None, 7, 7, 1216)  0           ['conv5_block11_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block11_1_conv (Conv2D)  (None, 7, 7, 128)    155648      ['conv5_block11_0_relu[0][0]']   
                                                                                                      
     conv5_block11_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block11_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block11_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block11_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block11_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block11_1_relu[0][0]']   
                                                                                                      
     conv5_block11_concat (Concaten  (None, 7, 7, 1248)  0           ['conv5_block10_concat[0][0]',   
     ate)                                                             'conv5_block11_2_conv[0][0]']   
                                                                                                      
     conv5_block12_0_bn (BatchNorma  (None, 7, 7, 1248)  4992        ['conv5_block11_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block12_0_relu (Activati  (None, 7, 7, 1248)  0           ['conv5_block12_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block12_1_conv (Conv2D)  (None, 7, 7, 128)    159744      ['conv5_block12_0_relu[0][0]']   
                                                                                                      
     conv5_block12_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block12_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block12_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block12_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block12_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block12_1_relu[0][0]']   
                                                                                                      
     conv5_block12_concat (Concaten  (None, 7, 7, 1280)  0           ['conv5_block11_concat[0][0]',   
     ate)                                                             'conv5_block12_2_conv[0][0]']   
                                                                                                      
     conv5_block13_0_bn (BatchNorma  (None, 7, 7, 1280)  5120        ['conv5_block12_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block13_0_relu (Activati  (None, 7, 7, 1280)  0           ['conv5_block13_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block13_1_conv (Conv2D)  (None, 7, 7, 128)    163840      ['conv5_block13_0_relu[0][0]']   
                                                                                                      
     conv5_block13_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block13_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block13_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block13_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block13_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block13_1_relu[0][0]']   
                                                                                                      
     conv5_block13_concat (Concaten  (None, 7, 7, 1312)  0           ['conv5_block12_concat[0][0]',   
     ate)                                                             'conv5_block13_2_conv[0][0]']   
                                                                                                      
     conv5_block14_0_bn (BatchNorma  (None, 7, 7, 1312)  5248        ['conv5_block13_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block14_0_relu (Activati  (None, 7, 7, 1312)  0           ['conv5_block14_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block14_1_conv (Conv2D)  (None, 7, 7, 128)    167936      ['conv5_block14_0_relu[0][0]']   
                                                                                                      
     conv5_block14_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block14_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block14_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block14_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block14_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block14_1_relu[0][0]']   
                                                                                                      
     conv5_block14_concat (Concaten  (None, 7, 7, 1344)  0           ['conv5_block13_concat[0][0]',   
     ate)                                                             'conv5_block14_2_conv[0][0]']   
                                                                                                      
     conv5_block15_0_bn (BatchNorma  (None, 7, 7, 1344)  5376        ['conv5_block14_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block15_0_relu (Activati  (None, 7, 7, 1344)  0           ['conv5_block15_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block15_1_conv (Conv2D)  (None, 7, 7, 128)    172032      ['conv5_block15_0_relu[0][0]']   
                                                                                                      
     conv5_block15_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block15_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block15_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block15_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block15_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block15_1_relu[0][0]']   
                                                                                                      
     conv5_block15_concat (Concaten  (None, 7, 7, 1376)  0           ['conv5_block14_concat[0][0]',   
     ate)                                                             'conv5_block15_2_conv[0][0]']   
                                                                                                      
     conv5_block16_0_bn (BatchNorma  (None, 7, 7, 1376)  5504        ['conv5_block15_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block16_0_relu (Activati  (None, 7, 7, 1376)  0           ['conv5_block16_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block16_1_conv (Conv2D)  (None, 7, 7, 128)    176128      ['conv5_block16_0_relu[0][0]']   
                                                                                                      
     conv5_block16_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block16_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block16_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block16_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block16_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block16_1_relu[0][0]']   
                                                                                                      
     conv5_block16_concat (Concaten  (None, 7, 7, 1408)  0           ['conv5_block15_concat[0][0]',   
     ate)                                                             'conv5_block16_2_conv[0][0]']   
                                                                                                      
     conv5_block17_0_bn (BatchNorma  (None, 7, 7, 1408)  5632        ['conv5_block16_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block17_0_relu (Activati  (None, 7, 7, 1408)  0           ['conv5_block17_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block17_1_conv (Conv2D)  (None, 7, 7, 128)    180224      ['conv5_block17_0_relu[0][0]']   
                                                                                                      
     conv5_block17_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block17_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block17_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block17_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block17_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block17_1_relu[0][0]']   
                                                                                                      
     conv5_block17_concat (Concaten  (None, 7, 7, 1440)  0           ['conv5_block16_concat[0][0]',   
     ate)                                                             'conv5_block17_2_conv[0][0]']   
                                                                                                      
     conv5_block18_0_bn (BatchNorma  (None, 7, 7, 1440)  5760        ['conv5_block17_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block18_0_relu (Activati  (None, 7, 7, 1440)  0           ['conv5_block18_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block18_1_conv (Conv2D)  (None, 7, 7, 128)    184320      ['conv5_block18_0_relu[0][0]']   
                                                                                                      
     conv5_block18_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block18_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block18_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block18_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block18_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block18_1_relu[0][0]']   
                                                                                                      
     conv5_block18_concat (Concaten  (None, 7, 7, 1472)  0           ['conv5_block17_concat[0][0]',   
     ate)                                                             'conv5_block18_2_conv[0][0]']   
                                                                                                      
     conv5_block19_0_bn (BatchNorma  (None, 7, 7, 1472)  5888        ['conv5_block18_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block19_0_relu (Activati  (None, 7, 7, 1472)  0           ['conv5_block19_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block19_1_conv (Conv2D)  (None, 7, 7, 128)    188416      ['conv5_block19_0_relu[0][0]']   
                                                                                                      
     conv5_block19_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block19_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block19_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block19_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block19_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block19_1_relu[0][0]']   
                                                                                                      
     conv5_block19_concat (Concaten  (None, 7, 7, 1504)  0           ['conv5_block18_concat[0][0]',   
     ate)                                                             'conv5_block19_2_conv[0][0]']   
                                                                                                      
     conv5_block20_0_bn (BatchNorma  (None, 7, 7, 1504)  6016        ['conv5_block19_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block20_0_relu (Activati  (None, 7, 7, 1504)  0           ['conv5_block20_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block20_1_conv (Conv2D)  (None, 7, 7, 128)    192512      ['conv5_block20_0_relu[0][0]']   
                                                                                                      
     conv5_block20_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block20_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block20_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block20_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block20_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block20_1_relu[0][0]']   
                                                                                                      
     conv5_block20_concat (Concaten  (None, 7, 7, 1536)  0           ['conv5_block19_concat[0][0]',   
     ate)                                                             'conv5_block20_2_conv[0][0]']   
                                                                                                      
     conv5_block21_0_bn (BatchNorma  (None, 7, 7, 1536)  6144        ['conv5_block20_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block21_0_relu (Activati  (None, 7, 7, 1536)  0           ['conv5_block21_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block21_1_conv (Conv2D)  (None, 7, 7, 128)    196608      ['conv5_block21_0_relu[0][0]']   
                                                                                                      
     conv5_block21_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block21_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block21_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block21_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block21_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block21_1_relu[0][0]']   
                                                                                                      
     conv5_block21_concat (Concaten  (None, 7, 7, 1568)  0           ['conv5_block20_concat[0][0]',   
     ate)                                                             'conv5_block21_2_conv[0][0]']   
                                                                                                      
     conv5_block22_0_bn (BatchNorma  (None, 7, 7, 1568)  6272        ['conv5_block21_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block22_0_relu (Activati  (None, 7, 7, 1568)  0           ['conv5_block22_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block22_1_conv (Conv2D)  (None, 7, 7, 128)    200704      ['conv5_block22_0_relu[0][0]']   
                                                                                                      
     conv5_block22_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block22_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block22_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block22_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block22_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block22_1_relu[0][0]']   
                                                                                                      
     conv5_block22_concat (Concaten  (None, 7, 7, 1600)  0           ['conv5_block21_concat[0][0]',   
     ate)                                                             'conv5_block22_2_conv[0][0]']   
                                                                                                      
     conv5_block23_0_bn (BatchNorma  (None, 7, 7, 1600)  6400        ['conv5_block22_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block23_0_relu (Activati  (None, 7, 7, 1600)  0           ['conv5_block23_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block23_1_conv (Conv2D)  (None, 7, 7, 128)    204800      ['conv5_block23_0_relu[0][0]']   
                                                                                                      
     conv5_block23_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block23_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block23_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block23_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block23_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block23_1_relu[0][0]']   
                                                                                                      
     conv5_block23_concat (Concaten  (None, 7, 7, 1632)  0           ['conv5_block22_concat[0][0]',   
     ate)                                                             'conv5_block23_2_conv[0][0]']   
                                                                                                      
     conv5_block24_0_bn (BatchNorma  (None, 7, 7, 1632)  6528        ['conv5_block23_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block24_0_relu (Activati  (None, 7, 7, 1632)  0           ['conv5_block24_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block24_1_conv (Conv2D)  (None, 7, 7, 128)    208896      ['conv5_block24_0_relu[0][0]']   
                                                                                                      
     conv5_block24_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block24_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block24_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block24_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block24_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block24_1_relu[0][0]']   
                                                                                                      
     conv5_block24_concat (Concaten  (None, 7, 7, 1664)  0           ['conv5_block23_concat[0][0]',   
     ate)                                                             'conv5_block24_2_conv[0][0]']   
                                                                                                      
     conv5_block25_0_bn (BatchNorma  (None, 7, 7, 1664)  6656        ['conv5_block24_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block25_0_relu (Activati  (None, 7, 7, 1664)  0           ['conv5_block25_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block25_1_conv (Conv2D)  (None, 7, 7, 128)    212992      ['conv5_block25_0_relu[0][0]']   
                                                                                                      
     conv5_block25_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block25_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block25_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block25_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block25_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block25_1_relu[0][0]']   
                                                                                                      
     conv5_block25_concat (Concaten  (None, 7, 7, 1696)  0           ['conv5_block24_concat[0][0]',   
     ate)                                                             'conv5_block25_2_conv[0][0]']   
                                                                                                      
     conv5_block26_0_bn (BatchNorma  (None, 7, 7, 1696)  6784        ['conv5_block25_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block26_0_relu (Activati  (None, 7, 7, 1696)  0           ['conv5_block26_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block26_1_conv (Conv2D)  (None, 7, 7, 128)    217088      ['conv5_block26_0_relu[0][0]']   
                                                                                                      
     conv5_block26_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block26_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block26_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block26_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block26_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block26_1_relu[0][0]']   
                                                                                                      
     conv5_block26_concat (Concaten  (None, 7, 7, 1728)  0           ['conv5_block25_concat[0][0]',   
     ate)                                                             'conv5_block26_2_conv[0][0]']   
                                                                                                      
     conv5_block27_0_bn (BatchNorma  (None, 7, 7, 1728)  6912        ['conv5_block26_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block27_0_relu (Activati  (None, 7, 7, 1728)  0           ['conv5_block27_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block27_1_conv (Conv2D)  (None, 7, 7, 128)    221184      ['conv5_block27_0_relu[0][0]']   
                                                                                                      
     conv5_block27_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block27_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block27_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block27_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block27_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block27_1_relu[0][0]']   
                                                                                                      
     conv5_block27_concat (Concaten  (None, 7, 7, 1760)  0           ['conv5_block26_concat[0][0]',   
     ate)                                                             'conv5_block27_2_conv[0][0]']   
                                                                                                      
     conv5_block28_0_bn (BatchNorma  (None, 7, 7, 1760)  7040        ['conv5_block27_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block28_0_relu (Activati  (None, 7, 7, 1760)  0           ['conv5_block28_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block28_1_conv (Conv2D)  (None, 7, 7, 128)    225280      ['conv5_block28_0_relu[0][0]']   
                                                                                                      
     conv5_block28_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block28_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block28_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block28_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block28_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block28_1_relu[0][0]']   
                                                                                                      
     conv5_block28_concat (Concaten  (None, 7, 7, 1792)  0           ['conv5_block27_concat[0][0]',   
     ate)                                                             'conv5_block28_2_conv[0][0]']   
                                                                                                      
     conv5_block29_0_bn (BatchNorma  (None, 7, 7, 1792)  7168        ['conv5_block28_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block29_0_relu (Activati  (None, 7, 7, 1792)  0           ['conv5_block29_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block29_1_conv (Conv2D)  (None, 7, 7, 128)    229376      ['conv5_block29_0_relu[0][0]']   
                                                                                                      
     conv5_block29_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block29_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block29_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block29_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block29_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block29_1_relu[0][0]']   
                                                                                                      
     conv5_block29_concat (Concaten  (None, 7, 7, 1824)  0           ['conv5_block28_concat[0][0]',   
     ate)                                                             'conv5_block29_2_conv[0][0]']   
                                                                                                      
     conv5_block30_0_bn (BatchNorma  (None, 7, 7, 1824)  7296        ['conv5_block29_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block30_0_relu (Activati  (None, 7, 7, 1824)  0           ['conv5_block30_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block30_1_conv (Conv2D)  (None, 7, 7, 128)    233472      ['conv5_block30_0_relu[0][0]']   
                                                                                                      
     conv5_block30_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block30_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block30_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block30_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block30_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block30_1_relu[0][0]']   
                                                                                                      
     conv5_block30_concat (Concaten  (None, 7, 7, 1856)  0           ['conv5_block29_concat[0][0]',   
     ate)                                                             'conv5_block30_2_conv[0][0]']   
                                                                                                      
     conv5_block31_0_bn (BatchNorma  (None, 7, 7, 1856)  7424        ['conv5_block30_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block31_0_relu (Activati  (None, 7, 7, 1856)  0           ['conv5_block31_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block31_1_conv (Conv2D)  (None, 7, 7, 128)    237568      ['conv5_block31_0_relu[0][0]']   
                                                                                                      
     conv5_block31_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block31_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block31_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block31_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block31_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block31_1_relu[0][0]']   
                                                                                                      
     conv5_block31_concat (Concaten  (None, 7, 7, 1888)  0           ['conv5_block30_concat[0][0]',   
     ate)                                                             'conv5_block31_2_conv[0][0]']   
                                                                                                      
     conv5_block32_0_bn (BatchNorma  (None, 7, 7, 1888)  7552        ['conv5_block31_concat[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block32_0_relu (Activati  (None, 7, 7, 1888)  0           ['conv5_block32_0_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block32_1_conv (Conv2D)  (None, 7, 7, 128)    241664      ['conv5_block32_0_relu[0][0]']   
                                                                                                      
     conv5_block32_1_bn (BatchNorma  (None, 7, 7, 128)   512         ['conv5_block32_1_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     conv5_block32_1_relu (Activati  (None, 7, 7, 128)   0           ['conv5_block32_1_bn[0][0]']     
     on)                                                                                              
                                                                                                      
     conv5_block32_2_conv (Conv2D)  (None, 7, 7, 32)     36864       ['conv5_block32_1_relu[0][0]']   
                                                                                                      
     conv5_block32_concat (Concaten  (None, 7, 7, 1920)  0           ['conv5_block31_concat[0][0]',   
     ate)                                                             'conv5_block32_2_conv[0][0]']   
                                                                                                      
     bn (BatchNormalization)        (None, 7, 7, 1920)   7680        ['conv5_block32_concat[0][0]']   
                                                                                                      
     relu (Activation)              (None, 7, 7, 1920)   0           ['bn[0][0]']                     
                                                                                                      
     avg_pool (GlobalAveragePooling  (None, 1920)        0           ['relu[0][0]']                   
     2D)                                                                                              
                                                                                                      
     predictions (Dense)            (None, 1000)         1921000     ['avg_pool[0][0]']               
                                                                                                      
    ==================================================================================================
    Total params: 20,242,984
    Trainable params: 20,013,928
    Non-trainable params: 229,056
    __________________________________________________________________________________________________
    


```python
!wget -O zebra.jpg https://upload.wikimedia.org/wikipedia/commons/f/f0/Zebra_standing_alone_crop.jpg
```

    --2023-03-09 15:03:49--  https://upload.wikimedia.org/wikipedia/commons/f/f0/Zebra_standing_alone_crop.jpg
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 103.102.166.240, 2001:df2:e500:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|103.102.166.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 188036 (184K) [image/jpeg]
    Saving to: ‘zebra.jpg’
    
    zebra.jpg           100%[===================>] 183.63K  --.-KB/s    in 0.007s  
    
    2023-03-09 15:03:49 (24.8 MB/s) - ‘zebra.jpg’ saved [188036/188036]
    
    


```python
# 다운로드한 zebra.jpg를 terget_size로 줄여줌
# zebra input은 [(None, 224, 224, 3  0)] 크기를 가짐
img = image.load_img('zebra.jpg', target_size=(224, 224))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = densenet.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    WARNING:tensorflow:6 out of the last 7 calls to <function Model.make_predict_function.<locals>.predict_function at 0x7f7bb90f8700> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    1/1 [==============================] - 4s 4s/step
    [[('n02391049', 'zebra', 0.9312889), ('n01518878', 'ostrich', 0.019832587), ('n02423022', 'gazelle', 0.011593062), ('n02397096', 'warthog', 0.0046255025), ('n02422106', 'hartebeest', 0.003152003)]]
    


    


## NasNet

* 신경망 구조를 사람이 설계하지 않고, complete search를 통해 자동으로 구조를 찾아냄
* 네트워크를 구성하는 layer를 하나씩 탐색하는 NAS 방법 대신, NasNet은 Convolution cell 단위를 먼저 추정하고, 이들을 조합하여 전체 네트워크 구성
* 성능은 높지만, 파라미터 수와 연산량은 절반 정도로 감소


```python
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input, decode_predictions
```


```python
nasnet = NASNetLarge(include_top=True, weights='imagenet',
                      input_tensor=None, input_shape=None,
                      pooling=None, classes=1000)
nasnet.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-large.h5
    359748576/359748576 [==============================] - 17s 0us/step
    Model: "NASNet"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_7 (InputLayer)           [(None, 331, 331, 3  0           []                               
                                    )]                                                                
                                                                                                      
     stem_conv1 (Conv2D)            (None, 165, 165, 96  2592        ['input_7[0][0]']                
                                    )                                                                 
                                                                                                      
     stem_bn1 (BatchNormalization)  (None, 165, 165, 96  384         ['stem_conv1[0][0]']             
                                    )                                                                 
                                                                                                      
     activation_94 (Activation)     (None, 165, 165, 96  0           ['stem_bn1[0][0]']               
                                    )                                                                 
                                                                                                      
     reduction_conv_1_stem_1 (Conv2  (None, 165, 165, 42  4032       ['activation_94[0][0]']          
     D)                             )                                                                 
                                                                                                      
     reduction_bn_1_stem_1 (BatchNo  (None, 165, 165, 42  168        ['reduction_conv_1_stem_1[0][0]']
     rmalization)                   )                                                                 
                                                                                                      
     activation_95 (Activation)     (None, 165, 165, 42  0           ['reduction_bn_1_stem_1[0][0]']  
                                    )                                                                 
                                                                                                      
     activation_97 (Activation)     (None, 165, 165, 96  0           ['stem_bn1[0][0]']               
                                    )                                                                 
                                                                                                      
     separable_conv_1_pad_reduction  (None, 169, 169, 42  0          ['activation_95[0][0]']          
     _left1_stem_1 (ZeroPadding2D)  )                                                                 
                                                                                                      
     separable_conv_1_pad_reduction  (None, 171, 171, 96  0          ['activation_97[0][0]']          
     _right1_stem_1 (ZeroPadding2D)  )                                                                
                                                                                                      
     separable_conv_1_reduction_lef  (None, 83, 83, 42)  2814        ['separable_conv_1_pad_reduction_
     t1_stem_1 (SeparableConv2D)                                     left1_stem_1[0][0]']             
                                                                                                      
     separable_conv_1_reduction_rig  (None, 83, 83, 42)  8736        ['separable_conv_1_pad_reduction_
     ht1_stem_1 (SeparableConv2D)                                    right1_stem_1[0][0]']            
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_1_reduction_left
     left1_stem_1 (BatchNormalizati                                  1_stem_1[0][0]']                 
     on)                                                                                              
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_1_reduction_righ
     right1_stem_1 (BatchNormalizat                                  t1_stem_1[0][0]']                
     ion)                                                                                             
                                                                                                      
     activation_96 (Activation)     (None, 83, 83, 42)   0           ['separable_conv_1_bn_reduction_l
                                                                     eft1_stem_1[0][0]']              
                                                                                                      
     activation_98 (Activation)     (None, 83, 83, 42)   0           ['separable_conv_1_bn_reduction_r
                                                                     ight1_stem_1[0][0]']             
                                                                                                      
     separable_conv_2_reduction_lef  (None, 83, 83, 42)  2814        ['activation_96[0][0]']          
     t1_stem_1 (SeparableConv2D)                                                                      
                                                                                                      
     separable_conv_2_reduction_rig  (None, 83, 83, 42)  3822        ['activation_98[0][0]']          
     ht1_stem_1 (SeparableConv2D)                                                                     
                                                                                                      
     activation_99 (Activation)     (None, 165, 165, 96  0           ['stem_bn1[0][0]']               
                                    )                                                                 
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_2_reduction_left
     left1_stem_1 (BatchNormalizati                                  1_stem_1[0][0]']                 
     on)                                                                                              
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_2_reduction_righ
     right1_stem_1 (BatchNormalizat                                  t1_stem_1[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_1_pad_reduction  (None, 171, 171, 96  0          ['activation_99[0][0]']          
     _right2_stem_1 (ZeroPadding2D)  )                                                                
                                                                                                      
     activation_101 (Activation)    (None, 165, 165, 96  0           ['stem_bn1[0][0]']               
                                    )                                                                 
                                                                                                      
     reduction_add_1_stem_1 (Add)   (None, 83, 83, 42)   0           ['separable_conv_2_bn_reduction_l
                                                                     eft1_stem_1[0][0]',              
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight1_stem_1[0][0]']             
                                                                                                      
     separable_conv_1_reduction_rig  (None, 83, 83, 42)  8736        ['separable_conv_1_pad_reduction_
     ht2_stem_1 (SeparableConv2D)                                    right2_stem_1[0][0]']            
                                                                                                      
     separable_conv_1_pad_reduction  (None, 169, 169, 96  0          ['activation_101[0][0]']         
     _right3_stem_1 (ZeroPadding2D)  )                                                                
                                                                                                      
     activation_103 (Activation)    (None, 83, 83, 42)   0           ['reduction_add_1_stem_1[0][0]'] 
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_1_reduction_righ
     right2_stem_1 (BatchNormalizat                                  t2_stem_1[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_1_reduction_rig  (None, 83, 83, 42)  6432        ['separable_conv_1_pad_reduction_
     ht3_stem_1 (SeparableConv2D)                                    right3_stem_1[0][0]']            
                                                                                                      
     separable_conv_1_reduction_lef  (None, 83, 83, 42)  2142        ['activation_103[0][0]']         
     t4_stem_1 (SeparableConv2D)                                                                      
                                                                                                      
     activation_100 (Activation)    (None, 83, 83, 42)   0           ['separable_conv_1_bn_reduction_r
                                                                     ight2_stem_1[0][0]']             
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_1_reduction_righ
     right3_stem_1 (BatchNormalizat                                  t3_stem_1[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_1_reduction_left
     left4_stem_1 (BatchNormalizati                                  4_stem_1[0][0]']                 
     on)                                                                                              
                                                                                                      
     reduction_pad_1_stem_1 (ZeroPa  (None, 167, 167, 42  0          ['reduction_bn_1_stem_1[0][0]']  
     dding2D)                       )                                                                 
                                                                                                      
     separable_conv_2_reduction_rig  (None, 83, 83, 42)  3822        ['activation_100[0][0]']         
     ht2_stem_1 (SeparableConv2D)                                                                     
                                                                                                      
     activation_102 (Activation)    (None, 83, 83, 42)   0           ['separable_conv_1_bn_reduction_r
                                                                     ight3_stem_1[0][0]']             
                                                                                                      
     activation_104 (Activation)    (None, 83, 83, 42)   0           ['separable_conv_1_bn_reduction_l
                                                                     eft4_stem_1[0][0]']              
                                                                                                      
     reduction_left2_stem_1 (MaxPoo  (None, 83, 83, 42)  0           ['reduction_pad_1_stem_1[0][0]'] 
     ling2D)                                                                                          
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_2_reduction_righ
     right2_stem_1 (BatchNormalizat                                  t2_stem_1[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_2_reduction_rig  (None, 83, 83, 42)  2814        ['activation_102[0][0]']         
     ht3_stem_1 (SeparableConv2D)                                                                     
                                                                                                      
     separable_conv_2_reduction_lef  (None, 83, 83, 42)  2142        ['activation_104[0][0]']         
     t4_stem_1 (SeparableConv2D)                                                                      
                                                                                                      
     adjust_relu_1_stem_2 (Activati  (None, 165, 165, 96  0          ['stem_bn1[0][0]']               
     on)                            )                                                                 
                                                                                                      
     reduction_add_2_stem_1 (Add)   (None, 83, 83, 42)   0           ['reduction_left2_stem_1[0][0]', 
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight2_stem_1[0][0]']             
                                                                                                      
     reduction_left3_stem_1 (Averag  (None, 83, 83, 42)  0           ['reduction_pad_1_stem_1[0][0]'] 
     ePooling2D)                                                                                      
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_2_reduction_righ
     right3_stem_1 (BatchNormalizat                                  t3_stem_1[0][0]']                
     ion)                                                                                             
                                                                                                      
     reduction_left4_stem_1 (Averag  (None, 83, 83, 42)  0           ['reduction_add_1_stem_1[0][0]'] 
     ePooling2D)                                                                                      
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 83, 83, 42)  168         ['separable_conv_2_reduction_left
     left4_stem_1 (BatchNormalizati                                  4_stem_1[0][0]']                 
     on)                                                                                              
                                                                                                      
     reduction_right5_stem_1 (MaxPo  (None, 83, 83, 42)  0           ['reduction_pad_1_stem_1[0][0]'] 
     oling2D)                                                                                         
                                                                                                      
     zero_padding2d_2 (ZeroPadding2  (None, 166, 166, 96  0          ['adjust_relu_1_stem_2[0][0]']   
     D)                             )                                                                 
                                                                                                      
     reduction_add3_stem_1 (Add)    (None, 83, 83, 42)   0           ['reduction_left3_stem_1[0][0]', 
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight3_stem_1[0][0]']             
                                                                                                      
     add_12 (Add)                   (None, 83, 83, 42)   0           ['reduction_add_2_stem_1[0][0]', 
                                                                      'reduction_left4_stem_1[0][0]'] 
                                                                                                      
     reduction_add4_stem_1 (Add)    (None, 83, 83, 42)   0           ['separable_conv_2_bn_reduction_l
                                                                     eft4_stem_1[0][0]',              
                                                                      'reduction_right5_stem_1[0][0]']
                                                                                                      
     cropping2d (Cropping2D)        (None, 165, 165, 96  0           ['zero_padding2d_2[0][0]']       
                                    )                                                                 
                                                                                                      
     reduction_concat_stem_1 (Conca  (None, 83, 83, 168)  0          ['reduction_add_2_stem_1[0][0]', 
     tenate)                                                          'reduction_add3_stem_1[0][0]',  
                                                                      'add_12[0][0]',                 
                                                                      'reduction_add4_stem_1[0][0]']  
                                                                                                      
     adjust_avg_pool_1_stem_2 (Aver  (None, 83, 83, 96)  0           ['adjust_relu_1_stem_2[0][0]']   
     agePooling2D)                                                                                    
                                                                                                      
     adjust_avg_pool_2_stem_2 (Aver  (None, 83, 83, 96)  0           ['cropping2d[0][0]']             
     agePooling2D)                                                                                    
                                                                                                      
     activation_105 (Activation)    (None, 83, 83, 168)  0           ['reduction_concat_stem_1[0][0]']
                                                                                                      
     adjust_conv_1_stem_2 (Conv2D)  (None, 83, 83, 42)   4032        ['adjust_avg_pool_1_stem_2[0][0]'
                                                                     ]                                
                                                                                                      
     adjust_conv_2_stem_2 (Conv2D)  (None, 83, 83, 42)   4032        ['adjust_avg_pool_2_stem_2[0][0]'
                                                                     ]                                
                                                                                                      
     reduction_conv_1_stem_2 (Conv2  (None, 83, 83, 84)  14112       ['activation_105[0][0]']         
     D)                                                                                               
                                                                                                      
     concatenate_2 (Concatenate)    (None, 83, 83, 84)   0           ['adjust_conv_1_stem_2[0][0]',   
                                                                      'adjust_conv_2_stem_2[0][0]']   
                                                                                                      
     reduction_bn_1_stem_2 (BatchNo  (None, 83, 83, 84)  336         ['reduction_conv_1_stem_2[0][0]']
     rmalization)                                                                                     
                                                                                                      
     adjust_bn_stem_2 (BatchNormali  (None, 83, 83, 84)  336         ['concatenate_2[0][0]']          
     zation)                                                                                          
                                                                                                      
     activation_106 (Activation)    (None, 83, 83, 84)   0           ['reduction_bn_1_stem_2[0][0]']  
                                                                                                      
     activation_108 (Activation)    (None, 83, 83, 84)   0           ['adjust_bn_stem_2[0][0]']       
                                                                                                      
     separable_conv_1_pad_reduction  (None, 87, 87, 84)  0           ['activation_106[0][0]']         
     _left1_stem_2 (ZeroPadding2D)                                                                    
                                                                                                      
     separable_conv_1_pad_reduction  (None, 89, 89, 84)  0           ['activation_108[0][0]']         
     _right1_stem_2 (ZeroPadding2D)                                                                   
                                                                                                      
     separable_conv_1_reduction_lef  (None, 42, 42, 84)  9156        ['separable_conv_1_pad_reduction_
     t1_stem_2 (SeparableConv2D)                                     left1_stem_2[0][0]']             
                                                                                                      
     separable_conv_1_reduction_rig  (None, 42, 42, 84)  11172       ['separable_conv_1_pad_reduction_
     ht1_stem_2 (SeparableConv2D)                                    right1_stem_2[0][0]']            
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_1_reduction_left
     left1_stem_2 (BatchNormalizati                                  1_stem_2[0][0]']                 
     on)                                                                                              
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_1_reduction_righ
     right1_stem_2 (BatchNormalizat                                  t1_stem_2[0][0]']                
     ion)                                                                                             
                                                                                                      
     activation_107 (Activation)    (None, 42, 42, 84)   0           ['separable_conv_1_bn_reduction_l
                                                                     eft1_stem_2[0][0]']              
                                                                                                      
     activation_109 (Activation)    (None, 42, 42, 84)   0           ['separable_conv_1_bn_reduction_r
                                                                     ight1_stem_2[0][0]']             
                                                                                                      
     separable_conv_2_reduction_lef  (None, 42, 42, 84)  9156        ['activation_107[0][0]']         
     t1_stem_2 (SeparableConv2D)                                                                      
                                                                                                      
     separable_conv_2_reduction_rig  (None, 42, 42, 84)  11172       ['activation_109[0][0]']         
     ht1_stem_2 (SeparableConv2D)                                                                     
                                                                                                      
     activation_110 (Activation)    (None, 83, 83, 84)   0           ['adjust_bn_stem_2[0][0]']       
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_2_reduction_left
     left1_stem_2 (BatchNormalizati                                  1_stem_2[0][0]']                 
     on)                                                                                              
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_2_reduction_righ
     right1_stem_2 (BatchNormalizat                                  t1_stem_2[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_1_pad_reduction  (None, 89, 89, 84)  0           ['activation_110[0][0]']         
     _right2_stem_2 (ZeroPadding2D)                                                                   
                                                                                                      
     activation_112 (Activation)    (None, 83, 83, 84)   0           ['adjust_bn_stem_2[0][0]']       
                                                                                                      
     reduction_add_1_stem_2 (Add)   (None, 42, 42, 84)   0           ['separable_conv_2_bn_reduction_l
                                                                     eft1_stem_2[0][0]',              
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight1_stem_2[0][0]']             
                                                                                                      
     separable_conv_1_reduction_rig  (None, 42, 42, 84)  11172       ['separable_conv_1_pad_reduction_
     ht2_stem_2 (SeparableConv2D)                                    right2_stem_2[0][0]']            
                                                                                                      
     separable_conv_1_pad_reduction  (None, 87, 87, 84)  0           ['activation_112[0][0]']         
     _right3_stem_2 (ZeroPadding2D)                                                                   
                                                                                                      
     activation_114 (Activation)    (None, 42, 42, 84)   0           ['reduction_add_1_stem_2[0][0]'] 
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_1_reduction_righ
     right2_stem_2 (BatchNormalizat                                  t2_stem_2[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_1_reduction_rig  (None, 42, 42, 84)  9156        ['separable_conv_1_pad_reduction_
     ht3_stem_2 (SeparableConv2D)                                    right3_stem_2[0][0]']            
                                                                                                      
     separable_conv_1_reduction_lef  (None, 42, 42, 84)  7812        ['activation_114[0][0]']         
     t4_stem_2 (SeparableConv2D)                                                                      
                                                                                                      
     activation_111 (Activation)    (None, 42, 42, 84)   0           ['separable_conv_1_bn_reduction_r
                                                                     ight2_stem_2[0][0]']             
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_1_reduction_righ
     right3_stem_2 (BatchNormalizat                                  t3_stem_2[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_1_reduction_left
     left4_stem_2 (BatchNormalizati                                  4_stem_2[0][0]']                 
     on)                                                                                              
                                                                                                      
     reduction_pad_1_stem_2 (ZeroPa  (None, 85, 85, 84)  0           ['reduction_bn_1_stem_2[0][0]']  
     dding2D)                                                                                         
                                                                                                      
     separable_conv_2_reduction_rig  (None, 42, 42, 84)  11172       ['activation_111[0][0]']         
     ht2_stem_2 (SeparableConv2D)                                                                     
                                                                                                      
     activation_113 (Activation)    (None, 42, 42, 84)   0           ['separable_conv_1_bn_reduction_r
                                                                     ight3_stem_2[0][0]']             
                                                                                                      
     activation_115 (Activation)    (None, 42, 42, 84)   0           ['separable_conv_1_bn_reduction_l
                                                                     eft4_stem_2[0][0]']              
                                                                                                      
     reduction_left2_stem_2 (MaxPoo  (None, 42, 42, 84)  0           ['reduction_pad_1_stem_2[0][0]'] 
     ling2D)                                                                                          
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_2_reduction_righ
     right2_stem_2 (BatchNormalizat                                  t2_stem_2[0][0]']                
     ion)                                                                                             
                                                                                                      
     separable_conv_2_reduction_rig  (None, 42, 42, 84)  9156        ['activation_113[0][0]']         
     ht3_stem_2 (SeparableConv2D)                                                                     
                                                                                                      
     separable_conv_2_reduction_lef  (None, 42, 42, 84)  7812        ['activation_115[0][0]']         
     t4_stem_2 (SeparableConv2D)                                                                      
                                                                                                      
     adjust_relu_1_0 (Activation)   (None, 83, 83, 168)  0           ['reduction_concat_stem_1[0][0]']
                                                                                                      
     reduction_add_2_stem_2 (Add)   (None, 42, 42, 84)   0           ['reduction_left2_stem_2[0][0]', 
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight2_stem_2[0][0]']             
                                                                                                      
     reduction_left3_stem_2 (Averag  (None, 42, 42, 84)  0           ['reduction_pad_1_stem_2[0][0]'] 
     ePooling2D)                                                                                      
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_2_reduction_righ
     right3_stem_2 (BatchNormalizat                                  t3_stem_2[0][0]']                
     ion)                                                                                             
                                                                                                      
     reduction_left4_stem_2 (Averag  (None, 42, 42, 84)  0           ['reduction_add_1_stem_2[0][0]'] 
     ePooling2D)                                                                                      
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 42, 42, 84)  336         ['separable_conv_2_reduction_left
     left4_stem_2 (BatchNormalizati                                  4_stem_2[0][0]']                 
     on)                                                                                              
                                                                                                      
     reduction_right5_stem_2 (MaxPo  (None, 42, 42, 84)  0           ['reduction_pad_1_stem_2[0][0]'] 
     oling2D)                                                                                         
                                                                                                      
     zero_padding2d_3 (ZeroPadding2  (None, 84, 84, 168)  0          ['adjust_relu_1_0[0][0]']        
     D)                                                                                               
                                                                                                      
     reduction_add3_stem_2 (Add)    (None, 42, 42, 84)   0           ['reduction_left3_stem_2[0][0]', 
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight3_stem_2[0][0]']             
                                                                                                      
     add_13 (Add)                   (None, 42, 42, 84)   0           ['reduction_add_2_stem_2[0][0]', 
                                                                      'reduction_left4_stem_2[0][0]'] 
                                                                                                      
     reduction_add4_stem_2 (Add)    (None, 42, 42, 84)   0           ['separable_conv_2_bn_reduction_l
                                                                     eft4_stem_2[0][0]',              
                                                                      'reduction_right5_stem_2[0][0]']
                                                                                                      
     cropping2d_1 (Cropping2D)      (None, 83, 83, 168)  0           ['zero_padding2d_3[0][0]']       
                                                                                                      
     reduction_concat_stem_2 (Conca  (None, 42, 42, 336)  0          ['reduction_add_2_stem_2[0][0]', 
     tenate)                                                          'reduction_add3_stem_2[0][0]',  
                                                                      'add_13[0][0]',                 
                                                                      'reduction_add4_stem_2[0][0]']  
                                                                                                      
     adjust_avg_pool_1_0 (AveragePo  (None, 42, 42, 168)  0          ['adjust_relu_1_0[0][0]']        
     oling2D)                                                                                         
                                                                                                      
     adjust_avg_pool_2_0 (AveragePo  (None, 42, 42, 168)  0          ['cropping2d_1[0][0]']           
     oling2D)                                                                                         
                                                                                                      
     adjust_conv_1_0 (Conv2D)       (None, 42, 42, 84)   14112       ['adjust_avg_pool_1_0[0][0]']    
                                                                                                      
     adjust_conv_2_0 (Conv2D)       (None, 42, 42, 84)   14112       ['adjust_avg_pool_2_0[0][0]']    
                                                                                                      
     activation_116 (Activation)    (None, 42, 42, 336)  0           ['reduction_concat_stem_2[0][0]']
                                                                                                      
     concatenate_3 (Concatenate)    (None, 42, 42, 168)  0           ['adjust_conv_1_0[0][0]',        
                                                                      'adjust_conv_2_0[0][0]']        
                                                                                                      
     normal_conv_1_0 (Conv2D)       (None, 42, 42, 168)  56448       ['activation_116[0][0]']         
                                                                                                      
     adjust_bn_0 (BatchNormalizatio  (None, 42, 42, 168)  672        ['concatenate_3[0][0]']          
     n)                                                                                               
                                                                                                      
     normal_bn_1_0 (BatchNormalizat  (None, 42, 42, 168)  672        ['normal_conv_1_0[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_117 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_0[0][0]']          
                                                                                                      
     activation_119 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_0[0][0]']            
                                                                                                      
     activation_121 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_0[0][0]']            
                                                                                                      
     activation_123 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_0[0][0]']            
                                                                                                      
     activation_125 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_0[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 42, 42, 168)  32424      ['activation_117[0][0]']         
     0 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 42, 42, 168)  29736      ['activation_119[0][0]']         
     _0 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 42, 42, 168)  32424      ['activation_121[0][0]']         
     0 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 42, 42, 168)  29736      ['activation_123[0][0]']         
     _0 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 42, 42, 168)  29736      ['activation_125[0][0]']         
     0 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left1_0
     t1_0 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right1_
     ht1_0 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left2_0
     t2_0 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right2_
     ht2_0 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left5_0
     t5_0 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_118 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     1_0[0][0]']                      
                                                                                                      
     activation_120 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_0[0][0]']                     
                                                                                                      
     activation_122 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     2_0[0][0]']                      
                                                                                                      
     activation_124 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_0[0][0]']                     
                                                                                                      
     activation_126 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     5_0[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 42, 42, 168)  32424      ['activation_118[0][0]']         
     0 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 42, 42, 168)  29736      ['activation_120[0][0]']         
     _0 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 42, 42, 168)  32424      ['activation_122[0][0]']         
     0 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 42, 42, 168)  29736      ['activation_124[0][0]']         
     _0 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 42, 42, 168)  29736      ['activation_126[0][0]']         
     0 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left1_0
     t1_0 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right1_
     ht1_0 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left2_0
     t2_0 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right2_
     ht2_0 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     normal_left3_0 (AveragePooling  (None, 42, 42, 168)  0          ['normal_bn_1_0[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_0 (AveragePooling  (None, 42, 42, 168)  0          ['adjust_bn_0[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_0 (AveragePoolin  (None, 42, 42, 168)  0          ['adjust_bn_0[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left5_0
     t5_0 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_0 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     1_0[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_0[0][0]']                     
                                                                                                      
     normal_add_2_0 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     2_0[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_0[0][0]']                     
                                                                                                      
     normal_add_3_0 (Add)           (None, 42, 42, 168)  0           ['normal_left3_0[0][0]',         
                                                                      'adjust_bn_0[0][0]']            
                                                                                                      
     normal_add_4_0 (Add)           (None, 42, 42, 168)  0           ['normal_left4_0[0][0]',         
                                                                      'normal_right4_0[0][0]']        
                                                                                                      
     normal_add_5_0 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     5_0[0][0]',                      
                                                                      'normal_bn_1_0[0][0]']          
                                                                                                      
     normal_concat_0 (Concatenate)  (None, 42, 42, 1008  0           ['adjust_bn_0[0][0]',            
                                    )                                 'normal_add_1_0[0][0]',         
                                                                      'normal_add_2_0[0][0]',         
                                                                      'normal_add_3_0[0][0]',         
                                                                      'normal_add_4_0[0][0]',         
                                                                      'normal_add_5_0[0][0]']         
                                                                                                      
     activation_127 (Activation)    (None, 42, 42, 336)  0           ['reduction_concat_stem_2[0][0]']
                                                                                                      
     activation_128 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_0[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_1 (Conv  (None, 42, 42, 168)  56448      ['activation_127[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_1 (Conv2D)       (None, 42, 42, 168)  169344      ['activation_128[0][0]']         
                                                                                                      
     adjust_bn_1 (BatchNormalizatio  (None, 42, 42, 168)  672        ['adjust_conv_projection_1[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_1 (BatchNormalizat  (None, 42, 42, 168)  672        ['normal_conv_1_1[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_129 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_1[0][0]']          
                                                                                                      
     activation_131 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_1[0][0]']            
                                                                                                      
     activation_133 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_1[0][0]']            
                                                                                                      
     activation_135 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_1[0][0]']            
                                                                                                      
     activation_137 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_1[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 42, 42, 168)  32424      ['activation_129[0][0]']         
     1 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 42, 42, 168)  29736      ['activation_131[0][0]']         
     _1 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 42, 42, 168)  32424      ['activation_133[0][0]']         
     1 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 42, 42, 168)  29736      ['activation_135[0][0]']         
     _1 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 42, 42, 168)  29736      ['activation_137[0][0]']         
     1 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left1_1
     t1_1 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right1_
     ht1_1 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left2_1
     t2_1 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right2_
     ht2_1 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left5_1
     t5_1 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_130 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     1_1[0][0]']                      
                                                                                                      
     activation_132 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_1[0][0]']                     
                                                                                                      
     activation_134 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     2_1[0][0]']                      
                                                                                                      
     activation_136 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_1[0][0]']                     
                                                                                                      
     activation_138 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     5_1[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 42, 42, 168)  32424      ['activation_130[0][0]']         
     1 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 42, 42, 168)  29736      ['activation_132[0][0]']         
     _1 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 42, 42, 168)  32424      ['activation_134[0][0]']         
     1 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 42, 42, 168)  29736      ['activation_136[0][0]']         
     _1 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 42, 42, 168)  29736      ['activation_138[0][0]']         
     1 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left1_1
     t1_1 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right1_
     ht1_1 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left2_1
     t2_1 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right2_
     ht2_1 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     normal_left3_1 (AveragePooling  (None, 42, 42, 168)  0          ['normal_bn_1_1[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_1 (AveragePooling  (None, 42, 42, 168)  0          ['adjust_bn_1[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_1 (AveragePoolin  (None, 42, 42, 168)  0          ['adjust_bn_1[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left5_1
     t5_1 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_1 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     1_1[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_1[0][0]']                     
                                                                                                      
     normal_add_2_1 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     2_1[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_1[0][0]']                     
                                                                                                      
     normal_add_3_1 (Add)           (None, 42, 42, 168)  0           ['normal_left3_1[0][0]',         
                                                                      'adjust_bn_1[0][0]']            
                                                                                                      
     normal_add_4_1 (Add)           (None, 42, 42, 168)  0           ['normal_left4_1[0][0]',         
                                                                      'normal_right4_1[0][0]']        
                                                                                                      
     normal_add_5_1 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     5_1[0][0]',                      
                                                                      'normal_bn_1_1[0][0]']          
                                                                                                      
     normal_concat_1 (Concatenate)  (None, 42, 42, 1008  0           ['adjust_bn_1[0][0]',            
                                    )                                 'normal_add_1_1[0][0]',         
                                                                      'normal_add_2_1[0][0]',         
                                                                      'normal_add_3_1[0][0]',         
                                                                      'normal_add_4_1[0][0]',         
                                                                      'normal_add_5_1[0][0]']         
                                                                                                      
     activation_139 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_0[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_140 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_1[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_2 (Conv  (None, 42, 42, 168)  169344     ['activation_139[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_2 (Conv2D)       (None, 42, 42, 168)  169344      ['activation_140[0][0]']         
                                                                                                      
     adjust_bn_2 (BatchNormalizatio  (None, 42, 42, 168)  672        ['adjust_conv_projection_2[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_2 (BatchNormalizat  (None, 42, 42, 168)  672        ['normal_conv_1_2[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_141 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_2[0][0]']          
                                                                                                      
     activation_143 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_2[0][0]']            
                                                                                                      
     activation_145 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_2[0][0]']            
                                                                                                      
     activation_147 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_2[0][0]']            
                                                                                                      
     activation_149 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_2[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 42, 42, 168)  32424      ['activation_141[0][0]']         
     2 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 42, 42, 168)  29736      ['activation_143[0][0]']         
     _2 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 42, 42, 168)  32424      ['activation_145[0][0]']         
     2 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 42, 42, 168)  29736      ['activation_147[0][0]']         
     _2 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 42, 42, 168)  29736      ['activation_149[0][0]']         
     2 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left1_2
     t1_2 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right1_
     ht1_2 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left2_2
     t2_2 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right2_
     ht2_2 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left5_2
     t5_2 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_142 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     1_2[0][0]']                      
                                                                                                      
     activation_144 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_2[0][0]']                     
                                                                                                      
     activation_146 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     2_2[0][0]']                      
                                                                                                      
     activation_148 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_2[0][0]']                     
                                                                                                      
     activation_150 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     5_2[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 42, 42, 168)  32424      ['activation_142[0][0]']         
     2 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 42, 42, 168)  29736      ['activation_144[0][0]']         
     _2 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 42, 42, 168)  32424      ['activation_146[0][0]']         
     2 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 42, 42, 168)  29736      ['activation_148[0][0]']         
     _2 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 42, 42, 168)  29736      ['activation_150[0][0]']         
     2 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left1_2
     t1_2 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right1_
     ht1_2 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left2_2
     t2_2 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right2_
     ht2_2 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     normal_left3_2 (AveragePooling  (None, 42, 42, 168)  0          ['normal_bn_1_2[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_2 (AveragePooling  (None, 42, 42, 168)  0          ['adjust_bn_2[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_2 (AveragePoolin  (None, 42, 42, 168)  0          ['adjust_bn_2[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left5_2
     t5_2 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_2 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     1_2[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_2[0][0]']                     
                                                                                                      
     normal_add_2_2 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     2_2[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_2[0][0]']                     
                                                                                                      
     normal_add_3_2 (Add)           (None, 42, 42, 168)  0           ['normal_left3_2[0][0]',         
                                                                      'adjust_bn_2[0][0]']            
                                                                                                      
     normal_add_4_2 (Add)           (None, 42, 42, 168)  0           ['normal_left4_2[0][0]',         
                                                                      'normal_right4_2[0][0]']        
                                                                                                      
     normal_add_5_2 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     5_2[0][0]',                      
                                                                      'normal_bn_1_2[0][0]']          
                                                                                                      
     normal_concat_2 (Concatenate)  (None, 42, 42, 1008  0           ['adjust_bn_2[0][0]',            
                                    )                                 'normal_add_1_2[0][0]',         
                                                                      'normal_add_2_2[0][0]',         
                                                                      'normal_add_3_2[0][0]',         
                                                                      'normal_add_4_2[0][0]',         
                                                                      'normal_add_5_2[0][0]']         
                                                                                                      
     activation_151 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_1[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_152 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_2[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_3 (Conv  (None, 42, 42, 168)  169344     ['activation_151[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_3 (Conv2D)       (None, 42, 42, 168)  169344      ['activation_152[0][0]']         
                                                                                                      
     adjust_bn_3 (BatchNormalizatio  (None, 42, 42, 168)  672        ['adjust_conv_projection_3[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_3 (BatchNormalizat  (None, 42, 42, 168)  672        ['normal_conv_1_3[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_153 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_3[0][0]']          
                                                                                                      
     activation_155 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_3[0][0]']            
                                                                                                      
     activation_157 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_3[0][0]']            
                                                                                                      
     activation_159 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_3[0][0]']            
                                                                                                      
     activation_161 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_3[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 42, 42, 168)  32424      ['activation_153[0][0]']         
     3 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 42, 42, 168)  29736      ['activation_155[0][0]']         
     _3 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 42, 42, 168)  32424      ['activation_157[0][0]']         
     3 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 42, 42, 168)  29736      ['activation_159[0][0]']         
     _3 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 42, 42, 168)  29736      ['activation_161[0][0]']         
     3 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left1_3
     t1_3 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right1_
     ht1_3 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left2_3
     t2_3 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right2_
     ht2_3 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left5_3
     t5_3 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_154 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     1_3[0][0]']                      
                                                                                                      
     activation_156 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_3[0][0]']                     
                                                                                                      
     activation_158 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     2_3[0][0]']                      
                                                                                                      
     activation_160 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_3[0][0]']                     
                                                                                                      
     activation_162 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     5_3[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 42, 42, 168)  32424      ['activation_154[0][0]']         
     3 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 42, 42, 168)  29736      ['activation_156[0][0]']         
     _3 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 42, 42, 168)  32424      ['activation_158[0][0]']         
     3 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 42, 42, 168)  29736      ['activation_160[0][0]']         
     _3 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 42, 42, 168)  29736      ['activation_162[0][0]']         
     3 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left1_3
     t1_3 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right1_
     ht1_3 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left2_3
     t2_3 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right2_
     ht2_3 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     normal_left3_3 (AveragePooling  (None, 42, 42, 168)  0          ['normal_bn_1_3[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_3 (AveragePooling  (None, 42, 42, 168)  0          ['adjust_bn_3[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_3 (AveragePoolin  (None, 42, 42, 168)  0          ['adjust_bn_3[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left5_3
     t5_3 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_3 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     1_3[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_3[0][0]']                     
                                                                                                      
     normal_add_2_3 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     2_3[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_3[0][0]']                     
                                                                                                      
     normal_add_3_3 (Add)           (None, 42, 42, 168)  0           ['normal_left3_3[0][0]',         
                                                                      'adjust_bn_3[0][0]']            
                                                                                                      
     normal_add_4_3 (Add)           (None, 42, 42, 168)  0           ['normal_left4_3[0][0]',         
                                                                      'normal_right4_3[0][0]']        
                                                                                                      
     normal_add_5_3 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     5_3[0][0]',                      
                                                                      'normal_bn_1_3[0][0]']          
                                                                                                      
     normal_concat_3 (Concatenate)  (None, 42, 42, 1008  0           ['adjust_bn_3[0][0]',            
                                    )                                 'normal_add_1_3[0][0]',         
                                                                      'normal_add_2_3[0][0]',         
                                                                      'normal_add_3_3[0][0]',         
                                                                      'normal_add_4_3[0][0]',         
                                                                      'normal_add_5_3[0][0]']         
                                                                                                      
     activation_163 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_2[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_164 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_3[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_4 (Conv  (None, 42, 42, 168)  169344     ['activation_163[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_4 (Conv2D)       (None, 42, 42, 168)  169344      ['activation_164[0][0]']         
                                                                                                      
     adjust_bn_4 (BatchNormalizatio  (None, 42, 42, 168)  672        ['adjust_conv_projection_4[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_4 (BatchNormalizat  (None, 42, 42, 168)  672        ['normal_conv_1_4[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_165 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_4[0][0]']          
                                                                                                      
     activation_167 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_4[0][0]']            
                                                                                                      
     activation_169 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_4[0][0]']            
                                                                                                      
     activation_171 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_4[0][0]']            
                                                                                                      
     activation_173 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_4[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 42, 42, 168)  32424      ['activation_165[0][0]']         
     4 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 42, 42, 168)  29736      ['activation_167[0][0]']         
     _4 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 42, 42, 168)  32424      ['activation_169[0][0]']         
     4 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 42, 42, 168)  29736      ['activation_171[0][0]']         
     _4 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 42, 42, 168)  29736      ['activation_173[0][0]']         
     4 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left1_4
     t1_4 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right1_
     ht1_4 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left2_4
     t2_4 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right2_
     ht2_4 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left5_4
     t5_4 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_166 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     1_4[0][0]']                      
                                                                                                      
     activation_168 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_4[0][0]']                     
                                                                                                      
     activation_170 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     2_4[0][0]']                      
                                                                                                      
     activation_172 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_4[0][0]']                     
                                                                                                      
     activation_174 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     5_4[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 42, 42, 168)  32424      ['activation_166[0][0]']         
     4 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 42, 42, 168)  29736      ['activation_168[0][0]']         
     _4 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 42, 42, 168)  32424      ['activation_170[0][0]']         
     4 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 42, 42, 168)  29736      ['activation_172[0][0]']         
     _4 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 42, 42, 168)  29736      ['activation_174[0][0]']         
     4 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left1_4
     t1_4 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right1_
     ht1_4 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left2_4
     t2_4 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right2_
     ht2_4 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     normal_left3_4 (AveragePooling  (None, 42, 42, 168)  0          ['normal_bn_1_4[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_4 (AveragePooling  (None, 42, 42, 168)  0          ['adjust_bn_4[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_4 (AveragePoolin  (None, 42, 42, 168)  0          ['adjust_bn_4[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left5_4
     t5_4 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_4 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     1_4[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_4[0][0]']                     
                                                                                                      
     normal_add_2_4 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     2_4[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_4[0][0]']                     
                                                                                                      
     normal_add_3_4 (Add)           (None, 42, 42, 168)  0           ['normal_left3_4[0][0]',         
                                                                      'adjust_bn_4[0][0]']            
                                                                                                      
     normal_add_4_4 (Add)           (None, 42, 42, 168)  0           ['normal_left4_4[0][0]',         
                                                                      'normal_right4_4[0][0]']        
                                                                                                      
     normal_add_5_4 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     5_4[0][0]',                      
                                                                      'normal_bn_1_4[0][0]']          
                                                                                                      
     normal_concat_4 (Concatenate)  (None, 42, 42, 1008  0           ['adjust_bn_4[0][0]',            
                                    )                                 'normal_add_1_4[0][0]',         
                                                                      'normal_add_2_4[0][0]',         
                                                                      'normal_add_3_4[0][0]',         
                                                                      'normal_add_4_4[0][0]',         
                                                                      'normal_add_5_4[0][0]']         
                                                                                                      
     activation_175 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_3[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_176 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_4[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_5 (Conv  (None, 42, 42, 168)  169344     ['activation_175[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_5 (Conv2D)       (None, 42, 42, 168)  169344      ['activation_176[0][0]']         
                                                                                                      
     adjust_bn_5 (BatchNormalizatio  (None, 42, 42, 168)  672        ['adjust_conv_projection_5[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_5 (BatchNormalizat  (None, 42, 42, 168)  672        ['normal_conv_1_5[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_177 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_5[0][0]']          
                                                                                                      
     activation_179 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_5[0][0]']            
                                                                                                      
     activation_181 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_5[0][0]']            
                                                                                                      
     activation_183 (Activation)    (None, 42, 42, 168)  0           ['adjust_bn_5[0][0]']            
                                                                                                      
     activation_185 (Activation)    (None, 42, 42, 168)  0           ['normal_bn_1_5[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 42, 42, 168)  32424      ['activation_177[0][0]']         
     5 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 42, 42, 168)  29736      ['activation_179[0][0]']         
     _5 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 42, 42, 168)  32424      ['activation_181[0][0]']         
     5 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 42, 42, 168)  29736      ['activation_183[0][0]']         
     _5 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 42, 42, 168)  29736      ['activation_185[0][0]']         
     5 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left1_5
     t1_5 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right1_
     ht1_5 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left2_5
     t2_5 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_1_normal_right2_
     ht2_5 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_1_normal_left5_5
     t5_5 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_178 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     1_5[0][0]']                      
                                                                                                      
     activation_180 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_5[0][0]']                     
                                                                                                      
     activation_182 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     2_5[0][0]']                      
                                                                                                      
     activation_184 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_5[0][0]']                     
                                                                                                      
     activation_186 (Activation)    (None, 42, 42, 168)  0           ['separable_conv_1_bn_normal_left
                                                                     5_5[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 42, 42, 168)  32424      ['activation_178[0][0]']         
     5 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 42, 42, 168)  29736      ['activation_180[0][0]']         
     _5 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 42, 42, 168)  32424      ['activation_182[0][0]']         
     5 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 42, 42, 168)  29736      ['activation_184[0][0]']         
     _5 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 42, 42, 168)  29736      ['activation_186[0][0]']         
     5 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left1_5
     t1_5 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right1_
     ht1_5 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left2_5
     t2_5 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 42, 42, 168)  672        ['separable_conv_2_normal_right2_
     ht2_5 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     normal_left3_5 (AveragePooling  (None, 42, 42, 168)  0          ['normal_bn_1_5[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_5 (AveragePooling  (None, 42, 42, 168)  0          ['adjust_bn_5[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_5 (AveragePoolin  (None, 42, 42, 168)  0          ['adjust_bn_5[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 42, 42, 168)  672        ['separable_conv_2_normal_left5_5
     t5_5 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_5 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     1_5[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_5[0][0]']                     
                                                                                                      
     normal_add_2_5 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     2_5[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_5[0][0]']                     
                                                                                                      
     normal_add_3_5 (Add)           (None, 42, 42, 168)  0           ['normal_left3_5[0][0]',         
                                                                      'adjust_bn_5[0][0]']            
                                                                                                      
     normal_add_4_5 (Add)           (None, 42, 42, 168)  0           ['normal_left4_5[0][0]',         
                                                                      'normal_right4_5[0][0]']        
                                                                                                      
     normal_add_5_5 (Add)           (None, 42, 42, 168)  0           ['separable_conv_2_bn_normal_left
                                                                     5_5[0][0]',                      
                                                                      'normal_bn_1_5[0][0]']          
                                                                                                      
     normal_concat_5 (Concatenate)  (None, 42, 42, 1008  0           ['adjust_bn_5[0][0]',            
                                    )                                 'normal_add_1_5[0][0]',         
                                                                      'normal_add_2_5[0][0]',         
                                                                      'normal_add_3_5[0][0]',         
                                                                      'normal_add_4_5[0][0]',         
                                                                      'normal_add_5_5[0][0]']         
                                                                                                      
     activation_188 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_5[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_187 (Activation)    (None, 42, 42, 1008  0           ['normal_concat_4[0][0]']        
                                    )                                                                 
                                                                                                      
     reduction_conv_1_reduce_6 (Con  (None, 42, 42, 336)  338688     ['activation_188[0][0]']         
     v2D)                                                                                             
                                                                                                      
     adjust_conv_projection_reduce_  (None, 42, 42, 336)  338688     ['activation_187[0][0]']         
     6 (Conv2D)                                                                                       
                                                                                                      
     reduction_bn_1_reduce_6 (Batch  (None, 42, 42, 336)  1344       ['reduction_conv_1_reduce_6[0][0]
     Normalization)                                                  ']                               
                                                                                                      
     adjust_bn_reduce_6 (BatchNorma  (None, 42, 42, 336)  1344       ['adjust_conv_projection_reduce_6
     lization)                                                       [0][0]']                         
                                                                                                      
     activation_189 (Activation)    (None, 42, 42, 336)  0           ['reduction_bn_1_reduce_6[0][0]']
                                                                                                      
     activation_191 (Activation)    (None, 42, 42, 336)  0           ['adjust_bn_reduce_6[0][0]']     
                                                                                                      
     separable_conv_1_pad_reduction  (None, 45, 45, 336)  0          ['activation_189[0][0]']         
     _left1_reduce_6 (ZeroPadding2D                                                                   
     )                                                                                                
                                                                                                      
     separable_conv_1_pad_reduction  (None, 47, 47, 336)  0          ['activation_191[0][0]']         
     _right1_reduce_6 (ZeroPadding2                                                                   
     D)                                                                                               
                                                                                                      
     separable_conv_1_reduction_lef  (None, 21, 21, 336)  121296     ['separable_conv_1_pad_reduction_
     t1_reduce_6 (SeparableConv2D)                                   left1_reduce_6[0][0]']           
                                                                                                      
     separable_conv_1_reduction_rig  (None, 21, 21, 336)  129360     ['separable_conv_1_pad_reduction_
     ht1_reduce_6 (SeparableConv2D)                                  right1_reduce_6[0][0]']          
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_1_reduction_left
     left1_reduce_6 (BatchNormaliza                                  1_reduce_6[0][0]']               
     tion)                                                                                            
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_1_reduction_righ
     right1_reduce_6 (BatchNormaliz                                  t1_reduce_6[0][0]']              
     ation)                                                                                           
                                                                                                      
     activation_190 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_reduction_l
                                                                     eft1_reduce_6[0][0]']            
                                                                                                      
     activation_192 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_reduction_r
                                                                     ight1_reduce_6[0][0]']           
                                                                                                      
     separable_conv_2_reduction_lef  (None, 21, 21, 336)  121296     ['activation_190[0][0]']         
     t1_reduce_6 (SeparableConv2D)                                                                    
                                                                                                      
     separable_conv_2_reduction_rig  (None, 21, 21, 336)  129360     ['activation_192[0][0]']         
     ht1_reduce_6 (SeparableConv2D)                                                                   
                                                                                                      
     activation_193 (Activation)    (None, 42, 42, 336)  0           ['adjust_bn_reduce_6[0][0]']     
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_2_reduction_left
     left1_reduce_6 (BatchNormaliza                                  1_reduce_6[0][0]']               
     tion)                                                                                            
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_2_reduction_righ
     right1_reduce_6 (BatchNormaliz                                  t1_reduce_6[0][0]']              
     ation)                                                                                           
                                                                                                      
     separable_conv_1_pad_reduction  (None, 47, 47, 336)  0          ['activation_193[0][0]']         
     _right2_reduce_6 (ZeroPadding2                                                                   
     D)                                                                                               
                                                                                                      
     activation_195 (Activation)    (None, 42, 42, 336)  0           ['adjust_bn_reduce_6[0][0]']     
                                                                                                      
     reduction_add_1_reduce_6 (Add)  (None, 21, 21, 336)  0          ['separable_conv_2_bn_reduction_l
                                                                     eft1_reduce_6[0][0]',            
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight1_reduce_6[0][0]']           
                                                                                                      
     separable_conv_1_reduction_rig  (None, 21, 21, 336)  129360     ['separable_conv_1_pad_reduction_
     ht2_reduce_6 (SeparableConv2D)                                  right2_reduce_6[0][0]']          
                                                                                                      
     separable_conv_1_pad_reduction  (None, 45, 45, 336)  0          ['activation_195[0][0]']         
     _right3_reduce_6 (ZeroPadding2                                                                   
     D)                                                                                               
                                                                                                      
     activation_197 (Activation)    (None, 21, 21, 336)  0           ['reduction_add_1_reduce_6[0][0]'
                                                                     ]                                
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_1_reduction_righ
     right2_reduce_6 (BatchNormaliz                                  t2_reduce_6[0][0]']              
     ation)                                                                                           
                                                                                                      
     separable_conv_1_reduction_rig  (None, 21, 21, 336)  121296     ['separable_conv_1_pad_reduction_
     ht3_reduce_6 (SeparableConv2D)                                  right3_reduce_6[0][0]']          
                                                                                                      
     separable_conv_1_reduction_lef  (None, 21, 21, 336)  115920     ['activation_197[0][0]']         
     t4_reduce_6 (SeparableConv2D)                                                                    
                                                                                                      
     activation_194 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_reduction_r
                                                                     ight2_reduce_6[0][0]']           
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_1_reduction_righ
     right3_reduce_6 (BatchNormaliz                                  t3_reduce_6[0][0]']              
     ation)                                                                                           
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_1_reduction_left
     left4_reduce_6 (BatchNormaliza                                  4_reduce_6[0][0]']               
     tion)                                                                                            
                                                                                                      
     reduction_pad_1_reduce_6 (Zero  (None, 43, 43, 336)  0          ['reduction_bn_1_reduce_6[0][0]']
     Padding2D)                                                                                       
                                                                                                      
     separable_conv_2_reduction_rig  (None, 21, 21, 336)  129360     ['activation_194[0][0]']         
     ht2_reduce_6 (SeparableConv2D)                                                                   
                                                                                                      
     activation_196 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_reduction_r
                                                                     ight3_reduce_6[0][0]']           
                                                                                                      
     activation_198 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_reduction_l
                                                                     eft4_reduce_6[0][0]']            
                                                                                                      
     reduction_left2_reduce_6 (MaxP  (None, 21, 21, 336)  0          ['reduction_pad_1_reduce_6[0][0]'
     ooling2D)                                                       ]                                
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_2_reduction_righ
     right2_reduce_6 (BatchNormaliz                                  t2_reduce_6[0][0]']              
     ation)                                                                                           
                                                                                                      
     separable_conv_2_reduction_rig  (None, 21, 21, 336)  121296     ['activation_196[0][0]']         
     ht3_reduce_6 (SeparableConv2D)                                                                   
                                                                                                      
     separable_conv_2_reduction_lef  (None, 21, 21, 336)  115920     ['activation_198[0][0]']         
     t4_reduce_6 (SeparableConv2D)                                                                    
                                                                                                      
     adjust_relu_1_7 (Activation)   (None, 42, 42, 1008  0           ['normal_concat_4[0][0]']        
                                    )                                                                 
                                                                                                      
     reduction_add_2_reduce_6 (Add)  (None, 21, 21, 336)  0          ['reduction_left2_reduce_6[0][0]'
                                                                     , 'separable_conv_2_bn_reduction_
                                                                     right2_reduce_6[0][0]']          
                                                                                                      
     reduction_left3_reduce_6 (Aver  (None, 21, 21, 336)  0          ['reduction_pad_1_reduce_6[0][0]'
     agePooling2D)                                                   ]                                
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_2_reduction_righ
     right3_reduce_6 (BatchNormaliz                                  t3_reduce_6[0][0]']              
     ation)                                                                                           
                                                                                                      
     reduction_left4_reduce_6 (Aver  (None, 21, 21, 336)  0          ['reduction_add_1_reduce_6[0][0]'
     agePooling2D)                                                   ]                                
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 21, 21, 336)  1344       ['separable_conv_2_reduction_left
     left4_reduce_6 (BatchNormaliza                                  4_reduce_6[0][0]']               
     tion)                                                                                            
                                                                                                      
     reduction_right5_reduce_6 (Max  (None, 21, 21, 336)  0          ['reduction_pad_1_reduce_6[0][0]'
     Pooling2D)                                                      ]                                
                                                                                                      
     zero_padding2d_4 (ZeroPadding2  (None, 43, 43, 1008  0          ['adjust_relu_1_7[0][0]']        
     D)                             )                                                                 
                                                                                                      
     reduction_add3_reduce_6 (Add)  (None, 21, 21, 336)  0           ['reduction_left3_reduce_6[0][0]'
                                                                     , 'separable_conv_2_bn_reduction_
                                                                     right3_reduce_6[0][0]']          
                                                                                                      
     add_14 (Add)                   (None, 21, 21, 336)  0           ['reduction_add_2_reduce_6[0][0]'
                                                                     , 'reduction_left4_reduce_6[0][0]
                                                                     ']                               
                                                                                                      
     reduction_add4_reduce_6 (Add)  (None, 21, 21, 336)  0           ['separable_conv_2_bn_reduction_l
                                                                     eft4_reduce_6[0][0]',            
                                                                      'reduction_right5_reduce_6[0][0]
                                                                     ']                               
                                                                                                      
     cropping2d_2 (Cropping2D)      (None, 42, 42, 1008  0           ['zero_padding2d_4[0][0]']       
                                    )                                                                 
                                                                                                      
     reduction_concat_reduce_6 (Con  (None, 21, 21, 1344  0          ['reduction_add_2_reduce_6[0][0]'
     catenate)                      )                                , 'reduction_add3_reduce_6[0][0]'
                                                                     , 'add_14[0][0]',                
                                                                      'reduction_add4_reduce_6[0][0]']
                                                                                                      
     adjust_avg_pool_1_7 (AveragePo  (None, 21, 21, 1008  0          ['adjust_relu_1_7[0][0]']        
     oling2D)                       )                                                                 
                                                                                                      
     adjust_avg_pool_2_7 (AveragePo  (None, 21, 21, 1008  0          ['cropping2d_2[0][0]']           
     oling2D)                       )                                                                 
                                                                                                      
     adjust_conv_1_7 (Conv2D)       (None, 21, 21, 168)  169344      ['adjust_avg_pool_1_7[0][0]']    
                                                                                                      
     adjust_conv_2_7 (Conv2D)       (None, 21, 21, 168)  169344      ['adjust_avg_pool_2_7[0][0]']    
                                                                                                      
     activation_199 (Activation)    (None, 21, 21, 1344  0           ['reduction_concat_reduce_6[0][0]
                                    )                                ']                               
                                                                                                      
     concatenate_4 (Concatenate)    (None, 21, 21, 336)  0           ['adjust_conv_1_7[0][0]',        
                                                                      'adjust_conv_2_7[0][0]']        
                                                                                                      
     normal_conv_1_7 (Conv2D)       (None, 21, 21, 336)  451584      ['activation_199[0][0]']         
                                                                                                      
     adjust_bn_7 (BatchNormalizatio  (None, 21, 21, 336)  1344       ['concatenate_4[0][0]']          
     n)                                                                                               
                                                                                                      
     normal_bn_1_7 (BatchNormalizat  (None, 21, 21, 336)  1344       ['normal_conv_1_7[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_200 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_7[0][0]']          
                                                                                                      
     activation_202 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_7[0][0]']            
                                                                                                      
     activation_204 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_7[0][0]']            
                                                                                                      
     activation_206 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_7[0][0]']            
                                                                                                      
     activation_208 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_7[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 21, 21, 336)  121296     ['activation_200[0][0]']         
     7 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 21, 21, 336)  115920     ['activation_202[0][0]']         
     _7 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 21, 21, 336)  121296     ['activation_204[0][0]']         
     7 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 21, 21, 336)  115920     ['activation_206[0][0]']         
     _7 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 21, 21, 336)  115920     ['activation_208[0][0]']         
     7 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left1_7
     t1_7 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right1_
     ht1_7 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left2_7
     t2_7 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right2_
     ht2_7 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left5_7
     t5_7 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_201 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     1_7[0][0]']                      
                                                                                                      
     activation_203 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_7[0][0]']                     
                                                                                                      
     activation_205 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     2_7[0][0]']                      
                                                                                                      
     activation_207 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_7[0][0]']                     
                                                                                                      
     activation_209 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     5_7[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 21, 21, 336)  121296     ['activation_201[0][0]']         
     7 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 21, 21, 336)  115920     ['activation_203[0][0]']         
     _7 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 21, 21, 336)  121296     ['activation_205[0][0]']         
     7 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 21, 21, 336)  115920     ['activation_207[0][0]']         
     _7 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 21, 21, 336)  115920     ['activation_209[0][0]']         
     7 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left1_7
     t1_7 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right1_
     ht1_7 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left2_7
     t2_7 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right2_
     ht2_7 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     normal_left3_7 (AveragePooling  (None, 21, 21, 336)  0          ['normal_bn_1_7[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_7 (AveragePooling  (None, 21, 21, 336)  0          ['adjust_bn_7[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_7 (AveragePoolin  (None, 21, 21, 336)  0          ['adjust_bn_7[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left5_7
     t5_7 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_7 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     1_7[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_7[0][0]']                     
                                                                                                      
     normal_add_2_7 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     2_7[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_7[0][0]']                     
                                                                                                      
     normal_add_3_7 (Add)           (None, 21, 21, 336)  0           ['normal_left3_7[0][0]',         
                                                                      'adjust_bn_7[0][0]']            
                                                                                                      
     normal_add_4_7 (Add)           (None, 21, 21, 336)  0           ['normal_left4_7[0][0]',         
                                                                      'normal_right4_7[0][0]']        
                                                                                                      
     normal_add_5_7 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     5_7[0][0]',                      
                                                                      'normal_bn_1_7[0][0]']          
                                                                                                      
     normal_concat_7 (Concatenate)  (None, 21, 21, 2016  0           ['adjust_bn_7[0][0]',            
                                    )                                 'normal_add_1_7[0][0]',         
                                                                      'normal_add_2_7[0][0]',         
                                                                      'normal_add_3_7[0][0]',         
                                                                      'normal_add_4_7[0][0]',         
                                                                      'normal_add_5_7[0][0]']         
                                                                                                      
     activation_210 (Activation)    (None, 21, 21, 1344  0           ['reduction_concat_reduce_6[0][0]
                                    )                                ']                               
                                                                                                      
     activation_211 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_7[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_8 (Conv  (None, 21, 21, 336)  451584     ['activation_210[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_8 (Conv2D)       (None, 21, 21, 336)  677376      ['activation_211[0][0]']         
                                                                                                      
     adjust_bn_8 (BatchNormalizatio  (None, 21, 21, 336)  1344       ['adjust_conv_projection_8[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_8 (BatchNormalizat  (None, 21, 21, 336)  1344       ['normal_conv_1_8[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_212 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_8[0][0]']          
                                                                                                      
     activation_214 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_8[0][0]']            
                                                                                                      
     activation_216 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_8[0][0]']            
                                                                                                      
     activation_218 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_8[0][0]']            
                                                                                                      
     activation_220 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_8[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 21, 21, 336)  121296     ['activation_212[0][0]']         
     8 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 21, 21, 336)  115920     ['activation_214[0][0]']         
     _8 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 21, 21, 336)  121296     ['activation_216[0][0]']         
     8 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 21, 21, 336)  115920     ['activation_218[0][0]']         
     _8 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 21, 21, 336)  115920     ['activation_220[0][0]']         
     8 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left1_8
     t1_8 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right1_
     ht1_8 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left2_8
     t2_8 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right2_
     ht2_8 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left5_8
     t5_8 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_213 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     1_8[0][0]']                      
                                                                                                      
     activation_215 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_8[0][0]']                     
                                                                                                      
     activation_217 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     2_8[0][0]']                      
                                                                                                      
     activation_219 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_8[0][0]']                     
                                                                                                      
     activation_221 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     5_8[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 21, 21, 336)  121296     ['activation_213[0][0]']         
     8 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 21, 21, 336)  115920     ['activation_215[0][0]']         
     _8 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 21, 21, 336)  121296     ['activation_217[0][0]']         
     8 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 21, 21, 336)  115920     ['activation_219[0][0]']         
     _8 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 21, 21, 336)  115920     ['activation_221[0][0]']         
     8 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left1_8
     t1_8 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right1_
     ht1_8 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left2_8
     t2_8 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right2_
     ht2_8 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     normal_left3_8 (AveragePooling  (None, 21, 21, 336)  0          ['normal_bn_1_8[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_8 (AveragePooling  (None, 21, 21, 336)  0          ['adjust_bn_8[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_8 (AveragePoolin  (None, 21, 21, 336)  0          ['adjust_bn_8[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left5_8
     t5_8 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_8 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     1_8[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_8[0][0]']                     
                                                                                                      
     normal_add_2_8 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     2_8[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_8[0][0]']                     
                                                                                                      
     normal_add_3_8 (Add)           (None, 21, 21, 336)  0           ['normal_left3_8[0][0]',         
                                                                      'adjust_bn_8[0][0]']            
                                                                                                      
     normal_add_4_8 (Add)           (None, 21, 21, 336)  0           ['normal_left4_8[0][0]',         
                                                                      'normal_right4_8[0][0]']        
                                                                                                      
     normal_add_5_8 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     5_8[0][0]',                      
                                                                      'normal_bn_1_8[0][0]']          
                                                                                                      
     normal_concat_8 (Concatenate)  (None, 21, 21, 2016  0           ['adjust_bn_8[0][0]',            
                                    )                                 'normal_add_1_8[0][0]',         
                                                                      'normal_add_2_8[0][0]',         
                                                                      'normal_add_3_8[0][0]',         
                                                                      'normal_add_4_8[0][0]',         
                                                                      'normal_add_5_8[0][0]']         
                                                                                                      
     activation_222 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_7[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_223 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_8[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_9 (Conv  (None, 21, 21, 336)  677376     ['activation_222[0][0]']         
     2D)                                                                                              
                                                                                                      
     normal_conv_1_9 (Conv2D)       (None, 21, 21, 336)  677376      ['activation_223[0][0]']         
                                                                                                      
     adjust_bn_9 (BatchNormalizatio  (None, 21, 21, 336)  1344       ['adjust_conv_projection_9[0][0]'
     n)                                                              ]                                
                                                                                                      
     normal_bn_1_9 (BatchNormalizat  (None, 21, 21, 336)  1344       ['normal_conv_1_9[0][0]']        
     ion)                                                                                             
                                                                                                      
     activation_224 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_9[0][0]']          
                                                                                                      
     activation_226 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_9[0][0]']            
                                                                                                      
     activation_228 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_9[0][0]']            
                                                                                                      
     activation_230 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_9[0][0]']            
                                                                                                      
     activation_232 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_9[0][0]']          
                                                                                                      
     separable_conv_1_normal_left1_  (None, 21, 21, 336)  121296     ['activation_224[0][0]']         
     9 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right1  (None, 21, 21, 336)  115920     ['activation_226[0][0]']         
     _9 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left2_  (None, 21, 21, 336)  121296     ['activation_228[0][0]']         
     9 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_normal_right2  (None, 21, 21, 336)  115920     ['activation_230[0][0]']         
     _9 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_left5_  (None, 21, 21, 336)  115920     ['activation_232[0][0]']         
     9 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left1_9
     t1_9 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right1_
     ht1_9 (BatchNormalization)                                      9[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left2_9
     t2_9 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right2_
     ht2_9 (BatchNormalization)                                      9[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left5_9
     t5_9 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     activation_225 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     1_9[0][0]']                      
                                                                                                      
     activation_227 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_9[0][0]']                     
                                                                                                      
     activation_229 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     2_9[0][0]']                      
                                                                                                      
     activation_231 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_9[0][0]']                     
                                                                                                      
     activation_233 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     5_9[0][0]']                      
                                                                                                      
     separable_conv_2_normal_left1_  (None, 21, 21, 336)  121296     ['activation_225[0][0]']         
     9 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right1  (None, 21, 21, 336)  115920     ['activation_227[0][0]']         
     _9 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left2_  (None, 21, 21, 336)  121296     ['activation_229[0][0]']         
     9 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_normal_right2  (None, 21, 21, 336)  115920     ['activation_231[0][0]']         
     _9 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_left5_  (None, 21, 21, 336)  115920     ['activation_233[0][0]']         
     9 (SeparableConv2D)                                                                              
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left1_9
     t1_9 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right1_
     ht1_9 (BatchNormalization)                                      9[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left2_9
     t2_9 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right2_
     ht2_9 (BatchNormalization)                                      9[0][0]']                        
                                                                                                      
     normal_left3_9 (AveragePooling  (None, 21, 21, 336)  0          ['normal_bn_1_9[0][0]']          
     2D)                                                                                              
                                                                                                      
     normal_left4_9 (AveragePooling  (None, 21, 21, 336)  0          ['adjust_bn_9[0][0]']            
     2D)                                                                                              
                                                                                                      
     normal_right4_9 (AveragePoolin  (None, 21, 21, 336)  0          ['adjust_bn_9[0][0]']            
     g2D)                                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left5_9
     t5_9 (BatchNormalization)                                       [0][0]']                         
                                                                                                      
     normal_add_1_9 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     1_9[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_9[0][0]']                     
                                                                                                      
     normal_add_2_9 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     2_9[0][0]',                      
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_9[0][0]']                     
                                                                                                      
     normal_add_3_9 (Add)           (None, 21, 21, 336)  0           ['normal_left3_9[0][0]',         
                                                                      'adjust_bn_9[0][0]']            
                                                                                                      
     normal_add_4_9 (Add)           (None, 21, 21, 336)  0           ['normal_left4_9[0][0]',         
                                                                      'normal_right4_9[0][0]']        
                                                                                                      
     normal_add_5_9 (Add)           (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     5_9[0][0]',                      
                                                                      'normal_bn_1_9[0][0]']          
                                                                                                      
     normal_concat_9 (Concatenate)  (None, 21, 21, 2016  0           ['adjust_bn_9[0][0]',            
                                    )                                 'normal_add_1_9[0][0]',         
                                                                      'normal_add_2_9[0][0]',         
                                                                      'normal_add_3_9[0][0]',         
                                                                      'normal_add_4_9[0][0]',         
                                                                      'normal_add_5_9[0][0]']         
                                                                                                      
     activation_234 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_8[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_235 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_9[0][0]']        
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_10 (Con  (None, 21, 21, 336)  677376     ['activation_234[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_10 (Conv2D)      (None, 21, 21, 336)  677376      ['activation_235[0][0]']         
                                                                                                      
     adjust_bn_10 (BatchNormalizati  (None, 21, 21, 336)  1344       ['adjust_conv_projection_10[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_10 (BatchNormaliza  (None, 21, 21, 336)  1344       ['normal_conv_1_10[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_236 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_10[0][0]']         
                                                                                                      
     activation_238 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_10[0][0]']           
                                                                                                      
     activation_240 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_10[0][0]']           
                                                                                                      
     activation_242 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_10[0][0]']           
                                                                                                      
     activation_244 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_10[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 21, 21, 336)  121296     ['activation_236[0][0]']         
     10 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 21, 21, 336)  115920     ['activation_238[0][0]']         
     _10 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 21, 21, 336)  121296     ['activation_240[0][0]']         
     10 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 21, 21, 336)  115920     ['activation_242[0][0]']         
     _10 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 21, 21, 336)  115920     ['activation_244[0][0]']         
     10 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left1_1
     t1_10 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right1_
     ht1_10 (BatchNormalization)                                     10[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left2_1
     t2_10 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right2_
     ht2_10 (BatchNormalization)                                     10[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left5_1
     t5_10 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     activation_237 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     1_10[0][0]']                     
                                                                                                      
     activation_239 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_10[0][0]']                    
                                                                                                      
     activation_241 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     2_10[0][0]']                     
                                                                                                      
     activation_243 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_10[0][0]']                    
                                                                                                      
     activation_245 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     5_10[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 21, 21, 336)  121296     ['activation_237[0][0]']         
     10 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 21, 21, 336)  115920     ['activation_239[0][0]']         
     _10 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 21, 21, 336)  121296     ['activation_241[0][0]']         
     10 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 21, 21, 336)  115920     ['activation_243[0][0]']         
     _10 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 21, 21, 336)  115920     ['activation_245[0][0]']         
     10 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left1_1
     t1_10 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right1_
     ht1_10 (BatchNormalization)                                     10[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left2_1
     t2_10 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right2_
     ht2_10 (BatchNormalization)                                     10[0][0]']                       
                                                                                                      
     normal_left3_10 (AveragePoolin  (None, 21, 21, 336)  0          ['normal_bn_1_10[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_10 (AveragePoolin  (None, 21, 21, 336)  0          ['adjust_bn_10[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_10 (AveragePooli  (None, 21, 21, 336)  0          ['adjust_bn_10[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left5_1
     t5_10 (BatchNormalization)                                      0[0][0]']                        
                                                                                                      
     normal_add_1_10 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     1_10[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_10[0][0]']                    
                                                                                                      
     normal_add_2_10 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     2_10[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_10[0][0]']                    
                                                                                                      
     normal_add_3_10 (Add)          (None, 21, 21, 336)  0           ['normal_left3_10[0][0]',        
                                                                      'adjust_bn_10[0][0]']           
                                                                                                      
     normal_add_4_10 (Add)          (None, 21, 21, 336)  0           ['normal_left4_10[0][0]',        
                                                                      'normal_right4_10[0][0]']       
                                                                                                      
     normal_add_5_10 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     5_10[0][0]',                     
                                                                      'normal_bn_1_10[0][0]']         
                                                                                                      
     normal_concat_10 (Concatenate)  (None, 21, 21, 2016  0          ['adjust_bn_10[0][0]',           
                                    )                                 'normal_add_1_10[0][0]',        
                                                                      'normal_add_2_10[0][0]',        
                                                                      'normal_add_3_10[0][0]',        
                                                                      'normal_add_4_10[0][0]',        
                                                                      'normal_add_5_10[0][0]']        
                                                                                                      
     activation_246 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_9[0][0]']        
                                    )                                                                 
                                                                                                      
     activation_247 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_10[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_11 (Con  (None, 21, 21, 336)  677376     ['activation_246[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_11 (Conv2D)      (None, 21, 21, 336)  677376      ['activation_247[0][0]']         
                                                                                                      
     adjust_bn_11 (BatchNormalizati  (None, 21, 21, 336)  1344       ['adjust_conv_projection_11[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_11 (BatchNormaliza  (None, 21, 21, 336)  1344       ['normal_conv_1_11[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_248 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_11[0][0]']         
                                                                                                      
     activation_250 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_11[0][0]']           
                                                                                                      
     activation_252 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_11[0][0]']           
                                                                                                      
     activation_254 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_11[0][0]']           
                                                                                                      
     activation_256 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_11[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 21, 21, 336)  121296     ['activation_248[0][0]']         
     11 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 21, 21, 336)  115920     ['activation_250[0][0]']         
     _11 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 21, 21, 336)  121296     ['activation_252[0][0]']         
     11 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 21, 21, 336)  115920     ['activation_254[0][0]']         
     _11 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 21, 21, 336)  115920     ['activation_256[0][0]']         
     11 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left1_1
     t1_11 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right1_
     ht1_11 (BatchNormalization)                                     11[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left2_1
     t2_11 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right2_
     ht2_11 (BatchNormalization)                                     11[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left5_1
     t5_11 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     activation_249 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     1_11[0][0]']                     
                                                                                                      
     activation_251 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_11[0][0]']                    
                                                                                                      
     activation_253 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     2_11[0][0]']                     
                                                                                                      
     activation_255 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_11[0][0]']                    
                                                                                                      
     activation_257 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     5_11[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 21, 21, 336)  121296     ['activation_249[0][0]']         
     11 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 21, 21, 336)  115920     ['activation_251[0][0]']         
     _11 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 21, 21, 336)  121296     ['activation_253[0][0]']         
     11 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 21, 21, 336)  115920     ['activation_255[0][0]']         
     _11 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 21, 21, 336)  115920     ['activation_257[0][0]']         
     11 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left1_1
     t1_11 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right1_
     ht1_11 (BatchNormalization)                                     11[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left2_1
     t2_11 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right2_
     ht2_11 (BatchNormalization)                                     11[0][0]']                       
                                                                                                      
     normal_left3_11 (AveragePoolin  (None, 21, 21, 336)  0          ['normal_bn_1_11[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_11 (AveragePoolin  (None, 21, 21, 336)  0          ['adjust_bn_11[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_11 (AveragePooli  (None, 21, 21, 336)  0          ['adjust_bn_11[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left5_1
     t5_11 (BatchNormalization)                                      1[0][0]']                        
                                                                                                      
     normal_add_1_11 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     1_11[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_11[0][0]']                    
                                                                                                      
     normal_add_2_11 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     2_11[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_11[0][0]']                    
                                                                                                      
     normal_add_3_11 (Add)          (None, 21, 21, 336)  0           ['normal_left3_11[0][0]',        
                                                                      'adjust_bn_11[0][0]']           
                                                                                                      
     normal_add_4_11 (Add)          (None, 21, 21, 336)  0           ['normal_left4_11[0][0]',        
                                                                      'normal_right4_11[0][0]']       
                                                                                                      
     normal_add_5_11 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     5_11[0][0]',                     
                                                                      'normal_bn_1_11[0][0]']         
                                                                                                      
     normal_concat_11 (Concatenate)  (None, 21, 21, 2016  0          ['adjust_bn_11[0][0]',           
                                    )                                 'normal_add_1_11[0][0]',        
                                                                      'normal_add_2_11[0][0]',        
                                                                      'normal_add_3_11[0][0]',        
                                                                      'normal_add_4_11[0][0]',        
                                                                      'normal_add_5_11[0][0]']        
                                                                                                      
     activation_258 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_10[0][0]']       
                                    )                                                                 
                                                                                                      
     activation_259 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_11[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_12 (Con  (None, 21, 21, 336)  677376     ['activation_258[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_12 (Conv2D)      (None, 21, 21, 336)  677376      ['activation_259[0][0]']         
                                                                                                      
     adjust_bn_12 (BatchNormalizati  (None, 21, 21, 336)  1344       ['adjust_conv_projection_12[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_12 (BatchNormaliza  (None, 21, 21, 336)  1344       ['normal_conv_1_12[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_260 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_12[0][0]']         
                                                                                                      
     activation_262 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_12[0][0]']           
                                                                                                      
     activation_264 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_12[0][0]']           
                                                                                                      
     activation_266 (Activation)    (None, 21, 21, 336)  0           ['adjust_bn_12[0][0]']           
                                                                                                      
     activation_268 (Activation)    (None, 21, 21, 336)  0           ['normal_bn_1_12[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 21, 21, 336)  121296     ['activation_260[0][0]']         
     12 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 21, 21, 336)  115920     ['activation_262[0][0]']         
     _12 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 21, 21, 336)  121296     ['activation_264[0][0]']         
     12 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 21, 21, 336)  115920     ['activation_266[0][0]']         
     _12 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 21, 21, 336)  115920     ['activation_268[0][0]']         
     12 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left1_1
     t1_12 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right1_
     ht1_12 (BatchNormalization)                                     12[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left2_1
     t2_12 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_right2_
     ht2_12 (BatchNormalization)                                     12[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_1_normal_left5_1
     t5_12 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     activation_261 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     1_12[0][0]']                     
                                                                                                      
     activation_263 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_12[0][0]']                    
                                                                                                      
     activation_265 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     2_12[0][0]']                     
                                                                                                      
     activation_267 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_12[0][0]']                    
                                                                                                      
     activation_269 (Activation)    (None, 21, 21, 336)  0           ['separable_conv_1_bn_normal_left
                                                                     5_12[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 21, 21, 336)  121296     ['activation_261[0][0]']         
     12 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 21, 21, 336)  115920     ['activation_263[0][0]']         
     _12 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 21, 21, 336)  121296     ['activation_265[0][0]']         
     12 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 21, 21, 336)  115920     ['activation_267[0][0]']         
     _12 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 21, 21, 336)  115920     ['activation_269[0][0]']         
     12 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left1_1
     t1_12 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right1_
     ht1_12 (BatchNormalization)                                     12[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left2_1
     t2_12 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_right2_
     ht2_12 (BatchNormalization)                                     12[0][0]']                       
                                                                                                      
     normal_left3_12 (AveragePoolin  (None, 21, 21, 336)  0          ['normal_bn_1_12[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_12 (AveragePoolin  (None, 21, 21, 336)  0          ['adjust_bn_12[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_12 (AveragePooli  (None, 21, 21, 336)  0          ['adjust_bn_12[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 21, 21, 336)  1344       ['separable_conv_2_normal_left5_1
     t5_12 (BatchNormalization)                                      2[0][0]']                        
                                                                                                      
     normal_add_1_12 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     1_12[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_12[0][0]']                    
                                                                                                      
     normal_add_2_12 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     2_12[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_12[0][0]']                    
                                                                                                      
     normal_add_3_12 (Add)          (None, 21, 21, 336)  0           ['normal_left3_12[0][0]',        
                                                                      'adjust_bn_12[0][0]']           
                                                                                                      
     normal_add_4_12 (Add)          (None, 21, 21, 336)  0           ['normal_left4_12[0][0]',        
                                                                      'normal_right4_12[0][0]']       
                                                                                                      
     normal_add_5_12 (Add)          (None, 21, 21, 336)  0           ['separable_conv_2_bn_normal_left
                                                                     5_12[0][0]',                     
                                                                      'normal_bn_1_12[0][0]']         
                                                                                                      
     normal_concat_12 (Concatenate)  (None, 21, 21, 2016  0          ['adjust_bn_12[0][0]',           
                                    )                                 'normal_add_1_12[0][0]',        
                                                                      'normal_add_2_12[0][0]',        
                                                                      'normal_add_3_12[0][0]',        
                                                                      'normal_add_4_12[0][0]',        
                                                                      'normal_add_5_12[0][0]']        
                                                                                                      
     activation_271 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_12[0][0]']       
                                    )                                                                 
                                                                                                      
     activation_270 (Activation)    (None, 21, 21, 2016  0           ['normal_concat_11[0][0]']       
                                    )                                                                 
                                                                                                      
     reduction_conv_1_reduce_12 (Co  (None, 21, 21, 672)  1354752    ['activation_271[0][0]']         
     nv2D)                                                                                            
                                                                                                      
     adjust_conv_projection_reduce_  (None, 21, 21, 672)  1354752    ['activation_270[0][0]']         
     12 (Conv2D)                                                                                      
                                                                                                      
     reduction_bn_1_reduce_12 (Batc  (None, 21, 21, 672)  2688       ['reduction_conv_1_reduce_12[0][0
     hNormalization)                                                 ]']                              
                                                                                                      
     adjust_bn_reduce_12 (BatchNorm  (None, 21, 21, 672)  2688       ['adjust_conv_projection_reduce_1
     alization)                                                      2[0][0]']                        
                                                                                                      
     activation_272 (Activation)    (None, 21, 21, 672)  0           ['reduction_bn_1_reduce_12[0][0]'
                                                                     ]                                
                                                                                                      
     activation_274 (Activation)    (None, 21, 21, 672)  0           ['adjust_bn_reduce_12[0][0]']    
                                                                                                      
     separable_conv_1_pad_reduction  (None, 25, 25, 672)  0          ['activation_272[0][0]']         
     _left1_reduce_12 (ZeroPadding2                                                                   
     D)                                                                                               
                                                                                                      
     separable_conv_1_pad_reduction  (None, 27, 27, 672)  0          ['activation_274[0][0]']         
     _right1_reduce_12 (ZeroPadding                                                                   
     2D)                                                                                              
                                                                                                      
     separable_conv_1_reduction_lef  (None, 11, 11, 672)  468384     ['separable_conv_1_pad_reduction_
     t1_reduce_12 (SeparableConv2D)                                  left1_reduce_12[0][0]']          
                                                                                                      
     separable_conv_1_reduction_rig  (None, 11, 11, 672)  484512     ['separable_conv_1_pad_reduction_
     ht1_reduce_12 (SeparableConv2D                                  right1_reduce_12[0][0]']         
     )                                                                                                
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_1_reduction_left
     left1_reduce_12 (BatchNormaliz                                  1_reduce_12[0][0]']              
     ation)                                                                                           
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_1_reduction_righ
     right1_reduce_12 (BatchNormali                                  t1_reduce_12[0][0]']             
     zation)                                                                                          
                                                                                                      
     activation_273 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_reduction_l
                                                                     eft1_reduce_12[0][0]']           
                                                                                                      
     activation_275 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_reduction_r
                                                                     ight1_reduce_12[0][0]']          
                                                                                                      
     separable_conv_2_reduction_lef  (None, 11, 11, 672)  468384     ['activation_273[0][0]']         
     t1_reduce_12 (SeparableConv2D)                                                                   
                                                                                                      
     separable_conv_2_reduction_rig  (None, 11, 11, 672)  484512     ['activation_275[0][0]']         
     ht1_reduce_12 (SeparableConv2D                                                                   
     )                                                                                                
                                                                                                      
     activation_276 (Activation)    (None, 21, 21, 672)  0           ['adjust_bn_reduce_12[0][0]']    
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_2_reduction_left
     left1_reduce_12 (BatchNormaliz                                  1_reduce_12[0][0]']              
     ation)                                                                                           
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_2_reduction_righ
     right1_reduce_12 (BatchNormali                                  t1_reduce_12[0][0]']             
     zation)                                                                                          
                                                                                                      
     separable_conv_1_pad_reduction  (None, 27, 27, 672)  0          ['activation_276[0][0]']         
     _right2_reduce_12 (ZeroPadding                                                                   
     2D)                                                                                              
                                                                                                      
     activation_278 (Activation)    (None, 21, 21, 672)  0           ['adjust_bn_reduce_12[0][0]']    
                                                                                                      
     reduction_add_1_reduce_12 (Add  (None, 11, 11, 672)  0          ['separable_conv_2_bn_reduction_l
     )                                                               eft1_reduce_12[0][0]',           
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight1_reduce_12[0][0]']          
                                                                                                      
     separable_conv_1_reduction_rig  (None, 11, 11, 672)  484512     ['separable_conv_1_pad_reduction_
     ht2_reduce_12 (SeparableConv2D                                  right2_reduce_12[0][0]']         
     )                                                                                                
                                                                                                      
     separable_conv_1_pad_reduction  (None, 25, 25, 672)  0          ['activation_278[0][0]']         
     _right3_reduce_12 (ZeroPadding                                                                   
     2D)                                                                                              
                                                                                                      
     activation_280 (Activation)    (None, 11, 11, 672)  0           ['reduction_add_1_reduce_12[0][0]
                                                                     ']                               
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_1_reduction_righ
     right2_reduce_12 (BatchNormali                                  t2_reduce_12[0][0]']             
     zation)                                                                                          
                                                                                                      
     separable_conv_1_reduction_rig  (None, 11, 11, 672)  468384     ['separable_conv_1_pad_reduction_
     ht3_reduce_12 (SeparableConv2D                                  right3_reduce_12[0][0]']         
     )                                                                                                
                                                                                                      
     separable_conv_1_reduction_lef  (None, 11, 11, 672)  457632     ['activation_280[0][0]']         
     t4_reduce_12 (SeparableConv2D)                                                                   
                                                                                                      
     activation_277 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_reduction_r
                                                                     ight2_reduce_12[0][0]']          
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_1_reduction_righ
     right3_reduce_12 (BatchNormali                                  t3_reduce_12[0][0]']             
     zation)                                                                                          
                                                                                                      
     separable_conv_1_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_1_reduction_left
     left4_reduce_12 (BatchNormaliz                                  4_reduce_12[0][0]']              
     ation)                                                                                           
                                                                                                      
     reduction_pad_1_reduce_12 (Zer  (None, 23, 23, 672)  0          ['reduction_bn_1_reduce_12[0][0]'
     oPadding2D)                                                     ]                                
                                                                                                      
     separable_conv_2_reduction_rig  (None, 11, 11, 672)  484512     ['activation_277[0][0]']         
     ht2_reduce_12 (SeparableConv2D                                                                   
     )                                                                                                
                                                                                                      
     activation_279 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_reduction_r
                                                                     ight3_reduce_12[0][0]']          
                                                                                                      
     activation_281 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_reduction_l
                                                                     eft4_reduce_12[0][0]']           
                                                                                                      
     reduction_left2_reduce_12 (Max  (None, 11, 11, 672)  0          ['reduction_pad_1_reduce_12[0][0]
     Pooling2D)                                                      ']                               
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_2_reduction_righ
     right2_reduce_12 (BatchNormali                                  t2_reduce_12[0][0]']             
     zation)                                                                                          
                                                                                                      
     separable_conv_2_reduction_rig  (None, 11, 11, 672)  468384     ['activation_279[0][0]']         
     ht3_reduce_12 (SeparableConv2D                                                                   
     )                                                                                                
                                                                                                      
     separable_conv_2_reduction_lef  (None, 11, 11, 672)  457632     ['activation_281[0][0]']         
     t4_reduce_12 (SeparableConv2D)                                                                   
                                                                                                      
     adjust_relu_1_13 (Activation)  (None, 21, 21, 2016  0           ['normal_concat_11[0][0]']       
                                    )                                                                 
                                                                                                      
     reduction_add_2_reduce_12 (Add  (None, 11, 11, 672)  0          ['reduction_left2_reduce_12[0][0]
     )                                                               ',                               
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight2_reduce_12[0][0]']          
                                                                                                      
     reduction_left3_reduce_12 (Ave  (None, 11, 11, 672)  0          ['reduction_pad_1_reduce_12[0][0]
     ragePooling2D)                                                  ']                               
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_2_reduction_righ
     right3_reduce_12 (BatchNormali                                  t3_reduce_12[0][0]']             
     zation)                                                                                          
                                                                                                      
     reduction_left4_reduce_12 (Ave  (None, 11, 11, 672)  0          ['reduction_add_1_reduce_12[0][0]
     ragePooling2D)                                                  ']                               
                                                                                                      
     separable_conv_2_bn_reduction_  (None, 11, 11, 672)  2688       ['separable_conv_2_reduction_left
     left4_reduce_12 (BatchNormaliz                                  4_reduce_12[0][0]']              
     ation)                                                                                           
                                                                                                      
     reduction_right5_reduce_12 (Ma  (None, 11, 11, 672)  0          ['reduction_pad_1_reduce_12[0][0]
     xPooling2D)                                                     ']                               
                                                                                                      
     zero_padding2d_5 (ZeroPadding2  (None, 22, 22, 2016  0          ['adjust_relu_1_13[0][0]']       
     D)                             )                                                                 
                                                                                                      
     reduction_add3_reduce_12 (Add)  (None, 11, 11, 672)  0          ['reduction_left3_reduce_12[0][0]
                                                                     ',                               
                                                                      'separable_conv_2_bn_reduction_r
                                                                     ight3_reduce_12[0][0]']          
                                                                                                      
     add_15 (Add)                   (None, 11, 11, 672)  0           ['reduction_add_2_reduce_12[0][0]
                                                                     ',                               
                                                                      'reduction_left4_reduce_12[0][0]
                                                                     ']                               
                                                                                                      
     reduction_add4_reduce_12 (Add)  (None, 11, 11, 672)  0          ['separable_conv_2_bn_reduction_l
                                                                     eft4_reduce_12[0][0]',           
                                                                      'reduction_right5_reduce_12[0][0
                                                                     ]']                              
                                                                                                      
     cropping2d_3 (Cropping2D)      (None, 21, 21, 2016  0           ['zero_padding2d_5[0][0]']       
                                    )                                                                 
                                                                                                      
     reduction_concat_reduce_12 (Co  (None, 11, 11, 2688  0          ['reduction_add_2_reduce_12[0][0]
     ncatenate)                     )                                ',                               
                                                                      'reduction_add3_reduce_12[0][0]'
                                                                     , 'add_15[0][0]',                
                                                                      'reduction_add4_reduce_12[0][0]'
                                                                     ]                                
                                                                                                      
     adjust_avg_pool_1_13 (AverageP  (None, 11, 11, 2016  0          ['adjust_relu_1_13[0][0]']       
     ooling2D)                      )                                                                 
                                                                                                      
     adjust_avg_pool_2_13 (AverageP  (None, 11, 11, 2016  0          ['cropping2d_3[0][0]']           
     ooling2D)                      )                                                                 
                                                                                                      
     adjust_conv_1_13 (Conv2D)      (None, 11, 11, 336)  677376      ['adjust_avg_pool_1_13[0][0]']   
                                                                                                      
     adjust_conv_2_13 (Conv2D)      (None, 11, 11, 336)  677376      ['adjust_avg_pool_2_13[0][0]']   
                                                                                                      
     activation_282 (Activation)    (None, 11, 11, 2688  0           ['reduction_concat_reduce_12[0][0
                                    )                                ]']                              
                                                                                                      
     concatenate_5 (Concatenate)    (None, 11, 11, 672)  0           ['adjust_conv_1_13[0][0]',       
                                                                      'adjust_conv_2_13[0][0]']       
                                                                                                      
     normal_conv_1_13 (Conv2D)      (None, 11, 11, 672)  1806336     ['activation_282[0][0]']         
                                                                                                      
     adjust_bn_13 (BatchNormalizati  (None, 11, 11, 672)  2688       ['concatenate_5[0][0]']          
     on)                                                                                              
                                                                                                      
     normal_bn_1_13 (BatchNormaliza  (None, 11, 11, 672)  2688       ['normal_conv_1_13[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_283 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_13[0][0]']         
                                                                                                      
     activation_285 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_13[0][0]']           
                                                                                                      
     activation_287 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_13[0][0]']           
                                                                                                      
     activation_289 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_13[0][0]']           
                                                                                                      
     activation_291 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_13[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 11, 11, 672)  468384     ['activation_283[0][0]']         
     13 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 11, 11, 672)  457632     ['activation_285[0][0]']         
     _13 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 11, 11, 672)  468384     ['activation_287[0][0]']         
     13 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 11, 11, 672)  457632     ['activation_289[0][0]']         
     _13 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 11, 11, 672)  457632     ['activation_291[0][0]']         
     13 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left1_1
     t1_13 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right1_
     ht1_13 (BatchNormalization)                                     13[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left2_1
     t2_13 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right2_
     ht2_13 (BatchNormalization)                                     13[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left5_1
     t5_13 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     activation_284 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     1_13[0][0]']                     
                                                                                                      
     activation_286 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_13[0][0]']                    
                                                                                                      
     activation_288 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     2_13[0][0]']                     
                                                                                                      
     activation_290 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_13[0][0]']                    
                                                                                                      
     activation_292 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     5_13[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 11, 11, 672)  468384     ['activation_284[0][0]']         
     13 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 11, 11, 672)  457632     ['activation_286[0][0]']         
     _13 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 11, 11, 672)  468384     ['activation_288[0][0]']         
     13 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 11, 11, 672)  457632     ['activation_290[0][0]']         
     _13 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 11, 11, 672)  457632     ['activation_292[0][0]']         
     13 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left1_1
     t1_13 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right1_
     ht1_13 (BatchNormalization)                                     13[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left2_1
     t2_13 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right2_
     ht2_13 (BatchNormalization)                                     13[0][0]']                       
                                                                                                      
     normal_left3_13 (AveragePoolin  (None, 11, 11, 672)  0          ['normal_bn_1_13[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_13 (AveragePoolin  (None, 11, 11, 672)  0          ['adjust_bn_13[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_13 (AveragePooli  (None, 11, 11, 672)  0          ['adjust_bn_13[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left5_1
     t5_13 (BatchNormalization)                                      3[0][0]']                        
                                                                                                      
     normal_add_1_13 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     1_13[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_13[0][0]']                    
                                                                                                      
     normal_add_2_13 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     2_13[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_13[0][0]']                    
                                                                                                      
     normal_add_3_13 (Add)          (None, 11, 11, 672)  0           ['normal_left3_13[0][0]',        
                                                                      'adjust_bn_13[0][0]']           
                                                                                                      
     normal_add_4_13 (Add)          (None, 11, 11, 672)  0           ['normal_left4_13[0][0]',        
                                                                      'normal_right4_13[0][0]']       
                                                                                                      
     normal_add_5_13 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     5_13[0][0]',                     
                                                                      'normal_bn_1_13[0][0]']         
                                                                                                      
     normal_concat_13 (Concatenate)  (None, 11, 11, 4032  0          ['adjust_bn_13[0][0]',           
                                    )                                 'normal_add_1_13[0][0]',        
                                                                      'normal_add_2_13[0][0]',        
                                                                      'normal_add_3_13[0][0]',        
                                                                      'normal_add_4_13[0][0]',        
                                                                      'normal_add_5_13[0][0]']        
                                                                                                      
     activation_293 (Activation)    (None, 11, 11, 2688  0           ['reduction_concat_reduce_12[0][0
                                    )                                ]']                              
                                                                                                      
     activation_294 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_13[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_14 (Con  (None, 11, 11, 672)  1806336    ['activation_293[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_14 (Conv2D)      (None, 11, 11, 672)  2709504     ['activation_294[0][0]']         
                                                                                                      
     adjust_bn_14 (BatchNormalizati  (None, 11, 11, 672)  2688       ['adjust_conv_projection_14[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_14 (BatchNormaliza  (None, 11, 11, 672)  2688       ['normal_conv_1_14[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_295 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_14[0][0]']         
                                                                                                      
     activation_297 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_14[0][0]']           
                                                                                                      
     activation_299 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_14[0][0]']           
                                                                                                      
     activation_301 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_14[0][0]']           
                                                                                                      
     activation_303 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_14[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 11, 11, 672)  468384     ['activation_295[0][0]']         
     14 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 11, 11, 672)  457632     ['activation_297[0][0]']         
     _14 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 11, 11, 672)  468384     ['activation_299[0][0]']         
     14 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 11, 11, 672)  457632     ['activation_301[0][0]']         
     _14 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 11, 11, 672)  457632     ['activation_303[0][0]']         
     14 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left1_1
     t1_14 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right1_
     ht1_14 (BatchNormalization)                                     14[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left2_1
     t2_14 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right2_
     ht2_14 (BatchNormalization)                                     14[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left5_1
     t5_14 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     activation_296 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     1_14[0][0]']                     
                                                                                                      
     activation_298 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_14[0][0]']                    
                                                                                                      
     activation_300 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     2_14[0][0]']                     
                                                                                                      
     activation_302 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_14[0][0]']                    
                                                                                                      
     activation_304 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     5_14[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 11, 11, 672)  468384     ['activation_296[0][0]']         
     14 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 11, 11, 672)  457632     ['activation_298[0][0]']         
     _14 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 11, 11, 672)  468384     ['activation_300[0][0]']         
     14 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 11, 11, 672)  457632     ['activation_302[0][0]']         
     _14 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 11, 11, 672)  457632     ['activation_304[0][0]']         
     14 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left1_1
     t1_14 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right1_
     ht1_14 (BatchNormalization)                                     14[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left2_1
     t2_14 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right2_
     ht2_14 (BatchNormalization)                                     14[0][0]']                       
                                                                                                      
     normal_left3_14 (AveragePoolin  (None, 11, 11, 672)  0          ['normal_bn_1_14[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_14 (AveragePoolin  (None, 11, 11, 672)  0          ['adjust_bn_14[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_14 (AveragePooli  (None, 11, 11, 672)  0          ['adjust_bn_14[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left5_1
     t5_14 (BatchNormalization)                                      4[0][0]']                        
                                                                                                      
     normal_add_1_14 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     1_14[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_14[0][0]']                    
                                                                                                      
     normal_add_2_14 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     2_14[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_14[0][0]']                    
                                                                                                      
     normal_add_3_14 (Add)          (None, 11, 11, 672)  0           ['normal_left3_14[0][0]',        
                                                                      'adjust_bn_14[0][0]']           
                                                                                                      
     normal_add_4_14 (Add)          (None, 11, 11, 672)  0           ['normal_left4_14[0][0]',        
                                                                      'normal_right4_14[0][0]']       
                                                                                                      
     normal_add_5_14 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     5_14[0][0]',                     
                                                                      'normal_bn_1_14[0][0]']         
                                                                                                      
     normal_concat_14 (Concatenate)  (None, 11, 11, 4032  0          ['adjust_bn_14[0][0]',           
                                    )                                 'normal_add_1_14[0][0]',        
                                                                      'normal_add_2_14[0][0]',        
                                                                      'normal_add_3_14[0][0]',        
                                                                      'normal_add_4_14[0][0]',        
                                                                      'normal_add_5_14[0][0]']        
                                                                                                      
     activation_305 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_13[0][0]']       
                                    )                                                                 
                                                                                                      
     activation_306 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_14[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_15 (Con  (None, 11, 11, 672)  2709504    ['activation_305[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_15 (Conv2D)      (None, 11, 11, 672)  2709504     ['activation_306[0][0]']         
                                                                                                      
     adjust_bn_15 (BatchNormalizati  (None, 11, 11, 672)  2688       ['adjust_conv_projection_15[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_15 (BatchNormaliza  (None, 11, 11, 672)  2688       ['normal_conv_1_15[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_307 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_15[0][0]']         
                                                                                                      
     activation_309 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_15[0][0]']           
                                                                                                      
     activation_311 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_15[0][0]']           
                                                                                                      
     activation_313 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_15[0][0]']           
                                                                                                      
     activation_315 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_15[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 11, 11, 672)  468384     ['activation_307[0][0]']         
     15 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 11, 11, 672)  457632     ['activation_309[0][0]']         
     _15 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 11, 11, 672)  468384     ['activation_311[0][0]']         
     15 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 11, 11, 672)  457632     ['activation_313[0][0]']         
     _15 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 11, 11, 672)  457632     ['activation_315[0][0]']         
     15 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left1_1
     t1_15 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right1_
     ht1_15 (BatchNormalization)                                     15[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left2_1
     t2_15 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right2_
     ht2_15 (BatchNormalization)                                     15[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left5_1
     t5_15 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     activation_308 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     1_15[0][0]']                     
                                                                                                      
     activation_310 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_15[0][0]']                    
                                                                                                      
     activation_312 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     2_15[0][0]']                     
                                                                                                      
     activation_314 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_15[0][0]']                    
                                                                                                      
     activation_316 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     5_15[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 11, 11, 672)  468384     ['activation_308[0][0]']         
     15 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 11, 11, 672)  457632     ['activation_310[0][0]']         
     _15 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 11, 11, 672)  468384     ['activation_312[0][0]']         
     15 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 11, 11, 672)  457632     ['activation_314[0][0]']         
     _15 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 11, 11, 672)  457632     ['activation_316[0][0]']         
     15 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left1_1
     t1_15 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right1_
     ht1_15 (BatchNormalization)                                     15[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left2_1
     t2_15 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right2_
     ht2_15 (BatchNormalization)                                     15[0][0]']                       
                                                                                                      
     normal_left3_15 (AveragePoolin  (None, 11, 11, 672)  0          ['normal_bn_1_15[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_15 (AveragePoolin  (None, 11, 11, 672)  0          ['adjust_bn_15[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_15 (AveragePooli  (None, 11, 11, 672)  0          ['adjust_bn_15[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left5_1
     t5_15 (BatchNormalization)                                      5[0][0]']                        
                                                                                                      
     normal_add_1_15 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     1_15[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_15[0][0]']                    
                                                                                                      
     normal_add_2_15 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     2_15[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_15[0][0]']                    
                                                                                                      
     normal_add_3_15 (Add)          (None, 11, 11, 672)  0           ['normal_left3_15[0][0]',        
                                                                      'adjust_bn_15[0][0]']           
                                                                                                      
     normal_add_4_15 (Add)          (None, 11, 11, 672)  0           ['normal_left4_15[0][0]',        
                                                                      'normal_right4_15[0][0]']       
                                                                                                      
     normal_add_5_15 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     5_15[0][0]',                     
                                                                      'normal_bn_1_15[0][0]']         
                                                                                                      
     normal_concat_15 (Concatenate)  (None, 11, 11, 4032  0          ['adjust_bn_15[0][0]',           
                                    )                                 'normal_add_1_15[0][0]',        
                                                                      'normal_add_2_15[0][0]',        
                                                                      'normal_add_3_15[0][0]',        
                                                                      'normal_add_4_15[0][0]',        
                                                                      'normal_add_5_15[0][0]']        
                                                                                                      
     activation_317 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_14[0][0]']       
                                    )                                                                 
                                                                                                      
     activation_318 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_15[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_16 (Con  (None, 11, 11, 672)  2709504    ['activation_317[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_16 (Conv2D)      (None, 11, 11, 672)  2709504     ['activation_318[0][0]']         
                                                                                                      
     adjust_bn_16 (BatchNormalizati  (None, 11, 11, 672)  2688       ['adjust_conv_projection_16[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_16 (BatchNormaliza  (None, 11, 11, 672)  2688       ['normal_conv_1_16[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_319 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_16[0][0]']         
                                                                                                      
     activation_321 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_16[0][0]']           
                                                                                                      
     activation_323 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_16[0][0]']           
                                                                                                      
     activation_325 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_16[0][0]']           
                                                                                                      
     activation_327 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_16[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 11, 11, 672)  468384     ['activation_319[0][0]']         
     16 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 11, 11, 672)  457632     ['activation_321[0][0]']         
     _16 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 11, 11, 672)  468384     ['activation_323[0][0]']         
     16 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 11, 11, 672)  457632     ['activation_325[0][0]']         
     _16 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 11, 11, 672)  457632     ['activation_327[0][0]']         
     16 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left1_1
     t1_16 (BatchNormalization)                                      6[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right1_
     ht1_16 (BatchNormalization)                                     16[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left2_1
     t2_16 (BatchNormalization)                                      6[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right2_
     ht2_16 (BatchNormalization)                                     16[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left5_1
     t5_16 (BatchNormalization)                                      6[0][0]']                        
                                                                                                      
     activation_320 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     1_16[0][0]']                     
                                                                                                      
     activation_322 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_16[0][0]']                    
                                                                                                      
     activation_324 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     2_16[0][0]']                     
                                                                                                      
     activation_326 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_16[0][0]']                    
                                                                                                      
     activation_328 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     5_16[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 11, 11, 672)  468384     ['activation_320[0][0]']         
     16 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 11, 11, 672)  457632     ['activation_322[0][0]']         
     _16 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 11, 11, 672)  468384     ['activation_324[0][0]']         
     16 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 11, 11, 672)  457632     ['activation_326[0][0]']         
     _16 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 11, 11, 672)  457632     ['activation_328[0][0]']         
     16 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left1_1
     t1_16 (BatchNormalization)                                      6[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right1_
     ht1_16 (BatchNormalization)                                     16[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left2_1
     t2_16 (BatchNormalization)                                      6[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right2_
     ht2_16 (BatchNormalization)                                     16[0][0]']                       
                                                                                                      
     normal_left3_16 (AveragePoolin  (None, 11, 11, 672)  0          ['normal_bn_1_16[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_16 (AveragePoolin  (None, 11, 11, 672)  0          ['adjust_bn_16[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_16 (AveragePooli  (None, 11, 11, 672)  0          ['adjust_bn_16[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left5_1
     t5_16 (BatchNormalization)                                      6[0][0]']                        
                                                                                                      
     normal_add_1_16 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     1_16[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_16[0][0]']                    
                                                                                                      
     normal_add_2_16 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     2_16[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_16[0][0]']                    
                                                                                                      
     normal_add_3_16 (Add)          (None, 11, 11, 672)  0           ['normal_left3_16[0][0]',        
                                                                      'adjust_bn_16[0][0]']           
                                                                                                      
     normal_add_4_16 (Add)          (None, 11, 11, 672)  0           ['normal_left4_16[0][0]',        
                                                                      'normal_right4_16[0][0]']       
                                                                                                      
     normal_add_5_16 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     5_16[0][0]',                     
                                                                      'normal_bn_1_16[0][0]']         
                                                                                                      
     normal_concat_16 (Concatenate)  (None, 11, 11, 4032  0          ['adjust_bn_16[0][0]',           
                                    )                                 'normal_add_1_16[0][0]',        
                                                                      'normal_add_2_16[0][0]',        
                                                                      'normal_add_3_16[0][0]',        
                                                                      'normal_add_4_16[0][0]',        
                                                                      'normal_add_5_16[0][0]']        
                                                                                                      
     activation_329 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_15[0][0]']       
                                    )                                                                 
                                                                                                      
     activation_330 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_16[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_17 (Con  (None, 11, 11, 672)  2709504    ['activation_329[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_17 (Conv2D)      (None, 11, 11, 672)  2709504     ['activation_330[0][0]']         
                                                                                                      
     adjust_bn_17 (BatchNormalizati  (None, 11, 11, 672)  2688       ['adjust_conv_projection_17[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_17 (BatchNormaliza  (None, 11, 11, 672)  2688       ['normal_conv_1_17[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_331 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_17[0][0]']         
                                                                                                      
     activation_333 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_17[0][0]']           
                                                                                                      
     activation_335 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_17[0][0]']           
                                                                                                      
     activation_337 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_17[0][0]']           
                                                                                                      
     activation_339 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_17[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 11, 11, 672)  468384     ['activation_331[0][0]']         
     17 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 11, 11, 672)  457632     ['activation_333[0][0]']         
     _17 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 11, 11, 672)  468384     ['activation_335[0][0]']         
     17 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 11, 11, 672)  457632     ['activation_337[0][0]']         
     _17 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 11, 11, 672)  457632     ['activation_339[0][0]']         
     17 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left1_1
     t1_17 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right1_
     ht1_17 (BatchNormalization)                                     17[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left2_1
     t2_17 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right2_
     ht2_17 (BatchNormalization)                                     17[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left5_1
     t5_17 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     activation_332 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     1_17[0][0]']                     
                                                                                                      
     activation_334 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_17[0][0]']                    
                                                                                                      
     activation_336 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     2_17[0][0]']                     
                                                                                                      
     activation_338 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_17[0][0]']                    
                                                                                                      
     activation_340 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     5_17[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 11, 11, 672)  468384     ['activation_332[0][0]']         
     17 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 11, 11, 672)  457632     ['activation_334[0][0]']         
     _17 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 11, 11, 672)  468384     ['activation_336[0][0]']         
     17 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 11, 11, 672)  457632     ['activation_338[0][0]']         
     _17 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 11, 11, 672)  457632     ['activation_340[0][0]']         
     17 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left1_1
     t1_17 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right1_
     ht1_17 (BatchNormalization)                                     17[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left2_1
     t2_17 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right2_
     ht2_17 (BatchNormalization)                                     17[0][0]']                       
                                                                                                      
     normal_left3_17 (AveragePoolin  (None, 11, 11, 672)  0          ['normal_bn_1_17[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_17 (AveragePoolin  (None, 11, 11, 672)  0          ['adjust_bn_17[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_17 (AveragePooli  (None, 11, 11, 672)  0          ['adjust_bn_17[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left5_1
     t5_17 (BatchNormalization)                                      7[0][0]']                        
                                                                                                      
     normal_add_1_17 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     1_17[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_17[0][0]']                    
                                                                                                      
     normal_add_2_17 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     2_17[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_17[0][0]']                    
                                                                                                      
     normal_add_3_17 (Add)          (None, 11, 11, 672)  0           ['normal_left3_17[0][0]',        
                                                                      'adjust_bn_17[0][0]']           
                                                                                                      
     normal_add_4_17 (Add)          (None, 11, 11, 672)  0           ['normal_left4_17[0][0]',        
                                                                      'normal_right4_17[0][0]']       
                                                                                                      
     normal_add_5_17 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     5_17[0][0]',                     
                                                                      'normal_bn_1_17[0][0]']         
                                                                                                      
     normal_concat_17 (Concatenate)  (None, 11, 11, 4032  0          ['adjust_bn_17[0][0]',           
                                    )                                 'normal_add_1_17[0][0]',        
                                                                      'normal_add_2_17[0][0]',        
                                                                      'normal_add_3_17[0][0]',        
                                                                      'normal_add_4_17[0][0]',        
                                                                      'normal_add_5_17[0][0]']        
                                                                                                      
     activation_341 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_16[0][0]']       
                                    )                                                                 
                                                                                                      
     activation_342 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_17[0][0]']       
                                    )                                                                 
                                                                                                      
     adjust_conv_projection_18 (Con  (None, 11, 11, 672)  2709504    ['activation_341[0][0]']         
     v2D)                                                                                             
                                                                                                      
     normal_conv_1_18 (Conv2D)      (None, 11, 11, 672)  2709504     ['activation_342[0][0]']         
                                                                                                      
     adjust_bn_18 (BatchNormalizati  (None, 11, 11, 672)  2688       ['adjust_conv_projection_18[0][0]
     on)                                                             ']                               
                                                                                                      
     normal_bn_1_18 (BatchNormaliza  (None, 11, 11, 672)  2688       ['normal_conv_1_18[0][0]']       
     tion)                                                                                            
                                                                                                      
     activation_343 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_18[0][0]']         
                                                                                                      
     activation_345 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_18[0][0]']           
                                                                                                      
     activation_347 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_18[0][0]']           
                                                                                                      
     activation_349 (Activation)    (None, 11, 11, 672)  0           ['adjust_bn_18[0][0]']           
                                                                                                      
     activation_351 (Activation)    (None, 11, 11, 672)  0           ['normal_bn_1_18[0][0]']         
                                                                                                      
     separable_conv_1_normal_left1_  (None, 11, 11, 672)  468384     ['activation_343[0][0]']         
     18 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right1  (None, 11, 11, 672)  457632     ['activation_345[0][0]']         
     _18 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left2_  (None, 11, 11, 672)  468384     ['activation_347[0][0]']         
     18 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_normal_right2  (None, 11, 11, 672)  457632     ['activation_349[0][0]']         
     _18 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_1_normal_left5_  (None, 11, 11, 672)  457632     ['activation_351[0][0]']         
     18 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left1_1
     t1_18 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right1_
     ht1_18 (BatchNormalization)                                     18[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left2_1
     t2_18 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_1_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_right2_
     ht2_18 (BatchNormalization)                                     18[0][0]']                       
                                                                                                      
     separable_conv_1_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_1_normal_left5_1
     t5_18 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     activation_344 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     1_18[0][0]']                     
                                                                                                      
     activation_346 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t1_18[0][0]']                    
                                                                                                      
     activation_348 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     2_18[0][0]']                     
                                                                                                      
     activation_350 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_righ
                                                                     t2_18[0][0]']                    
                                                                                                      
     activation_352 (Activation)    (None, 11, 11, 672)  0           ['separable_conv_1_bn_normal_left
                                                                     5_18[0][0]']                     
                                                                                                      
     separable_conv_2_normal_left1_  (None, 11, 11, 672)  468384     ['activation_344[0][0]']         
     18 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right1  (None, 11, 11, 672)  457632     ['activation_346[0][0]']         
     _18 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left2_  (None, 11, 11, 672)  468384     ['activation_348[0][0]']         
     18 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_normal_right2  (None, 11, 11, 672)  457632     ['activation_350[0][0]']         
     _18 (SeparableConv2D)                                                                            
                                                                                                      
     separable_conv_2_normal_left5_  (None, 11, 11, 672)  457632     ['activation_352[0][0]']         
     18 (SeparableConv2D)                                                                             
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left1_1
     t1_18 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right1_
     ht1_18 (BatchNormalization)                                     18[0][0]']                       
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left2_1
     t2_18 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     separable_conv_2_bn_normal_rig  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_right2_
     ht2_18 (BatchNormalization)                                     18[0][0]']                       
                                                                                                      
     normal_left3_18 (AveragePoolin  (None, 11, 11, 672)  0          ['normal_bn_1_18[0][0]']         
     g2D)                                                                                             
                                                                                                      
     normal_left4_18 (AveragePoolin  (None, 11, 11, 672)  0          ['adjust_bn_18[0][0]']           
     g2D)                                                                                             
                                                                                                      
     normal_right4_18 (AveragePooli  (None, 11, 11, 672)  0          ['adjust_bn_18[0][0]']           
     ng2D)                                                                                            
                                                                                                      
     separable_conv_2_bn_normal_lef  (None, 11, 11, 672)  2688       ['separable_conv_2_normal_left5_1
     t5_18 (BatchNormalization)                                      8[0][0]']                        
                                                                                                      
     normal_add_1_18 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     1_18[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t1_18[0][0]']                    
                                                                                                      
     normal_add_2_18 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     2_18[0][0]',                     
                                                                      'separable_conv_2_bn_normal_righ
                                                                     t2_18[0][0]']                    
                                                                                                      
     normal_add_3_18 (Add)          (None, 11, 11, 672)  0           ['normal_left3_18[0][0]',        
                                                                      'adjust_bn_18[0][0]']           
                                                                                                      
     normal_add_4_18 (Add)          (None, 11, 11, 672)  0           ['normal_left4_18[0][0]',        
                                                                      'normal_right4_18[0][0]']       
                                                                                                      
     normal_add_5_18 (Add)          (None, 11, 11, 672)  0           ['separable_conv_2_bn_normal_left
                                                                     5_18[0][0]',                     
                                                                      'normal_bn_1_18[0][0]']         
                                                                                                      
     normal_concat_18 (Concatenate)  (None, 11, 11, 4032  0          ['adjust_bn_18[0][0]',           
                                    )                                 'normal_add_1_18[0][0]',        
                                                                      'normal_add_2_18[0][0]',        
                                                                      'normal_add_3_18[0][0]',        
                                                                      'normal_add_4_18[0][0]',        
                                                                      'normal_add_5_18[0][0]']        
                                                                                                      
     activation_353 (Activation)    (None, 11, 11, 4032  0           ['normal_concat_18[0][0]']       
                                    )                                                                 
                                                                                                      
     global_average_pooling2d_1 (Gl  (None, 4032)        0           ['activation_353[0][0]']         
     obalAveragePooling2D)                                                                            
                                                                                                      
     predictions (Dense)            (None, 1000)         4033000     ['global_average_pooling2d_1[0][0
                                                                     ]']                              
                                                                                                      
    ==================================================================================================
    Total params: 88,949,818
    Trainable params: 88,753,150
    Non-trainable params: 196,668
    __________________________________________________________________________________________________
    


```python
!wget -O notebook.jpg https://cdn.pixabay.com/photo/2016/07/11/03/35/macbook-1508998_1280.jpg
```

    --2023-03-09 15:07:53--  https://cdn.pixabay.com/photo/2016/07/11/03/35/macbook-1508998_1280.jpg
    Resolving cdn.pixabay.com (cdn.pixabay.com)... 104.18.14.16, 104.18.15.16, 2606:4700::6812:f10, ...
    Connecting to cdn.pixabay.com (cdn.pixabay.com)|104.18.14.16|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 270631 (264K) [binary/octet-stream]
    Saving to: ‘notebook.jpg’
    
    notebook.jpg        100%[===================>] 264.29K  --.-KB/s    in 0.01s   
    
    2023-03-09 15:07:54 (20.4 MB/s) - ‘notebook.jpg’ saved [270631/270631]
    
    


```python
# 다운로드한 notebook.jpg를 terget_size로 줄여줌
# notebook input은 [(None, 331, 331, 3  0)] 크기를 가짐
img = image.load_img('notebook.jpg', target_size=(331, 331))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = nasnet.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    1/1 [==============================] - 6s 6s/step
    [[('n03832673', 'notebook', 0.79294485), ('n03642806', 'laptop', 0.048423648), ('n04264628', 'space_bar', 0.034534946), ('n03085013', 'computer_keyboard', 0.020163987), ('n03777754', 'modem', 0.011375744)]]
    




## EfficientNet

* EfficientNetB0인 작은 모델에서 주어진 Task에 최적화된 구조로 수정해나가는 형태
* 복잡한 Task에 맞춰 모델의 Capacity를 늘리기 위해 Wide Scaling, Deep Scaling, 그리고 Resolution Scaling을 사용
* EfficientNet은 Wide, Deep, Resolution을 함께 고려하는 Compound Scaling을 사용


```python
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input, decode_predictions
```


```python
eff = EfficientNetB1(include_top=True, weights='imagenet',
                      input_tensor=None, input_shape=None,
                      pooling=None, classes=1000)
eff.summary()
```

    Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb1.h5
    32148312/32148312 [==============================] - 3s 0us/step
    Model: "efficientnetb1"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_8 (InputLayer)           [(None, 240, 240, 3  0           []                               
                                    )]                                                                
                                                                                                      
     rescaling (Rescaling)          (None, 240, 240, 3)  0           ['input_8[0][0]']                
                                                                                                      
     normalization (Normalization)  (None, 240, 240, 3)  7           ['rescaling[0][0]']              
                                                                                                      
     rescaling_1 (Rescaling)        (None, 240, 240, 3)  0           ['normalization[0][0]']          
                                                                                                      
     stem_conv_pad (ZeroPadding2D)  (None, 241, 241, 3)  0           ['rescaling_1[0][0]']            
                                                                                                      
     stem_conv (Conv2D)             (None, 120, 120, 32  864         ['stem_conv_pad[0][0]']          
                                    )                                                                 
                                                                                                      
     stem_bn (BatchNormalization)   (None, 120, 120, 32  128         ['stem_conv[0][0]']              
                                    )                                                                 
                                                                                                      
     stem_activation (Activation)   (None, 120, 120, 32  0           ['stem_bn[0][0]']                
                                    )                                                                 
                                                                                                      
     block1a_dwconv (DepthwiseConv2  (None, 120, 120, 32  288        ['stem_activation[0][0]']        
     D)                             )                                                                 
                                                                                                      
     block1a_bn (BatchNormalization  (None, 120, 120, 32  128        ['block1a_dwconv[0][0]']         
     )                              )                                                                 
                                                                                                      
     block1a_activation (Activation  (None, 120, 120, 32  0          ['block1a_bn[0][0]']             
     )                              )                                                                 
                                                                                                      
     block1a_se_squeeze (GlobalAver  (None, 32)          0           ['block1a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block1a_se_reshape (Reshape)   (None, 1, 1, 32)     0           ['block1a_se_squeeze[0][0]']     
                                                                                                      
     block1a_se_reduce (Conv2D)     (None, 1, 1, 8)      264         ['block1a_se_reshape[0][0]']     
                                                                                                      
     block1a_se_expand (Conv2D)     (None, 1, 1, 32)     288         ['block1a_se_reduce[0][0]']      
                                                                                                      
     block1a_se_excite (Multiply)   (None, 120, 120, 32  0           ['block1a_activation[0][0]',     
                                    )                                 'block1a_se_expand[0][0]']      
                                                                                                      
     block1a_project_conv (Conv2D)  (None, 120, 120, 16  512         ['block1a_se_excite[0][0]']      
                                    )                                                                 
                                                                                                      
     block1a_project_bn (BatchNorma  (None, 120, 120, 16  64         ['block1a_project_conv[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     block1b_dwconv (DepthwiseConv2  (None, 120, 120, 16  144        ['block1a_project_bn[0][0]']     
     D)                             )                                                                 
                                                                                                      
     block1b_bn (BatchNormalization  (None, 120, 120, 16  64         ['block1b_dwconv[0][0]']         
     )                              )                                                                 
                                                                                                      
     block1b_activation (Activation  (None, 120, 120, 16  0          ['block1b_bn[0][0]']             
     )                              )                                                                 
                                                                                                      
     block1b_se_squeeze (GlobalAver  (None, 16)          0           ['block1b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block1b_se_reshape (Reshape)   (None, 1, 1, 16)     0           ['block1b_se_squeeze[0][0]']     
                                                                                                      
     block1b_se_reduce (Conv2D)     (None, 1, 1, 4)      68          ['block1b_se_reshape[0][0]']     
                                                                                                      
     block1b_se_expand (Conv2D)     (None, 1, 1, 16)     80          ['block1b_se_reduce[0][0]']      
                                                                                                      
     block1b_se_excite (Multiply)   (None, 120, 120, 16  0           ['block1b_activation[0][0]',     
                                    )                                 'block1b_se_expand[0][0]']      
                                                                                                      
     block1b_project_conv (Conv2D)  (None, 120, 120, 16  256         ['block1b_se_excite[0][0]']      
                                    )                                                                 
                                                                                                      
     block1b_project_bn (BatchNorma  (None, 120, 120, 16  64         ['block1b_project_conv[0][0]']   
     lization)                      )                                                                 
                                                                                                      
     block1b_drop (Dropout)         (None, 120, 120, 16  0           ['block1b_project_bn[0][0]']     
                                    )                                                                 
                                                                                                      
     block1b_add (Add)              (None, 120, 120, 16  0           ['block1b_drop[0][0]',           
                                    )                                 'block1a_project_bn[0][0]']     
                                                                                                      
     block2a_expand_conv (Conv2D)   (None, 120, 120, 96  1536        ['block1b_add[0][0]']            
                                    )                                                                 
                                                                                                      
     block2a_expand_bn (BatchNormal  (None, 120, 120, 96  384        ['block2a_expand_conv[0][0]']    
     ization)                       )                                                                 
                                                                                                      
     block2a_expand_activation (Act  (None, 120, 120, 96  0          ['block2a_expand_bn[0][0]']      
     ivation)                       )                                                                 
                                                                                                      
     block2a_dwconv_pad (ZeroPaddin  (None, 121, 121, 96  0          ['block2a_expand_activation[0][0]
     g2D)                           )                                ']                               
                                                                                                      
     block2a_dwconv (DepthwiseConv2  (None, 60, 60, 96)  864         ['block2a_dwconv_pad[0][0]']     
     D)                                                                                               
                                                                                                      
     block2a_bn (BatchNormalization  (None, 60, 60, 96)  384         ['block2a_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block2a_activation (Activation  (None, 60, 60, 96)  0           ['block2a_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block2a_se_squeeze (GlobalAver  (None, 96)          0           ['block2a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block2a_se_reshape (Reshape)   (None, 1, 1, 96)     0           ['block2a_se_squeeze[0][0]']     
                                                                                                      
     block2a_se_reduce (Conv2D)     (None, 1, 1, 4)      388         ['block2a_se_reshape[0][0]']     
                                                                                                      
     block2a_se_expand (Conv2D)     (None, 1, 1, 96)     480         ['block2a_se_reduce[0][0]']      
                                                                                                      
     block2a_se_excite (Multiply)   (None, 60, 60, 96)   0           ['block2a_activation[0][0]',     
                                                                      'block2a_se_expand[0][0]']      
                                                                                                      
     block2a_project_conv (Conv2D)  (None, 60, 60, 24)   2304        ['block2a_se_excite[0][0]']      
                                                                                                      
     block2a_project_bn (BatchNorma  (None, 60, 60, 24)  96          ['block2a_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block2b_expand_conv (Conv2D)   (None, 60, 60, 144)  3456        ['block2a_project_bn[0][0]']     
                                                                                                      
     block2b_expand_bn (BatchNormal  (None, 60, 60, 144)  576        ['block2b_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block2b_expand_activation (Act  (None, 60, 60, 144)  0          ['block2b_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block2b_dwconv (DepthwiseConv2  (None, 60, 60, 144)  1296       ['block2b_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block2b_bn (BatchNormalization  (None, 60, 60, 144)  576        ['block2b_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block2b_activation (Activation  (None, 60, 60, 144)  0          ['block2b_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block2b_se_squeeze (GlobalAver  (None, 144)         0           ['block2b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block2b_se_reshape (Reshape)   (None, 1, 1, 144)    0           ['block2b_se_squeeze[0][0]']     
                                                                                                      
     block2b_se_reduce (Conv2D)     (None, 1, 1, 6)      870         ['block2b_se_reshape[0][0]']     
                                                                                                      
     block2b_se_expand (Conv2D)     (None, 1, 1, 144)    1008        ['block2b_se_reduce[0][0]']      
                                                                                                      
     block2b_se_excite (Multiply)   (None, 60, 60, 144)  0           ['block2b_activation[0][0]',     
                                                                      'block2b_se_expand[0][0]']      
                                                                                                      
     block2b_project_conv (Conv2D)  (None, 60, 60, 24)   3456        ['block2b_se_excite[0][0]']      
                                                                                                      
     block2b_project_bn (BatchNorma  (None, 60, 60, 24)  96          ['block2b_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block2b_drop (Dropout)         (None, 60, 60, 24)   0           ['block2b_project_bn[0][0]']     
                                                                                                      
     block2b_add (Add)              (None, 60, 60, 24)   0           ['block2b_drop[0][0]',           
                                                                      'block2a_project_bn[0][0]']     
                                                                                                      
     block2c_expand_conv (Conv2D)   (None, 60, 60, 144)  3456        ['block2b_add[0][0]']            
                                                                                                      
     block2c_expand_bn (BatchNormal  (None, 60, 60, 144)  576        ['block2c_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block2c_expand_activation (Act  (None, 60, 60, 144)  0          ['block2c_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block2c_dwconv (DepthwiseConv2  (None, 60, 60, 144)  1296       ['block2c_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block2c_bn (BatchNormalization  (None, 60, 60, 144)  576        ['block2c_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block2c_activation (Activation  (None, 60, 60, 144)  0          ['block2c_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block2c_se_squeeze (GlobalAver  (None, 144)         0           ['block2c_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block2c_se_reshape (Reshape)   (None, 1, 1, 144)    0           ['block2c_se_squeeze[0][0]']     
                                                                                                      
     block2c_se_reduce (Conv2D)     (None, 1, 1, 6)      870         ['block2c_se_reshape[0][0]']     
                                                                                                      
     block2c_se_expand (Conv2D)     (None, 1, 1, 144)    1008        ['block2c_se_reduce[0][0]']      
                                                                                                      
     block2c_se_excite (Multiply)   (None, 60, 60, 144)  0           ['block2c_activation[0][0]',     
                                                                      'block2c_se_expand[0][0]']      
                                                                                                      
     block2c_project_conv (Conv2D)  (None, 60, 60, 24)   3456        ['block2c_se_excite[0][0]']      
                                                                                                      
     block2c_project_bn (BatchNorma  (None, 60, 60, 24)  96          ['block2c_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block2c_drop (Dropout)         (None, 60, 60, 24)   0           ['block2c_project_bn[0][0]']     
                                                                                                      
     block2c_add (Add)              (None, 60, 60, 24)   0           ['block2c_drop[0][0]',           
                                                                      'block2b_add[0][0]']            
                                                                                                      
     block3a_expand_conv (Conv2D)   (None, 60, 60, 144)  3456        ['block2c_add[0][0]']            
                                                                                                      
     block3a_expand_bn (BatchNormal  (None, 60, 60, 144)  576        ['block3a_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block3a_expand_activation (Act  (None, 60, 60, 144)  0          ['block3a_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block3a_dwconv_pad (ZeroPaddin  (None, 63, 63, 144)  0          ['block3a_expand_activation[0][0]
     g2D)                                                            ']                               
                                                                                                      
     block3a_dwconv (DepthwiseConv2  (None, 30, 30, 144)  3600       ['block3a_dwconv_pad[0][0]']     
     D)                                                                                               
                                                                                                      
     block3a_bn (BatchNormalization  (None, 30, 30, 144)  576        ['block3a_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block3a_activation (Activation  (None, 30, 30, 144)  0          ['block3a_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block3a_se_squeeze (GlobalAver  (None, 144)         0           ['block3a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block3a_se_reshape (Reshape)   (None, 1, 1, 144)    0           ['block3a_se_squeeze[0][0]']     
                                                                                                      
     block3a_se_reduce (Conv2D)     (None, 1, 1, 6)      870         ['block3a_se_reshape[0][0]']     
                                                                                                      
     block3a_se_expand (Conv2D)     (None, 1, 1, 144)    1008        ['block3a_se_reduce[0][0]']      
                                                                                                      
     block3a_se_excite (Multiply)   (None, 30, 30, 144)  0           ['block3a_activation[0][0]',     
                                                                      'block3a_se_expand[0][0]']      
                                                                                                      
     block3a_project_conv (Conv2D)  (None, 30, 30, 40)   5760        ['block3a_se_excite[0][0]']      
                                                                                                      
     block3a_project_bn (BatchNorma  (None, 30, 30, 40)  160         ['block3a_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block3b_expand_conv (Conv2D)   (None, 30, 30, 240)  9600        ['block3a_project_bn[0][0]']     
                                                                                                      
     block3b_expand_bn (BatchNormal  (None, 30, 30, 240)  960        ['block3b_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block3b_expand_activation (Act  (None, 30, 30, 240)  0          ['block3b_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block3b_dwconv (DepthwiseConv2  (None, 30, 30, 240)  6000       ['block3b_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block3b_bn (BatchNormalization  (None, 30, 30, 240)  960        ['block3b_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block3b_activation (Activation  (None, 30, 30, 240)  0          ['block3b_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block3b_se_squeeze (GlobalAver  (None, 240)         0           ['block3b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block3b_se_reshape (Reshape)   (None, 1, 1, 240)    0           ['block3b_se_squeeze[0][0]']     
                                                                                                      
     block3b_se_reduce (Conv2D)     (None, 1, 1, 10)     2410        ['block3b_se_reshape[0][0]']     
                                                                                                      
     block3b_se_expand (Conv2D)     (None, 1, 1, 240)    2640        ['block3b_se_reduce[0][0]']      
                                                                                                      
     block3b_se_excite (Multiply)   (None, 30, 30, 240)  0           ['block3b_activation[0][0]',     
                                                                      'block3b_se_expand[0][0]']      
                                                                                                      
     block3b_project_conv (Conv2D)  (None, 30, 30, 40)   9600        ['block3b_se_excite[0][0]']      
                                                                                                      
     block3b_project_bn (BatchNorma  (None, 30, 30, 40)  160         ['block3b_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block3b_drop (Dropout)         (None, 30, 30, 40)   0           ['block3b_project_bn[0][0]']     
                                                                                                      
     block3b_add (Add)              (None, 30, 30, 40)   0           ['block3b_drop[0][0]',           
                                                                      'block3a_project_bn[0][0]']     
                                                                                                      
     block3c_expand_conv (Conv2D)   (None, 30, 30, 240)  9600        ['block3b_add[0][0]']            
                                                                                                      
     block3c_expand_bn (BatchNormal  (None, 30, 30, 240)  960        ['block3c_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block3c_expand_activation (Act  (None, 30, 30, 240)  0          ['block3c_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block3c_dwconv (DepthwiseConv2  (None, 30, 30, 240)  6000       ['block3c_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block3c_bn (BatchNormalization  (None, 30, 30, 240)  960        ['block3c_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block3c_activation (Activation  (None, 30, 30, 240)  0          ['block3c_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block3c_se_squeeze (GlobalAver  (None, 240)         0           ['block3c_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block3c_se_reshape (Reshape)   (None, 1, 1, 240)    0           ['block3c_se_squeeze[0][0]']     
                                                                                                      
     block3c_se_reduce (Conv2D)     (None, 1, 1, 10)     2410        ['block3c_se_reshape[0][0]']     
                                                                                                      
     block3c_se_expand (Conv2D)     (None, 1, 1, 240)    2640        ['block3c_se_reduce[0][0]']      
                                                                                                      
     block3c_se_excite (Multiply)   (None, 30, 30, 240)  0           ['block3c_activation[0][0]',     
                                                                      'block3c_se_expand[0][0]']      
                                                                                                      
     block3c_project_conv (Conv2D)  (None, 30, 30, 40)   9600        ['block3c_se_excite[0][0]']      
                                                                                                      
     block3c_project_bn (BatchNorma  (None, 30, 30, 40)  160         ['block3c_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block3c_drop (Dropout)         (None, 30, 30, 40)   0           ['block3c_project_bn[0][0]']     
                                                                                                      
     block3c_add (Add)              (None, 30, 30, 40)   0           ['block3c_drop[0][0]',           
                                                                      'block3b_add[0][0]']            
                                                                                                      
     block4a_expand_conv (Conv2D)   (None, 30, 30, 240)  9600        ['block3c_add[0][0]']            
                                                                                                      
     block4a_expand_bn (BatchNormal  (None, 30, 30, 240)  960        ['block4a_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block4a_expand_activation (Act  (None, 30, 30, 240)  0          ['block4a_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block4a_dwconv_pad (ZeroPaddin  (None, 31, 31, 240)  0          ['block4a_expand_activation[0][0]
     g2D)                                                            ']                               
                                                                                                      
     block4a_dwconv (DepthwiseConv2  (None, 15, 15, 240)  2160       ['block4a_dwconv_pad[0][0]']     
     D)                                                                                               
                                                                                                      
     block4a_bn (BatchNormalization  (None, 15, 15, 240)  960        ['block4a_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block4a_activation (Activation  (None, 15, 15, 240)  0          ['block4a_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block4a_se_squeeze (GlobalAver  (None, 240)         0           ['block4a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block4a_se_reshape (Reshape)   (None, 1, 1, 240)    0           ['block4a_se_squeeze[0][0]']     
                                                                                                      
     block4a_se_reduce (Conv2D)     (None, 1, 1, 10)     2410        ['block4a_se_reshape[0][0]']     
                                                                                                      
     block4a_se_expand (Conv2D)     (None, 1, 1, 240)    2640        ['block4a_se_reduce[0][0]']      
                                                                                                      
     block4a_se_excite (Multiply)   (None, 15, 15, 240)  0           ['block4a_activation[0][0]',     
                                                                      'block4a_se_expand[0][0]']      
                                                                                                      
     block4a_project_conv (Conv2D)  (None, 15, 15, 80)   19200       ['block4a_se_excite[0][0]']      
                                                                                                      
     block4a_project_bn (BatchNorma  (None, 15, 15, 80)  320         ['block4a_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block4b_expand_conv (Conv2D)   (None, 15, 15, 480)  38400       ['block4a_project_bn[0][0]']     
                                                                                                      
     block4b_expand_bn (BatchNormal  (None, 15, 15, 480)  1920       ['block4b_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block4b_expand_activation (Act  (None, 15, 15, 480)  0          ['block4b_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block4b_dwconv (DepthwiseConv2  (None, 15, 15, 480)  4320       ['block4b_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block4b_bn (BatchNormalization  (None, 15, 15, 480)  1920       ['block4b_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block4b_activation (Activation  (None, 15, 15, 480)  0          ['block4b_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block4b_se_squeeze (GlobalAver  (None, 480)         0           ['block4b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block4b_se_reshape (Reshape)   (None, 1, 1, 480)    0           ['block4b_se_squeeze[0][0]']     
                                                                                                      
     block4b_se_reduce (Conv2D)     (None, 1, 1, 20)     9620        ['block4b_se_reshape[0][0]']     
                                                                                                      
     block4b_se_expand (Conv2D)     (None, 1, 1, 480)    10080       ['block4b_se_reduce[0][0]']      
                                                                                                      
     block4b_se_excite (Multiply)   (None, 15, 15, 480)  0           ['block4b_activation[0][0]',     
                                                                      'block4b_se_expand[0][0]']      
                                                                                                      
     block4b_project_conv (Conv2D)  (None, 15, 15, 80)   38400       ['block4b_se_excite[0][0]']      
                                                                                                      
     block4b_project_bn (BatchNorma  (None, 15, 15, 80)  320         ['block4b_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block4b_drop (Dropout)         (None, 15, 15, 80)   0           ['block4b_project_bn[0][0]']     
                                                                                                      
     block4b_add (Add)              (None, 15, 15, 80)   0           ['block4b_drop[0][0]',           
                                                                      'block4a_project_bn[0][0]']     
                                                                                                      
     block4c_expand_conv (Conv2D)   (None, 15, 15, 480)  38400       ['block4b_add[0][0]']            
                                                                                                      
     block4c_expand_bn (BatchNormal  (None, 15, 15, 480)  1920       ['block4c_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block4c_expand_activation (Act  (None, 15, 15, 480)  0          ['block4c_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block4c_dwconv (DepthwiseConv2  (None, 15, 15, 480)  4320       ['block4c_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block4c_bn (BatchNormalization  (None, 15, 15, 480)  1920       ['block4c_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block4c_activation (Activation  (None, 15, 15, 480)  0          ['block4c_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block4c_se_squeeze (GlobalAver  (None, 480)         0           ['block4c_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block4c_se_reshape (Reshape)   (None, 1, 1, 480)    0           ['block4c_se_squeeze[0][0]']     
                                                                                                      
     block4c_se_reduce (Conv2D)     (None, 1, 1, 20)     9620        ['block4c_se_reshape[0][0]']     
                                                                                                      
     block4c_se_expand (Conv2D)     (None, 1, 1, 480)    10080       ['block4c_se_reduce[0][0]']      
                                                                                                      
     block4c_se_excite (Multiply)   (None, 15, 15, 480)  0           ['block4c_activation[0][0]',     
                                                                      'block4c_se_expand[0][0]']      
                                                                                                      
     block4c_project_conv (Conv2D)  (None, 15, 15, 80)   38400       ['block4c_se_excite[0][0]']      
                                                                                                      
     block4c_project_bn (BatchNorma  (None, 15, 15, 80)  320         ['block4c_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block4c_drop (Dropout)         (None, 15, 15, 80)   0           ['block4c_project_bn[0][0]']     
                                                                                                      
     block4c_add (Add)              (None, 15, 15, 80)   0           ['block4c_drop[0][0]',           
                                                                      'block4b_add[0][0]']            
                                                                                                      
     block4d_expand_conv (Conv2D)   (None, 15, 15, 480)  38400       ['block4c_add[0][0]']            
                                                                                                      
     block4d_expand_bn (BatchNormal  (None, 15, 15, 480)  1920       ['block4d_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block4d_expand_activation (Act  (None, 15, 15, 480)  0          ['block4d_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block4d_dwconv (DepthwiseConv2  (None, 15, 15, 480)  4320       ['block4d_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block4d_bn (BatchNormalization  (None, 15, 15, 480)  1920       ['block4d_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block4d_activation (Activation  (None, 15, 15, 480)  0          ['block4d_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block4d_se_squeeze (GlobalAver  (None, 480)         0           ['block4d_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block4d_se_reshape (Reshape)   (None, 1, 1, 480)    0           ['block4d_se_squeeze[0][0]']     
                                                                                                      
     block4d_se_reduce (Conv2D)     (None, 1, 1, 20)     9620        ['block4d_se_reshape[0][0]']     
                                                                                                      
     block4d_se_expand (Conv2D)     (None, 1, 1, 480)    10080       ['block4d_se_reduce[0][0]']      
                                                                                                      
     block4d_se_excite (Multiply)   (None, 15, 15, 480)  0           ['block4d_activation[0][0]',     
                                                                      'block4d_se_expand[0][0]']      
                                                                                                      
     block4d_project_conv (Conv2D)  (None, 15, 15, 80)   38400       ['block4d_se_excite[0][0]']      
                                                                                                      
     block4d_project_bn (BatchNorma  (None, 15, 15, 80)  320         ['block4d_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block4d_drop (Dropout)         (None, 15, 15, 80)   0           ['block4d_project_bn[0][0]']     
                                                                                                      
     block4d_add (Add)              (None, 15, 15, 80)   0           ['block4d_drop[0][0]',           
                                                                      'block4c_add[0][0]']            
                                                                                                      
     block5a_expand_conv (Conv2D)   (None, 15, 15, 480)  38400       ['block4d_add[0][0]']            
                                                                                                      
     block5a_expand_bn (BatchNormal  (None, 15, 15, 480)  1920       ['block5a_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block5a_expand_activation (Act  (None, 15, 15, 480)  0          ['block5a_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block5a_dwconv (DepthwiseConv2  (None, 15, 15, 480)  12000      ['block5a_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block5a_bn (BatchNormalization  (None, 15, 15, 480)  1920       ['block5a_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block5a_activation (Activation  (None, 15, 15, 480)  0          ['block5a_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block5a_se_squeeze (GlobalAver  (None, 480)         0           ['block5a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block5a_se_reshape (Reshape)   (None, 1, 1, 480)    0           ['block5a_se_squeeze[0][0]']     
                                                                                                      
     block5a_se_reduce (Conv2D)     (None, 1, 1, 20)     9620        ['block5a_se_reshape[0][0]']     
                                                                                                      
     block5a_se_expand (Conv2D)     (None, 1, 1, 480)    10080       ['block5a_se_reduce[0][0]']      
                                                                                                      
     block5a_se_excite (Multiply)   (None, 15, 15, 480)  0           ['block5a_activation[0][0]',     
                                                                      'block5a_se_expand[0][0]']      
                                                                                                      
     block5a_project_conv (Conv2D)  (None, 15, 15, 112)  53760       ['block5a_se_excite[0][0]']      
                                                                                                      
     block5a_project_bn (BatchNorma  (None, 15, 15, 112)  448        ['block5a_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block5b_expand_conv (Conv2D)   (None, 15, 15, 672)  75264       ['block5a_project_bn[0][0]']     
                                                                                                      
     block5b_expand_bn (BatchNormal  (None, 15, 15, 672)  2688       ['block5b_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block5b_expand_activation (Act  (None, 15, 15, 672)  0          ['block5b_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block5b_dwconv (DepthwiseConv2  (None, 15, 15, 672)  16800      ['block5b_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block5b_bn (BatchNormalization  (None, 15, 15, 672)  2688       ['block5b_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block5b_activation (Activation  (None, 15, 15, 672)  0          ['block5b_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block5b_se_squeeze (GlobalAver  (None, 672)         0           ['block5b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block5b_se_reshape (Reshape)   (None, 1, 1, 672)    0           ['block5b_se_squeeze[0][0]']     
                                                                                                      
     block5b_se_reduce (Conv2D)     (None, 1, 1, 28)     18844       ['block5b_se_reshape[0][0]']     
                                                                                                      
     block5b_se_expand (Conv2D)     (None, 1, 1, 672)    19488       ['block5b_se_reduce[0][0]']      
                                                                                                      
     block5b_se_excite (Multiply)   (None, 15, 15, 672)  0           ['block5b_activation[0][0]',     
                                                                      'block5b_se_expand[0][0]']      
                                                                                                      
     block5b_project_conv (Conv2D)  (None, 15, 15, 112)  75264       ['block5b_se_excite[0][0]']      
                                                                                                      
     block5b_project_bn (BatchNorma  (None, 15, 15, 112)  448        ['block5b_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block5b_drop (Dropout)         (None, 15, 15, 112)  0           ['block5b_project_bn[0][0]']     
                                                                                                      
     block5b_add (Add)              (None, 15, 15, 112)  0           ['block5b_drop[0][0]',           
                                                                      'block5a_project_bn[0][0]']     
                                                                                                      
     block5c_expand_conv (Conv2D)   (None, 15, 15, 672)  75264       ['block5b_add[0][0]']            
                                                                                                      
     block5c_expand_bn (BatchNormal  (None, 15, 15, 672)  2688       ['block5c_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block5c_expand_activation (Act  (None, 15, 15, 672)  0          ['block5c_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block5c_dwconv (DepthwiseConv2  (None, 15, 15, 672)  16800      ['block5c_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block5c_bn (BatchNormalization  (None, 15, 15, 672)  2688       ['block5c_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block5c_activation (Activation  (None, 15, 15, 672)  0          ['block5c_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block5c_se_squeeze (GlobalAver  (None, 672)         0           ['block5c_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block5c_se_reshape (Reshape)   (None, 1, 1, 672)    0           ['block5c_se_squeeze[0][0]']     
                                                                                                      
     block5c_se_reduce (Conv2D)     (None, 1, 1, 28)     18844       ['block5c_se_reshape[0][0]']     
                                                                                                      
     block5c_se_expand (Conv2D)     (None, 1, 1, 672)    19488       ['block5c_se_reduce[0][0]']      
                                                                                                      
     block5c_se_excite (Multiply)   (None, 15, 15, 672)  0           ['block5c_activation[0][0]',     
                                                                      'block5c_se_expand[0][0]']      
                                                                                                      
     block5c_project_conv (Conv2D)  (None, 15, 15, 112)  75264       ['block5c_se_excite[0][0]']      
                                                                                                      
     block5c_project_bn (BatchNorma  (None, 15, 15, 112)  448        ['block5c_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block5c_drop (Dropout)         (None, 15, 15, 112)  0           ['block5c_project_bn[0][0]']     
                                                                                                      
     block5c_add (Add)              (None, 15, 15, 112)  0           ['block5c_drop[0][0]',           
                                                                      'block5b_add[0][0]']            
                                                                                                      
     block5d_expand_conv (Conv2D)   (None, 15, 15, 672)  75264       ['block5c_add[0][0]']            
                                                                                                      
     block5d_expand_bn (BatchNormal  (None, 15, 15, 672)  2688       ['block5d_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block5d_expand_activation (Act  (None, 15, 15, 672)  0          ['block5d_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block5d_dwconv (DepthwiseConv2  (None, 15, 15, 672)  16800      ['block5d_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block5d_bn (BatchNormalization  (None, 15, 15, 672)  2688       ['block5d_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block5d_activation (Activation  (None, 15, 15, 672)  0          ['block5d_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block5d_se_squeeze (GlobalAver  (None, 672)         0           ['block5d_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block5d_se_reshape (Reshape)   (None, 1, 1, 672)    0           ['block5d_se_squeeze[0][0]']     
                                                                                                      
     block5d_se_reduce (Conv2D)     (None, 1, 1, 28)     18844       ['block5d_se_reshape[0][0]']     
                                                                                                      
     block5d_se_expand (Conv2D)     (None, 1, 1, 672)    19488       ['block5d_se_reduce[0][0]']      
                                                                                                      
     block5d_se_excite (Multiply)   (None, 15, 15, 672)  0           ['block5d_activation[0][0]',     
                                                                      'block5d_se_expand[0][0]']      
                                                                                                      
     block5d_project_conv (Conv2D)  (None, 15, 15, 112)  75264       ['block5d_se_excite[0][0]']      
                                                                                                      
     block5d_project_bn (BatchNorma  (None, 15, 15, 112)  448        ['block5d_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block5d_drop (Dropout)         (None, 15, 15, 112)  0           ['block5d_project_bn[0][0]']     
                                                                                                      
     block5d_add (Add)              (None, 15, 15, 112)  0           ['block5d_drop[0][0]',           
                                                                      'block5c_add[0][0]']            
                                                                                                      
     block6a_expand_conv (Conv2D)   (None, 15, 15, 672)  75264       ['block5d_add[0][0]']            
                                                                                                      
     block6a_expand_bn (BatchNormal  (None, 15, 15, 672)  2688       ['block6a_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block6a_expand_activation (Act  (None, 15, 15, 672)  0          ['block6a_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block6a_dwconv_pad (ZeroPaddin  (None, 19, 19, 672)  0          ['block6a_expand_activation[0][0]
     g2D)                                                            ']                               
                                                                                                      
     block6a_dwconv (DepthwiseConv2  (None, 8, 8, 672)   16800       ['block6a_dwconv_pad[0][0]']     
     D)                                                                                               
                                                                                                      
     block6a_bn (BatchNormalization  (None, 8, 8, 672)   2688        ['block6a_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block6a_activation (Activation  (None, 8, 8, 672)   0           ['block6a_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block6a_se_squeeze (GlobalAver  (None, 672)         0           ['block6a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block6a_se_reshape (Reshape)   (None, 1, 1, 672)    0           ['block6a_se_squeeze[0][0]']     
                                                                                                      
     block6a_se_reduce (Conv2D)     (None, 1, 1, 28)     18844       ['block6a_se_reshape[0][0]']     
                                                                                                      
     block6a_se_expand (Conv2D)     (None, 1, 1, 672)    19488       ['block6a_se_reduce[0][0]']      
                                                                                                      
     block6a_se_excite (Multiply)   (None, 8, 8, 672)    0           ['block6a_activation[0][0]',     
                                                                      'block6a_se_expand[0][0]']      
                                                                                                      
     block6a_project_conv (Conv2D)  (None, 8, 8, 192)    129024      ['block6a_se_excite[0][0]']      
                                                                                                      
     block6a_project_bn (BatchNorma  (None, 8, 8, 192)   768         ['block6a_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block6b_expand_conv (Conv2D)   (None, 8, 8, 1152)   221184      ['block6a_project_bn[0][0]']     
                                                                                                      
     block6b_expand_bn (BatchNormal  (None, 8, 8, 1152)  4608        ['block6b_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block6b_expand_activation (Act  (None, 8, 8, 1152)  0           ['block6b_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block6b_dwconv (DepthwiseConv2  (None, 8, 8, 1152)  28800       ['block6b_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block6b_bn (BatchNormalization  (None, 8, 8, 1152)  4608        ['block6b_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block6b_activation (Activation  (None, 8, 8, 1152)  0           ['block6b_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block6b_se_squeeze (GlobalAver  (None, 1152)        0           ['block6b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block6b_se_reshape (Reshape)   (None, 1, 1, 1152)   0           ['block6b_se_squeeze[0][0]']     
                                                                                                      
     block6b_se_reduce (Conv2D)     (None, 1, 1, 48)     55344       ['block6b_se_reshape[0][0]']     
                                                                                                      
     block6b_se_expand (Conv2D)     (None, 1, 1, 1152)   56448       ['block6b_se_reduce[0][0]']      
                                                                                                      
     block6b_se_excite (Multiply)   (None, 8, 8, 1152)   0           ['block6b_activation[0][0]',     
                                                                      'block6b_se_expand[0][0]']      
                                                                                                      
     block6b_project_conv (Conv2D)  (None, 8, 8, 192)    221184      ['block6b_se_excite[0][0]']      
                                                                                                      
     block6b_project_bn (BatchNorma  (None, 8, 8, 192)   768         ['block6b_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block6b_drop (Dropout)         (None, 8, 8, 192)    0           ['block6b_project_bn[0][0]']     
                                                                                                      
     block6b_add (Add)              (None, 8, 8, 192)    0           ['block6b_drop[0][0]',           
                                                                      'block6a_project_bn[0][0]']     
                                                                                                      
     block6c_expand_conv (Conv2D)   (None, 8, 8, 1152)   221184      ['block6b_add[0][0]']            
                                                                                                      
     block6c_expand_bn (BatchNormal  (None, 8, 8, 1152)  4608        ['block6c_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block6c_expand_activation (Act  (None, 8, 8, 1152)  0           ['block6c_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block6c_dwconv (DepthwiseConv2  (None, 8, 8, 1152)  28800       ['block6c_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block6c_bn (BatchNormalization  (None, 8, 8, 1152)  4608        ['block6c_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block6c_activation (Activation  (None, 8, 8, 1152)  0           ['block6c_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block6c_se_squeeze (GlobalAver  (None, 1152)        0           ['block6c_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block6c_se_reshape (Reshape)   (None, 1, 1, 1152)   0           ['block6c_se_squeeze[0][0]']     
                                                                                                      
     block6c_se_reduce (Conv2D)     (None, 1, 1, 48)     55344       ['block6c_se_reshape[0][0]']     
                                                                                                      
     block6c_se_expand (Conv2D)     (None, 1, 1, 1152)   56448       ['block6c_se_reduce[0][0]']      
                                                                                                      
     block6c_se_excite (Multiply)   (None, 8, 8, 1152)   0           ['block6c_activation[0][0]',     
                                                                      'block6c_se_expand[0][0]']      
                                                                                                      
     block6c_project_conv (Conv2D)  (None, 8, 8, 192)    221184      ['block6c_se_excite[0][0]']      
                                                                                                      
     block6c_project_bn (BatchNorma  (None, 8, 8, 192)   768         ['block6c_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block6c_drop (Dropout)         (None, 8, 8, 192)    0           ['block6c_project_bn[0][0]']     
                                                                                                      
     block6c_add (Add)              (None, 8, 8, 192)    0           ['block6c_drop[0][0]',           
                                                                      'block6b_add[0][0]']            
                                                                                                      
     block6d_expand_conv (Conv2D)   (None, 8, 8, 1152)   221184      ['block6c_add[0][0]']            
                                                                                                      
     block6d_expand_bn (BatchNormal  (None, 8, 8, 1152)  4608        ['block6d_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block6d_expand_activation (Act  (None, 8, 8, 1152)  0           ['block6d_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block6d_dwconv (DepthwiseConv2  (None, 8, 8, 1152)  28800       ['block6d_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block6d_bn (BatchNormalization  (None, 8, 8, 1152)  4608        ['block6d_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block6d_activation (Activation  (None, 8, 8, 1152)  0           ['block6d_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block6d_se_squeeze (GlobalAver  (None, 1152)        0           ['block6d_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block6d_se_reshape (Reshape)   (None, 1, 1, 1152)   0           ['block6d_se_squeeze[0][0]']     
                                                                                                      
     block6d_se_reduce (Conv2D)     (None, 1, 1, 48)     55344       ['block6d_se_reshape[0][0]']     
                                                                                                      
     block6d_se_expand (Conv2D)     (None, 1, 1, 1152)   56448       ['block6d_se_reduce[0][0]']      
                                                                                                      
     block6d_se_excite (Multiply)   (None, 8, 8, 1152)   0           ['block6d_activation[0][0]',     
                                                                      'block6d_se_expand[0][0]']      
                                                                                                      
     block6d_project_conv (Conv2D)  (None, 8, 8, 192)    221184      ['block6d_se_excite[0][0]']      
                                                                                                      
     block6d_project_bn (BatchNorma  (None, 8, 8, 192)   768         ['block6d_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block6d_drop (Dropout)         (None, 8, 8, 192)    0           ['block6d_project_bn[0][0]']     
                                                                                                      
     block6d_add (Add)              (None, 8, 8, 192)    0           ['block6d_drop[0][0]',           
                                                                      'block6c_add[0][0]']            
                                                                                                      
     block6e_expand_conv (Conv2D)   (None, 8, 8, 1152)   221184      ['block6d_add[0][0]']            
                                                                                                      
     block6e_expand_bn (BatchNormal  (None, 8, 8, 1152)  4608        ['block6e_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block6e_expand_activation (Act  (None, 8, 8, 1152)  0           ['block6e_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block6e_dwconv (DepthwiseConv2  (None, 8, 8, 1152)  28800       ['block6e_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block6e_bn (BatchNormalization  (None, 8, 8, 1152)  4608        ['block6e_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block6e_activation (Activation  (None, 8, 8, 1152)  0           ['block6e_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block6e_se_squeeze (GlobalAver  (None, 1152)        0           ['block6e_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block6e_se_reshape (Reshape)   (None, 1, 1, 1152)   0           ['block6e_se_squeeze[0][0]']     
                                                                                                      
     block6e_se_reduce (Conv2D)     (None, 1, 1, 48)     55344       ['block6e_se_reshape[0][0]']     
                                                                                                      
     block6e_se_expand (Conv2D)     (None, 1, 1, 1152)   56448       ['block6e_se_reduce[0][0]']      
                                                                                                      
     block6e_se_excite (Multiply)   (None, 8, 8, 1152)   0           ['block6e_activation[0][0]',     
                                                                      'block6e_se_expand[0][0]']      
                                                                                                      
     block6e_project_conv (Conv2D)  (None, 8, 8, 192)    221184      ['block6e_se_excite[0][0]']      
                                                                                                      
     block6e_project_bn (BatchNorma  (None, 8, 8, 192)   768         ['block6e_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block6e_drop (Dropout)         (None, 8, 8, 192)    0           ['block6e_project_bn[0][0]']     
                                                                                                      
     block6e_add (Add)              (None, 8, 8, 192)    0           ['block6e_drop[0][0]',           
                                                                      'block6d_add[0][0]']            
                                                                                                      
     block7a_expand_conv (Conv2D)   (None, 8, 8, 1152)   221184      ['block6e_add[0][0]']            
                                                                                                      
     block7a_expand_bn (BatchNormal  (None, 8, 8, 1152)  4608        ['block7a_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block7a_expand_activation (Act  (None, 8, 8, 1152)  0           ['block7a_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block7a_dwconv (DepthwiseConv2  (None, 8, 8, 1152)  10368       ['block7a_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block7a_bn (BatchNormalization  (None, 8, 8, 1152)  4608        ['block7a_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block7a_activation (Activation  (None, 8, 8, 1152)  0           ['block7a_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block7a_se_squeeze (GlobalAver  (None, 1152)        0           ['block7a_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block7a_se_reshape (Reshape)   (None, 1, 1, 1152)   0           ['block7a_se_squeeze[0][0]']     
                                                                                                      
     block7a_se_reduce (Conv2D)     (None, 1, 1, 48)     55344       ['block7a_se_reshape[0][0]']     
                                                                                                      
     block7a_se_expand (Conv2D)     (None, 1, 1, 1152)   56448       ['block7a_se_reduce[0][0]']      
                                                                                                      
     block7a_se_excite (Multiply)   (None, 8, 8, 1152)   0           ['block7a_activation[0][0]',     
                                                                      'block7a_se_expand[0][0]']      
                                                                                                      
     block7a_project_conv (Conv2D)  (None, 8, 8, 320)    368640      ['block7a_se_excite[0][0]']      
                                                                                                      
     block7a_project_bn (BatchNorma  (None, 8, 8, 320)   1280        ['block7a_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block7b_expand_conv (Conv2D)   (None, 8, 8, 1920)   614400      ['block7a_project_bn[0][0]']     
                                                                                                      
     block7b_expand_bn (BatchNormal  (None, 8, 8, 1920)  7680        ['block7b_expand_conv[0][0]']    
     ization)                                                                                         
                                                                                                      
     block7b_expand_activation (Act  (None, 8, 8, 1920)  0           ['block7b_expand_bn[0][0]']      
     ivation)                                                                                         
                                                                                                      
     block7b_dwconv (DepthwiseConv2  (None, 8, 8, 1920)  17280       ['block7b_expand_activation[0][0]
     D)                                                              ']                               
                                                                                                      
     block7b_bn (BatchNormalization  (None, 8, 8, 1920)  7680        ['block7b_dwconv[0][0]']         
     )                                                                                                
                                                                                                      
     block7b_activation (Activation  (None, 8, 8, 1920)  0           ['block7b_bn[0][0]']             
     )                                                                                                
                                                                                                      
     block7b_se_squeeze (GlobalAver  (None, 1920)        0           ['block7b_activation[0][0]']     
     agePooling2D)                                                                                    
                                                                                                      
     block7b_se_reshape (Reshape)   (None, 1, 1, 1920)   0           ['block7b_se_squeeze[0][0]']     
                                                                                                      
     block7b_se_reduce (Conv2D)     (None, 1, 1, 80)     153680      ['block7b_se_reshape[0][0]']     
                                                                                                      
     block7b_se_expand (Conv2D)     (None, 1, 1, 1920)   155520      ['block7b_se_reduce[0][0]']      
                                                                                                      
     block7b_se_excite (Multiply)   (None, 8, 8, 1920)   0           ['block7b_activation[0][0]',     
                                                                      'block7b_se_expand[0][0]']      
                                                                                                      
     block7b_project_conv (Conv2D)  (None, 8, 8, 320)    614400      ['block7b_se_excite[0][0]']      
                                                                                                      
     block7b_project_bn (BatchNorma  (None, 8, 8, 320)   1280        ['block7b_project_conv[0][0]']   
     lization)                                                                                        
                                                                                                      
     block7b_drop (Dropout)         (None, 8, 8, 320)    0           ['block7b_project_bn[0][0]']     
                                                                                                      
     block7b_add (Add)              (None, 8, 8, 320)    0           ['block7b_drop[0][0]',           
                                                                      'block7a_project_bn[0][0]']     
                                                                                                      
     top_conv (Conv2D)              (None, 8, 8, 1280)   409600      ['block7b_add[0][0]']            
                                                                                                      
     top_bn (BatchNormalization)    (None, 8, 8, 1280)   5120        ['top_conv[0][0]']               
                                                                                                      
     top_activation (Activation)    (None, 8, 8, 1280)   0           ['top_bn[0][0]']                 
                                                                                                      
     avg_pool (GlobalAveragePooling  (None, 1280)        0           ['top_activation[0][0]']         
     2D)                                                                                              
                                                                                                      
     top_dropout (Dropout)          (None, 1280)         0           ['avg_pool[0][0]']               
                                                                                                      
     predictions (Dense)            (None, 1000)         1281000     ['top_dropout[0][0]']            
                                                                                                      
    ==================================================================================================
    Total params: 7,856,239
    Trainable params: 7,794,184
    Non-trainable params: 62,055
    __________________________________________________________________________________________________
    


```python
!wget -O plane.jpg https://upload.wikimedia.org/wikipedia/commons/1/12/Plane-in-flight.jpg
```

    --2023-03-09 15:11:13--  https://upload.wikimedia.org/wikipedia/commons/1/12/Plane-in-flight.jpg
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 103.102.166.240, 2001:df2:e500:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|103.102.166.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 487351 (476K) [image/jpeg]
    Saving to: ‘plane.jpg’
    
    plane.jpg           100%[===================>] 475.93K  --.-KB/s    in 0.02s   
    
    2023-03-09 15:11:14 (29.2 MB/s) - ‘plane.jpg’ saved [487351/487351]
    
    


```python
# 다운로드한 plane.jpg를 terget_size로 줄여줌
# plane input은 [(None, 240, 240, 3  0)] 크기를 가짐
img = image.load_img('plane.jpg', target_size=(240, 240))

plt.imshow(img)

# 이미지를 array 형태로 변환(학습할 모델의 입력으로 사용하기 위해)
x = image.img_to_array(img)
# 기존 모형에 batch_size=1 추가
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

# 예측
preds = eff.predict(x)

# 예측 결과 decode_predictions을 통해 반환
print(decode_predictions(preds))
```

    1/1 [==============================] - 2s 2s/step
    [[('n02690373', 'airliner', 0.75877386), ('n04592741', 'wing', 0.100808546), ('n04552348', 'warplane', 0.08239551), ('n04266014', 'space_shuttle', 0.0063430285), ('n02692877', 'airship', 0.00088727544)]]
    
