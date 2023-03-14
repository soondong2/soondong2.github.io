---
title: "YOLOv5를 활용한 영어 수어 탐지"
date: 2023-03-14

categories:
  - AI
  - Deep Learning
tags:
  - Computer Vision
  - Object Detection
  - YOLO
---

```python
!curl -L "https://public.roboflow.com/ds/SWOifNPYMj?key=c5J8MPAoer" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
``` 
    
```python
%cd /content
!git clone https://github.com/ultralytics/yolov5.git
```


```python
%cd /content/yolov5/
!pip install -r requirements.txt
```


```python
%cat /content/dataset/data.yaml
```

    train: ../train/images
    val: ../valid/images
    
    nc: 26
    names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


```python
from glob import glob

train_img_list = glob('/content/dataset/train/images/*.jpg')
test_img_list = glob('/content/dataset/test/images/*.jpg')
valid_img_list = glob('/content/dataset/valid/images/*.jpg')

print('Train Images : ', len(train_img_list))
print('Valid Images : ', len(valid_img_list))
print('Test Images : ', len(test_img_list))
```

    Train Images :  1512
    Valid Images :  144
    Test Images :  72
    


```python
# 이미지 주소 및 파일명 확인
print(train_img_list[0])
print(valid_img_list[0])
print(test_img_list[0])
```

    /content/dataset/train/images/H12_jpg.rf.c6e4d175bd242e793456f9a8dfad8bba.jpg
    /content/dataset/valid/images/J8_jpg.rf.b1ba18316b18b810d7d2c805627f13e0.jpg
    /content/dataset/test/images/T13_jpg.rf.bf67ceb39727be048066c0de76801971.jpg
    


```python
# Image를 txt file로 저장
with open('/content/dataset/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('/content/dataset/valid.txt', 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')

with open('/content/dataset/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')
```

- 경로 변경


```python
import yaml

with open('/content/dataset/data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

print(data)

data['train'] = '/content/dataset/train.txt'
data['val'] = '/content/dataset/valid.txt'

with open('/content/dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)

print(data)
```

    {'train': '../train/images', 'val': '../valid/images', 'nc': 26, 'names': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']}
    {'train': '/content/dataset/train.txt', 'val': '/content/dataset/valid.txt', 'nc': 26, 'names': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']}
    

- 사전 학습 모델을 통한 학습


```python
%cd /content/yolov5
!python train.py --img 300 --batch 16 --epochs 50 --data /content/dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name sign_language_yolov5s_results
```
    
    Overriding model.yaml nc=80 with nc=26
    
                     from  n    params  module                                  arguments                     
      0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
      1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
      2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
      3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
      4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
      5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
      6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
      7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
      8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
      9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
     10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
     14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
     18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
     21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     24      [17, 20, 23]  1     83607  models.yolo.Detect                      [26, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    YOLOv5s summary: 214 layers, 7089751 parameters, 7089751 gradients, 16.2 GFLOPs
    
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           0/49     0.893G     0.1231    0.01537    0.09194         35        320:   0% 0/95 [00:01<?, ?it/s]WARNING ⚠️ TensorBoard graph visualization failure Sizes of tensors must match except in dimension 1. Expected size 20 but got size 19 for tensor number 1 in the list.
           0/49     0.958G    0.08144     0.0207    0.08547         16        320: 100% 95/95 [00:11<00:00,  8.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:02<00:00,  2.42it/s]
                       all        144        144     0.0314      0.452      0.071     0.0281
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
           1/49      1.21G    0.04911    0.01805    0.08012         16        320: 100% 95/95 [00:08<00:00, 11.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:00<00:00,  6.33it/s]
                       all        144        144     0.0342      0.842      0.119     0.0778

          ...
          
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          48/49      1.21G    0.01746   0.008136    0.01865         21        320: 100% 95/95 [00:07<00:00, 12.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:00<00:00,  6.52it/s]
                       all        144        144      0.902      0.868      0.936      0.765
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          49/49      1.21G    0.01736   0.008104     0.0189         16        320: 100% 95/95 [00:07<00:00, 11.91it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 5/5 [00:00<00:00,  6.61it/s]
                       all        144        144      0.867      0.915      0.946      0.781
    
    50 epochs completed in 0.128 hours.
    Optimizer stripped from runs/train/sign_language_yolov5s_results2/weights/last.pt, 14.4MB
    Optimizer stripped from runs/train/sign_language_yolov5s_results2/weights/best.pt, 14.4MB
    
    Validating runs/train/sign_language_yolov5s_results2/weights/best.pt...
    Fusing layers... 
    YOLOv5s summary: 157 layers, 7080247 parameters, 0 gradients, 16.0 GFLOPs
Results saved to runs/train/sign_language_yolov5s_results2
    

- Tensorboard


```python
%load_ext tensorboard
%tensorboard --logdir /content/yolov5/runs/
```

![image](https://user-images.githubusercontent.com/100760303/224992618-1df5b1c6-7449-4bb0-b72d-946265b9f488.png)



- 이미지


```python
from IPython.display import Image
import os

val_img = valid_img_list[0]

!python detect.py --weights /content/yolov5/runs/train/sign_language_yolov5s_results2/weights/best.pt --img 300 --conf 0.5 --source '{val_img}'
Image(os.path.join('runs/detect/exp', os.path.basename(val_img)))
```
Results saved to runs/detect/exp3
    
![image](https://user-images.githubusercontent.com/100760303/224990956-68a5c839-31d5-4ed8-9dc5-6428d73d1066.png)

- 이 외에도 번호를 변경해가며 예측한 이미지

![image](https://user-images.githubusercontent.com/100760303/224991226-c86e06b7-ad53-4338-8c3f-eb5beace15f3.png)

![image](https://user-images.githubusercontent.com/100760303/224991373-06c3eb71-520e-4d94-be5a-d1a01c6290b1.png)

![image](https://user-images.githubusercontent.com/100760303/224991559-8b3a593a-90fe-4a7c-9631-0d577c968520.png)


- 비디오


```python
!python detect.py --source ../test2.mp4 --weights /content/yolov5/runs/train/sign_language_yolov5s_results2/weights/best.pt
```
Results saved to runs/detect/exp16

![test (1)](https://user-images.githubusercontent.com/100760303/224994561-dd02ad63-e1dc-470a-bb40-0dd2f5623b03.gif)

용량 제한으로 크기를 강제로 줄였더니 화질이 이상해졌다..
  
- Webcam


```python
# !python detect.py --source 0 --weights /content/yolov5/runs/train/sign_language_yolov5s_results2/weights/best.pt
```
