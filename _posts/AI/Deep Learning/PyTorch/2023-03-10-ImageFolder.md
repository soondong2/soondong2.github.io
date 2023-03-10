---
title: "[PyTorch] ImageFolder"
date: 2023-03-10

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - PyTorch
---

## Library Call


```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```




    'cuda'




```python
# seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

seed_everything(111)
```

- `transforms.Compose()`는 transform을 여러 개 수행할 경우 묶어주는 역할
- `transforms.Resize((height, width))` : image size 변경
- `ImageFolder`를 사용해 이미지 불러옴


```python
trans = transforms.Compose([transforms.Resize((64, 128))])

train_data = torchvision.datasets.ImageFolder(root='', transform=trans)
```

주의할 점은 `data.save`에 지정된 `path`에 폴더가 생성되어 있어야 한다.


```python
for num, value in enumerate(train_data):
    # data(image information), label
    data, label = value
    print(num, data, label)
    
    if (label == 0):
        data.save('custom_data/train_data/gray/%d_%d.jpeg'%(num, label))
    else:
        data.save('custom_data/train_data/red/%d_%d.jpeg'%(num, label))
```

- 초창기 데이터 학습을 위해 train image data를 불러올 때


```python
data_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, num_workers=2)
```

- 모델 학습 이후 예측을 위해 test image data를 불러올 때


```python
test_set = DataLoader(dataset = test_data, batch_size = len(test_data))
```
