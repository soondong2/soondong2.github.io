---
title: "[Paper] PyTorch로 U-Net 구현하기"
date: 2023-06-14

categories:
  - AI
  - Deep Learning
tags:
    - Paper
---

## 0. Library Call

```python
import os
import numpy as np
from PIL import Image # conda install pillow
import matplotlib.pyplot as plt
```

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets # conda install pytorch torchvision -c pytorch
```

## 1. Dataset Download and Formatting

```python
# 30개의 frame으로 이루어져 있는 tif 파일을 하나의 frame씩 나누어 따로 저장하도록

dir_data = "/content/"

name_label = "train-labels.tif"
name_input = "train-volume.tif"

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames
```

```python
# Train & Validation & Test
nframe_train = 24
nframe_val = 3
nframe_test = 3

# 데이터가 저장될 디렉토리 설정
dir_save_train = os.path.join(dir_data, "train")
dir_save_val = os.path.join(dir_data, "valid")
dir_save_test = os.path.join(dir_data, "test")

# 디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)
```

```python
# 랜덤하게 저장하기 위한 frame에 대한 랜덤 인덱스 생성
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)
```

```python
# Train Dataset Save
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, "label_%03d.npy" % i), label_) # label_001.npy
    np.save(os.path.join(dir_save_train, "input_%03d.npy" % i), input_) # input_001.npy
```

```python
# Validation Dataset Save
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, "label_%03d.npy" % i), label_) # label_001.npy
    np.save(os.path.join(dir_save_val, "input_%03d.npy" % i), input_) # input_001.npy
```

```python
# Test Dataset Save
offset_nframe += nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, "label_%03d.npy" % i), label_) # label_001.npy
    np.save(os.path.join(dir_save_test, "input_%03d.npy" % i), input_) # input_001.npy
```

```python
# 생성된 데이터셋 출력 - 하얀색 255 / 검은색 0
plt.subplot(121)
plt.imshow(label_, cmap="gray")
plt.title("label")

plt.subplot(122)
plt.imshow(input_, cmap="gray")
plt.title("Input")

plt.show()
```

## 2. U-Net Architecture

```python
# 하이퍼 파라미터
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = "/content/" # 데이터가 저장되어 있는 디렉토리
check_dir = "./checkpoint" # 훈련 데이터가 저장 될 디렉토리
log_dir = "./log" # 텐서보드 디렉토리
result_dir = "./results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, "png"))
    os.makedirs(os.path.join(result_dir, "numpy"))
```

```python
# Network 구현

# U-Net Layer 생성
class UNet(nn.Module):
    # UNet을 정의하는 데에 필요한 Layer 선언
    def __init__(self):
        super(UNet, self).__init__()

        # Convolution, Batch Normalization, ReLU (반복적으로 사용하기 위해 정의)
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []

            # Convolutional
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]

            # Batch Normalization
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            # ReLU
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contraction Path(Encoder)
        self.encoder1_1 = CBR2d(in_channels=1, out_channels=64)
        self.encoder1_2 = CBR2d(in_channels=64, out_channels=64)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2_1 = CBR2d(in_channels=64, out_channels=128)
        self.encoder2_2 = CBR2d(in_channels=128, out_channels=128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3_1 = CBR2d(in_channels=128, out_channels=256)
        self.encoder3_2 = CBR2d(in_channels=256, out_channels=256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder4_1 = CBR2d(in_channels=256, out_channels=512)
        self.encoder4_2 = CBR2d(in_channels=512, out_channels=512)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.encoder5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive Path(Decoder)
        self.decoder5_1 = CBR2d(in_channels=1024, out_channels=512) # Encoder의 마지막 stage와 반대되는 size로 진행

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.decoder4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.decoder3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.decoder2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.decoder1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1, stride=1, padding=0)


    # U-Net Layer 연결 : x는 input image를 의미
    def forward(self, x):
        # Encoder Part 연결
        encoder1_1 = self.encoder1_1(x)
        encoder1_2 = self.encoder1_2(encoder1_1)
        maxpool1 = self.maxpool1(encoder1_2)

        encoder2_1 = self.encoder2_1(maxpool1)
        encoder2_2 = self.encoder2_2(encoder2_1)
        maxpool2 = self.maxpool2(encoder2_2)

        encoder3_1 = self.encoder3_1(maxpool2)
        encoder3_2 = self.encoder3_2(encoder3_1)
        maxpool3 = self.maxpool3(encoder3_2)

        encoder4_1 = self.encoder4_1(maxpool3)
        encoder4_2 = self.encoder4_2(encoder4_1)
        maxpool4 = self.maxpool4(encoder4_2)

        encoder5_1 = self.encoder5_1(maxpool4)

        # Decoder Part 연결
        decoder5_1 = self.decoder5_1(encoder5_1)

        unpool4 = self.unpool4(decoder5_1)
        concat4 = torch.cat((unpool4, encoder4_2), dim=1) # dim=0(batch), dim=1(channel), dim=2(height), dim3=(width)
        decoder4_2 = self.decoder4_2(concat4)
        decoder4_1 = self.decoder4_1(decoder4_2)

        unpool3 = self.unpool3(decoder4_1)
        concat3 = torch.cat((unpool3, encoder3_2), dim=1)
        decoder3_2 = self.decoder3_2(concat3)
        decoder3_1 = self.decoder3_1(decoder3_2)

        unpool2 = self.unpool2(decoder3_1)
        concat2 = torch.cat((unpool2, encoder2_2), dim=1)
        decoder2_2 = self.decoder2_2(concat2)
        decoder2_1 = self.decoder2_1(decoder2_2)

        unpool1 = self.unpool1(decoder2_1)
        concat1 = torch.cat((unpool1, encoder1_2), dim=1)
        decoder1_2 = self.decoder1_2(concat1)
        decoder1_1 = self.decoder1_1(decoder1_2)

        x = self.fc(decoder1_1)

        return x
```

## 3. Data Loader & Transform
### Data Loader
```python
# Data Loader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 디렉토리에 있는 모든 파일의 리스트
        lst_data = os.listdir(self.data_dir)

        # label, input 리스트
        lst_label = [f for f in lst_data if f.startswith("label")]
        lst_input = [f for f in lst_data if f.startswith("input")]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    # index에 해당하는 파일
    def __getitem__(self, index):
        # dataset이 numpy 형태로 저장되어 있기 때문에 np.load 사용
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 0~255를 0~1 사이로 normalization
        label = label / 255.0
        input = input / 255.0

        # Neural Network에 들어가는 input은 dim=3
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {"input" : input, "label" : label}

        if self.transform:
            data = self.transform(data)

        return data
```

```python
dataset_train = Dataset(data_dir=os.path.join(data_dir, "train"))

data = dataset_train.__getitem__(0)

input = data["input"]
label = data["label"]
```

```python
plt.subplot(121)
plt.imshow(input)
plt.title("Input")


plt.subplot(122)
plt.imshow(label)
plt.title("Label")

plt.show()
```

### Transform

```python
# ToTensor() : numpy -> tensor
class ToTensor(object):
    # dictionary 형태로 input, label을 갖는 data
    def __call__(self, data):
        label, input = data["label"], data["input"]

        # numpy에서 channel을 첫 번째로 옮기고 나머지는 그대로
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # numpy -> tensor
        data = {"label" : torch.from_numpy(label), "input" : torch.from_numpy(input)}

        return data

# Narmalization
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data["label"], data["input"]

        input = (input - self.mean) / self.std

        data = {"label" : label, "input" : input}

        return data

# RandomFlip() : 랜덤하게 좌우 상하 플립
class RandomFlip(object):
    def __call__(self, data):
        label, input = data["label"], data["input"]

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.fliplr(input)

        data = {"label" : label, "input" : input}

        return data
```

```python
# Transform이 제대로 작동하는지 테스트
transform = transforms.Compose([Normalization(mean=0.5, std=9.5),
                                RandomFlip(),
                                ToTensor()])

dataset_train = Dataset(data_dir=os.path.join(data_dir, "train"), transform=transform)

data = dataset_train.__getitem__(0)

input = data["input"]
label = data["label"]

plt.subplot(121)
plt.imshow(input.squeeze())


plt.subplot(122)
plt.imshow(label.squeeze())

plt.show()
```

## 4. Model Training

```python
transform = transforms.Compose([Normalization(mean=0.5, std=9.5),
                                RandomFlip(),
                                ToTensor()])
# Train dataset Load
dataset_train = Dataset(data_dir=os.path.join(data_dir, "train"), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

# Validation dataset Load
dataset_val = Dataset(data_dir=os.path.join(data_dir, "valid"), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)
```

```python
# Network 생성
net = UNet().to(device)

# 손실 함수(Loss Function)
fn_loss = nn.BCEWithLogitsLoss().to(device)

# 옵티마이저(Optimizer)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Loss 계산에 필요한 부수적인 variable
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_train / batch_size)

# Output 저장을 위한 함수
fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1) # tensor -> numpy
fn_denorm = lambda x, mean, std: (x * std) + mean # normalization -> denormalization
fn_class = lambda x: 1.0 * (x > 0.5) # output image를 binary class로 분류

# Tensorboard를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, "train"))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, "valid"))
```

```python
# Network Save
def save(check_dir, net, optim, epoch):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    
    torch.save({"net" : net.state_dict(), "optim" : optim.state_dict()},
               "./%s/model_epoch%d.pth" % (check_dir, epoch))

# Networ Load
def load(check_dir, net, optim):
    if not os.path.exists(check_dir):
        epoch = 0

        return net, optim, epoch

    check_lst = os.listdir(check_dir)
    check_lst.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    dict_model = torch.load(".%s/%s" % (check_dir, check_lst[-1]))

    net.load_state_dict(dict_model["net"])
    optim.load_state_dict(dict_model["optim"])
    epoch = int(check_lst[-1].split("epoch")[1].split("pth")[0])

    return net, optim, epoch
```

```python
# Traning
st_epoch = 0

net, optim, st_epoch = load(check_dir=check_dir, net=net, optim=optim)

for epoch in range(st_epoch + 1, num_epoch + 1):
    # Network에게 Training임을 알려줌
    net.train()
    
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        # Forward pass
        label = data["label"].to(device)
        input = data["input"].to(device)

        output = net(input)

        # Backward pass(역전파)
        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()

        optim.step()

        # 손실 함수 계산
        loss_arr += [loss.item()]

        print("Train: Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f" %
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
        
    # Validation
    # validation할 때는 backpropagation을 사용하지 않으므로 torch.no_grad() 사용
    with torch.no_grad():
        # Netword에게 vadidation임을 명시
        net.eval()

        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # Forward pass
            label = data["label"].to(device)
            input = data["input"].to(device)

            output = net(input)

            # backproagation을 하지 않으므로 backward 불필요

            # 손실 함수 계산
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print("Valid: Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f" % 
             (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

    # 1-epoch이 진행될 때마다 네트워크 저장
    save(check_dir=check_dir, net=net, optim=optim, epoch=epoch)
```

## 5. Evaluation

```python
transform = transforms.Compose([Normalization(mean=0.5, std=9.5),
                                ToTensor()])
# Train dataset Load
dataset_test = Dataset(data_dir=os.path.join(data_dir, "train"), transform=transform)
loader_test = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=8)
```

```python
# Network 생성
net = UNet().to(device)

# 손실 함수(Loss Function)
fn_loss = nn.BCEWithLogitsLoss().to(device)

# 옵티마이저(Optimizer)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Loss 계산에 필요한 부수적인 variable
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)

# Output 저장을 위한 함수
fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1) # tensor -> numpy
fn_denorm = lambda x, mean, std: (x * std) + mean # normalization -> denormalization
fn_class = lambda x: 1.0 * (x > 0.5) # output image를 binary class로 분류
```

```python
# 아래 코드에서 checkpoint 디렉토리 오류날 경우 실행
check_dir = "/checkpoint"

# result_dir 디렉토리 오류 아래와 같이 수정
result_dir = "/content/results/"
```

```python
st_epoch = 0

net, optim, st_epoch = load(check_dir=check_dir, net=net, optim=optim)

with torch.no_grad():
    # Netword에게 vadidation임을 명시
    net.eval()

    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        # Forward pass
        label = data["label"].to(device)
        input = data["input"].to(device)

        output = net(input)

        # backproagation을 하지 않으므로 backward 불필요

        # 손실 함수 계산
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("Test: Batch %04d / %04d | Loss %.4f" % 
         (batch, num_batch_test, np.mean(loss_arr)))
        
        # Tensorboard의 input, output, label을 저장
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = num_batch_test * (batch - 1) + j

            # png 파일
            # "/content/results/png"
            plt.imsave(os.path.join(result_dir, "png", "label_%04d.png" % id), label[j].squeeze(), cmap="gray")
            plt.imsave(os.path.join(result_dir, "png", "input_%04d.png" % id), input[j].squeeze(), cmap="gray")
            plt.imsave(os.path.join(result_dir, "png", "output_%04d.png" % id), output[j].squeeze(), cmap="gray")

            # numpy type 파일
            # "/content/results/numpy"
            np.save(os.path.join(result_dir, "numpy", "label_%04d.npy" % id), label[j].squeeze())
            np.save(os.path.join(result_dir, "numpy", "input_%04d.npy" % id), input[j].squeeze())
            np.save(os.path.join(result_dir, "numpy", "output_%04d.npy" % id), output[j].squeeze())

print("Average Test: Batch %04d / %04d | Loss %.4f" %
      (batch, num_batch_test, np.mean(loss_arr)))
```

```python
result_dir = "./results/numpy"

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith("label")]
lst_input = [f for f in lst_data if f.startswith("input")]
lst_output = [f for f in lst_data if f.startswith("output")]

lst_label.sort()
lst_input.sort()
lst_output.sort()
```

```python
# output이 검은색으로 나온 이유는 최소 100 epoch를 학습시켜야 하지만, 코랩 컴퓨팅 자원 고갈로 2 epoch 밖에 안 했기 때문
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

plt.subplot(131)
plt.imshow(input, cmap="gray")
plt.title("Input")

plt.subplot(132)
plt.imshow(label, cmap="gray")
plt.title("Label")

plt.subplot(133)
plt.imshow(output, cmap="gray")
plt.title("Output")
```