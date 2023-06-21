---
title: "[PyTorch] nn.ConvTranspose2d()"
date: 2023-06-21

categories:
  - AI
  - Deep Learning
tags:
    - PyTorch
---

## torch.nn.ConvTranspose2d()

Input으로 들어갈 tensor를 생성한다. 4차원으로 만든 이유는, PyTorch에서 기본적으로 4차원 Input만을 지원하기 때문이다. 실제로 값이 있는 것은 2 x 3 형태이지만, shape를 맞추기 위해 4차원으로 구성한 것이다.

```python
import torch
import torch.nn as nn
```

```python
test_input = torch.Tensor([[[[1, 2, 3], [4, 5, 6]]]])

print("input size: ", test_input.shape)
print("test input: ", test_input)
```

input size:  torch.Size([1, 1, 2, 3])
test input:  tensor([[[[1., 2., 3.],
          [4., 5., 6.]]]])

오늘 공부하려고 하는 `nn.ConvTranspose2d()`를 만들어준다. 파라미터는 다음과 같다. 계산을 간단하게 하기 위해 bias=False로 설정하였다.

- in_channels = 1
- out_channels = 3
- kernel_size = 4
- stride = 1
- padding = 0

```python
class sample(nn.Module):
  def __init__(self):
    super(sample, self).__init__()
    self.main = nn.ConvTranspose2d(1, 3, 4, 1, 0, bias = False)

  def forward(self, input):
    return self.main(input)
```

모델의 기본 weights이다. ConvTranspose2d()에서 kernel_size=4로 설정했기 때문에 filter의 사이즈가 4 x 4 형태로 나타난다. out_channels=3으로 지정했기 때문에 3이 나타났다. 즉, 가로 4, 세로 3, 높이 4인 직육면체로 생각하면 된다.

```python
Model = sample()

# Print model's original parameters.
for name, param in Model.state_dict().items():
  print("name: ", name)
  print("Param: ", param)
```

Param:  tensor([[[[-0.1415,  0.0249,  0.1062,  0.0484],
          [ 0.1260, -0.0197, -0.0335, -0.0480],
          [-0.0308,  0.0576,  0.0031,  0.0795],
          [-0.0197,  0.0875,  0.1386,  0.0724]],

         [[ 0.1015,  0.0026, -0.0775,  0.1388],
          [ 0.1351, -0.1060, -0.1006,  0.0680],
          [-0.1009, -0.1432,  0.1185,  0.0214],
          [-0.1267,  0.1334,  0.0859, -0.0433]],

         [[ 0.0303,  0.0461,  0.0357, -0.0122],
          [-0.1437, -0.0509,  0.0760, -0.0131],
          [ 0.0889,  0.0432,  0.1315, -0.0877],
          [-0.0085,  0.1286,  0.0586, -0.0928]]]])

결과를 쉽게 보기 위해 모델의 파라미터를 변경하는 과정을 거친다. 그러면 다음과 같이 만들어진다.

<img width="707" alt="스크린샷 2023-06-21 오후 5 55 39" src="https://github.com/soondong2/TIL/assets/100760303/70fb86d8-0205-40ef-86c7-858fe443395f">


이후 (1, 1, 2, 3) input을 위의 weights를 이용해 nn.ConvTranspose2d() 연산을 수행한다.

```python
result = Model(test_input)

print("Result shape: ", result.shape)
print("Result: ", result)
```

Result shape:  torch.Size([1, 3, 5, 6])
Result:  tensor([[[[ 0.1000,  0.4000,  1.0000,  1.6000,  1.7000,  1.2000],
          [ 0.9000,  2.9000,  6.2000,  8.3000,  7.5000,  4.8000],
          [ 2.9000,  7.7000, 14.6000, 16.7000, 13.9000,  8.4000],
          [ 4.9000, 12.5000, 23.0000, 25.1000, 20.3000, 12.0000],
          [ 5.2000, 12.1000, 20.8000, 22.3000, 17.0000,  9.6000]],

         [[ 1.7000,  5.2000, 10.6000, 11.2000,  9.7000,  6.0000],
          [ 8.9000, 22.1000, 39.8000, 41.9000, 33.1000, 19.2000],
          [10.9000, 26.9000, 48.2000, 50.3000, 39.5000, 22.8000],
          [12.9000, 31.7000, 56.6000, 58.7000, 45.9000, 26.4000],
          [11.6000, 26.5000, 44.8000, 46.3000, 34.6000, 19.2000]],

         [[ 3.3000, 10.0000, 20.2000, 20.8000, 17.7000, 10.8000],
          [16.9000, 41.3000, 73.4000, 75.5000, 58.7000, 33.6000],
          [18.9000, 46.1000, 81.8000, 83.9000, 65.1000, 37.2000],
          [20.9000, 50.9000, 90.2000, 92.3000, 71.5000, 40.8000],
          [18.0000, 40.9000, 68.8000, 70.3000, 52.2000, 28.8000]]]],
       grad_fn=<ConvolutionBackward0>)
       
## 참고 자료
https://cumulu-s.tistory.com/29