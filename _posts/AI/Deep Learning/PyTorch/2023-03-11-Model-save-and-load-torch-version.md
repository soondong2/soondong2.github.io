---
title: "[PyTorch] Model Save & Load"
date: 2023-03-11

categories:
  - AI
  - Deep Learning
tags:
  - DL
  - PyTorch
---

## 모델 저장 및 로드
- `torch.save` : net.state_dict()를 저장
- `torch.load` : load_state_dict로 모델을 로드

```python
PATH = '저장할 경로'
torch.save(net.state_dict(), PATH)
```
```python
net = NeuralNet()
net.load_state_dict(torch.load(PATH))
```
