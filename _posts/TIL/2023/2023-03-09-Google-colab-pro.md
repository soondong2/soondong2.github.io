---
title: "Colab Pro"
date: 2023-03-09

categories:
  - TIL
tags:
  - TIL
---

## Googl Colab Pro
모델 돌리다가 어느순간 "사용량 제한으로 인해 GPU에 연결할 수 없습니다." 라는 문구가 떴다. 못 쓴다니까.. 없이 쓰지 뭐 하고 이어서 모델을 돌리자 답답함에 미칠 것 같았다. 어쩔 수 없이 Pro 결제했다. 지현언니가 Pro Plus 사용하고 있다는 얘기는 들었었는데.. Pro도 부족해지면 나도 플러스를 결제해야 할지도 모르겠다.

`런타임` > `런타임 유형 변경`에서 GPU를 선택해준다. 이후 아래 코드를 실행시켜주면 된다고 한다.
```python
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
```

아래는 추가 메모리 사용 방법이라고 한다.
```python
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
```
