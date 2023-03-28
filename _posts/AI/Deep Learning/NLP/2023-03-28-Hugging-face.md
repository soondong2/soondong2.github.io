---
title: "[NLP] Hugging Face란?"
date: 2023-03-28

categories:
  - AI
  - Deep Learning
tags:
  - NLP
  - Hugging Face
---

## 트랜스포머(Transformer)
- 인공신경망 알고리즘은 크게, 합성곱 신경망(CNN), 순환 신경망(RNN), 트랜스포머(Transformer) 3가지로 나뉘어진다.
- 트랜스포머는, 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로, `셀프 에텐션(Self-Attention)`이라는 방식을 사용하는 모델이다.
- 트랜스포머는 `어텐션` 방식을 사용해, 문장 전체를 `병렬 구조`로 번역할 뿐만 아니라, 멀리 있는 단어까지도 연관성을 만들어 유사성을 높였으며, RNN의 한계를 극복했다.
- 이미지나 언어 번역에 폭넓게 쓰이고 있으며, `GPT-3`, `BERT` 등이 가장 관심을 많이 받고 있는 모델이다.

## Hugging Face
- `허깅 페이스`는 자연어 처리 스타트업이 개발했다.
- 다양한 `트랜스포머 모델(transformer.models)`과 `학습 스크립트(transformer.Trainer)`를 제공하는 모듈이다.
- 허깅 페이스를 사용한다면, 트랜스포머 모델 사용시 layer, model 등을 선언하거나 학습 스크립트를 구현해야하는 수고를 덜 수 있다.
- 참고 링크 : https://github.com/huggingface/transformers
- 일반적인 layer.py, model.py는 `transformer.models`로, train.py는 `transformer.Trainer`로 대응해서 사용할 수 있다.

### transformers.models
- 트랜스포머 기반의 다양한 모델을 `pytorch`, `tensorflow`로 각각 구현해놓은 모듈이다.
- 각 모델에 맞는 `tokenizer`도 구현되어 있다.

### transformers.Trainer
- 딥러닝 학습 및 평가에 필요한 `optimizer`, `weight updt`, `learning rate schedule`, `ckpt`, `tensorbord`, `evaluation` 등을 수행하는 모듈다.
- `Trainer.train` 함수를 호출하면, 이 모든 과정이 사용자가 원하는 arguments에 맞게 실행된다.
- `pytorch lightning`과 비슷하게 공통적으로 사용되는 학습 스크립트를 모듈화 하여 편하게 사용할 수 있다.
