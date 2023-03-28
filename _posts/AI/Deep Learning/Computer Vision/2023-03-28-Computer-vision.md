---
title: "[CV] 컴퓨터 비전(Computer Vision) 세부 분야"
date: 2023-03-28

categories:
  - AI
  - Deep Learning
tags:
  - Computer Vision
---

## Classification

- `ImageNet` 대회로 많이 알려진 분야
- 이미지가 입력되면 특정 class로 분류하는 분야

## Detection

- 이미지에서 물체를 포함하는 `bounding box`를 찾는 알고리즘
- 다수의 물체에 대해 bbox 좌표와 물체의 class를 출력

## Segmentation

- 픽셀 단위로 class를 분류하는 알고리즘
- 분할 알고리즘이 포함되며 지도학습 뿐만 아니라 `meanshift` 등과 같은 비지도 학습 기반 방법도 사용

## Feature Extraction

- 특징을 추출하는 알고리즘
- 특징이란, 물체의 형태나 크기, 위치가 변해도 쉽게 식별이 가능한 점
- 대체로 코너점이 특징점으로 이용하기 쉬우며 `Harris corner` 알고리즘 등이 포함

## Feature Description

- 추출된 특징점이 특징 조건을 만족할 때까지 특징 기술자로 불림
- 조건은 `분별력`, `불변`, `크기`
- 대표적인 기술자로는 `SIFT`, `HOG`, `FAST`

## Matching

- 서로 다른 영상의 시점이 다를 때, 각 영상을 변형하여 하나의 좌표계로 표현하는 방법
- 특정 기술자를 이용해 동일한 위치의 특징점을 찾아 선형/비선형 변환하는 방법으로 구현
- 영상 의학에서 `CT`, `MRI` 상의 환부를 동일한 위치로 표현할 때 사용

## 3D Vision

- 서로 다른 2개 이상의 영상을 이용해 3차원 좌표로 변환
- 삼각 측량법이 주로 사용됨 `stereo vision`, `multiple-view`, `visual SLAM` 등
- 카메라 파라미터 계산부터 영상 정합

## Tracking

- 영상에서 관심 물체를 시간축으로 추적하는 알고리즘
- `One-shot learning`이라고도 표현
- `Kalman filter`는 가장 유명한 추적 알고리즘

## Generative Model

- 딥러닝의 발전으로 영상을 생성할 수 있게 됨
- manifold learning 등을 통해 latent vector로부터 고차원 이미지 생성이 가능
- `GAN`, `DCGAN` 부터 공부 시작
