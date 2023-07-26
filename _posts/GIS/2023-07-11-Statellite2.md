---
title: "[GIS] 인공위성 데이터 이해2"
date: 2023-07-11

categories:
  - GIS
tags:
    - GIS
---

## 인공위성 구성 요소

위성 영상의 파라미터 중 가장 중요하다고 볼 수 있는 `resolusion`은 아래의 네 가지로 정의된다.

## 공간 해상도(Spatial Resolution)

**얼마나 작고 가까이 있는 물체까지 자세하게 관측할 수 있는가**를 나타내는 지표이다. 센서의 특성과 위성에서 물체까지 거리에 따라 결정된다.

공간 해상도는 `GSD`(Ground Sample Distance)와 `GRD`(Gound Resolved Distance)로 구분해서 설명할 수 있다. GSD는 영상에서 한 픽셀에 해당하는 실제 거리를 뜻하고, GRD는 두 물체를 구분할 수 있는 최소한의 거리를 뜻한다.

<img width="491" alt="image" src="https://github.com/soondong2/soondong2.github.io/assets/100760303/ade9a873-0856-4709-b8b2-460263879242">

## 분광 해상도(Spectral Resolution)

**관측하는 전자기파 파장의 범위**를 뜻한다. 위성 영상의 Band는 각각의 관측 파장대를 의미하며, 관측하는 물체의 해당 파장대에 대한 특성을 보여주게 된다. 위성에 따라 밴드의 구성이 달라지게 된다. 그 중 1번 밴드가 Panchromatic(PAN) 밴드인 경우가 종종 있다. 대신 특정 파장대의 특성을 나타내지 못하기 때문에 흑백으로 나타난다. PAN 영상을 이용해 RGB 밴드 영상에 샤프닝을 적용해 PS-RGB 형태로 영상이 제공되는 경우가 많다. 연구 목적에 따라 적절한 밴드를 선정하고 조합하는 과정이 필요하다.

<img width="489" alt="image" src="https://github.com/soondong2/soondong2.github.io/assets/100760303/9decdb71-310b-4253-85bb-6900b753b4db">

## 시간 해상도(Temporal Resolution)

**동일한 지역을 얼마나 자주 촬영 하는가**를 의미한다. 위성이 높은 고도에서 지구의 자전과 같은 속도로 회전하며 동일한 지역만 지속적으로 관측하는 정지궤도 위성의 경우 분 단위 관측이 가능하다. 반면 낮은 고도에서 회전하며 전지구를 관측하는 일반적인 극궤도 위성이라면 위성이 지구를 한 바퀴 돌아 다시 동일 위치에 도달할 때까지 기다려야 한다.

<img width="489" alt="image" src="https://github.com/soondong2/soondong2.github.io/assets/100760303/a2cffaa1-d726-45dd-8159-1e9de0cfcfae">

## 방사 해상도(Radiometric Resolution)

에너지 강도에 대해 얼마나 민감한가를 의미하며 이는 영상의 `bit` 수로 표현된다. 일반 영상은 0~255의 8-bit 영상이지만, 위성 영상의 경우 16-bit, 14-bit 등 다양하게 나타난다.

또한 모든 조건을 만족하기에는 물리적인 한계가 존재하므로 공간 해상도, 분광 해상도, 시간 해상도는 `trade-off` 관게를 가질 수 밖에 없다. 