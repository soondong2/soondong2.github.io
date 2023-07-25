---
title: "[GEE] Google Earth Engine Python API"
date: 2023-07-25

categories:
  - GIS
tags:
    - GIS
    - Google Earth Engine
---

## Python API 활용 방법

```python
import ee
from google.auth.transport.requests import AuthorizedSession

ee.Authenticate()  #  or !earthengine authenticate --auth_mode=gcloud
session = AuthorizedSession(ee.data.get_persistent_credentials())
```

```python
ee.Initialize()
```

```python
Map = geemap.Map()
Map
```

<img width="752" alt="image" src="https://github.com/soondong2/soondong2.github.io/assets/100760303/5300a437-b8a0-4b9b-ae39-e743e64850ca">

```python
# "" 안에 이미지 입력
image = ee.Image("")
print(image.getInfo())
```

```
{'bands': [{'crs': 'EPSG:32652',
            'crs_transform': [0.06415149669355084,
                              -0.5992946744683894,
                              273556.7841501166,
                              -0.5992946744683978,
                              -0.06415149669356095,
                              3841252.1478648423],
            'data_type': {'max': 255,
                          'min': 0,
                          'precision': 'int',
                          'type': 'PixelType'},
            'dimensions': [6640, 6639],
            'id': 'b1'}],
 'id': 'projects/mrv-poc-practice/assets/2023-07-06-01-02-32_UMBRA-05',
 'properties': {'system:asset_size': 47204960,
                'system:footprint': {'coordinates': [[126.49053822328602,
                                                      34.6477593487759],
                                                     [126.49053913086848,
                                                      34.647759400947876],
                                                     [126.5337944059228,
                                                      34.65248134371928],
                                                     [126.53379704464895,
                                                      34.65248386985626],
                                                     [126.53380009567597,
                                                      34.652486033507095],
...
                                                      34.6477593487759]],
                                     'type': 'LinearRing'}},
 'type': 'Image',
 'version': 1690204818489588}
```

```python
Map.addLayer(image)
```

<img width="747" alt="image" src="https://github.com/soondong2/soondong2.github.io/assets/100760303/fe538d73-9f8b-40ed-aed9-992f0b091f74">