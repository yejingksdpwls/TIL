오늘은 YOLO를 활용한 이미지 객체 탐지와 FastAI를 이용한 이미지 분류 모델을 학습하는 방법을 공부했습니다. YOLO를 통해 객체의 위치와 종류를 실시간으로 예측하는 방법을 배우고, FastAI에서 사전학습된 모델을 불러와 파인튜닝을 수행하여 이미지 분류를 구현해보았습니다.

&nbsp;
## [ YOLO : 이미지 객체 탐지 ]

- 모델 정의
    - 이미지에서 객체의 *위치* 와 *종류* 를 동시에 예측하는 강력한 딥러닝 모델
    - 한 번의 신경망 전파만으로 객체 탐지하므로 실시간 처리가 가능할 정도로 매우 빠름

- 특징
    - **속도** : 전체 이미지를 한 번에 처리해 빠르게 객체 탐지할 수 있음
    - **정확도** : 여러 객체가 존재하는 복잡한 이미지에서도 높은 정확도를 보임
    - **다양한 크기** : 다양한 크기의 이미지,객체 처리 가능
    

- 작동 원리
    - 하나의 이미지가 들어왔을 때 그리드로 분할 함
    - 바운딩 박스 (탐지된 객체를 둘러싸는 직사각형) 예측
        - 중심 좌표
        - 너비, 높이
        - 신뢰도 점수 (해당 바운딩 박스가 객체를 포함할 가능성 나타냄) : 객체가 존재할 확률 * 바운딩 박스 정확도
    - 클래스 확률 예측 (객체를 포함한다는 가정 하에, 해당 객체가 특정 클래스에 포함될 확률)
    - 최종 예측 (이게 뭘 의미하는 걸까?)

&nbsp;
### [ YOLO 모델 코드 실습 ]

1. 라이브러리 임포트

```python
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
```

- `ultralytics`  : YOLOv8 모델을 위한 라이브러리
- `OpenCV` : 이미지 처리와 관련된 다양한 기능을 제공
- `matplotlib.pyplot` : 이미지 시각화를 위해 사용

&nbsp;
2. YOLOv8 모델 로드

```python

model = YOLO('yolov8n.pt')
```

- YOLOv8 모델 중 가장 가벼운 YOLOv8n(‘nano’ 버전) 모델을 로드합니다.

&nbsp;
3. 이미지 파일 경로 지정

```python
image_path = 'cat.jpeg'
```

&nbsp;
4. 객체 탐지 수행 및 결과 가져오기

```python
results = model(image_path)
result = results[0]
img_with_boxes = result.plot()
```

- `results[0]` : 첫 번째 탐지 결과입니다.
- `result.plot()` : 탐지된 객체의 바운딩 박스를 포함한 이미지를 반환합니다.

&nbsp;
5. 이미지 시각화

```python
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

- `cv2.cvtColor()`를 통해 OpenCV의 BGR 이미지를 Matplotlib에서 사용하는 RGB 이미지로 변환합니다.

&nbsp;
## [ FastAI : 사전학습된 모델 사용해보기 ]

1. 라이브러리 및 데이터셋 로드

```python
from fastai.vision.all import *

# 데이터셋 로드
path = untar_data(URLs.PETS)  # PETS 데이터셋 다운로드 및 압축 해제
path_imgs = path/'images'
```

- `fastai.vision.all`에서 필요한 모든 모듈을 불러옵니다.
- `untar_data` 함수를 사용하여 URL로부터 PETS 데이터셋을 다운로드하고, 압축을 해제한 후 경로를 `path`에 저장합니다.

&nbsp;
2. 이미지 라벨링 함수 정의

```python
def is_cat(x): return x[0].isupper()
```

- 파일명에서 첫 글자가 대문자이면 고양이, 소문자이면 개로 분류하는 간단한 라벨링 함수를 정의합니다.

&nbsp;
3. 데이터 블록 정의

```python
dls = ImageDataLoaders.from_name_func(
    path_imgs, get_image_files(path_imgs), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
```

- `ImageDataLoaders.from_name_func`를 사용하여 이미지 데이터 로더(`dls`)를 정의합니다.
- 이미지 파일을 불러오고, 80%를 학습, 20%를 검증 데이터로 나눕니다.
- 이미지를 `224x224`로 크기 조정합니다(`Resize(224)`).

&nbsp;
4. 사전 학습된 모델 로드 및 학습

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
```

- `cnn_learner`를 사용해 사전 학습된 `ResNet34` 모델을 불러와 학습기를 생성합니다.
- 모델의 평가 지표로 `error_rate`(오류율)를 설정합니다.

&nbsp;
5. 모델 학습, 평가 및 결과 확인

```python
learn.fine_tune(3)

learn.show_results()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

- 사전 학습된 모델을 3 에포크 동안 파인튜닝합니다.
- `ClassificationInterpretation`을 통해 혼동 행렬을 생성하여 각 클래스별 예측 성능을 시각화합니다.

&nbsp;
6. 새로운 이미지에 대한 예측 및 결과 출력

```python
img = PILImage.create('path_your_image.jpg')
pred, _, probs = learn.predict(img)

print(f'Prediction: {pred}, Probability: {probs.max():.4f}')
img.show()
```

- `PILImage.create`를 통해 새로운 이미지를 불러와 모델에 예측을 요청합니다.
- 예측 결과 `pred`, 확률 `probs`를 반환받습니다.
- 최종 예측 클래스와 그에 대한 확률을 출력하며 이미지를 함께 표시합니다.


&nbsp;
### [ 배운 점 ]
* YOLO는 객체 탐지에 최적화된 모델로, 실시간 처리가 가능해 다양한 응용 분야에 적합함을 확인할 수 있었습니다.

* FastAI의 간결한 API를 통해 사전학습된 모델을 쉽게 활용하여 이미지 분류 성능을 높일 수 있었습니다.

* 객체 탐지와 이미지 분류에 적합한 다양한 모델의 특징을 비교하며 실습을 통해 모델을 더 깊이 이해할 수 있었습니다.