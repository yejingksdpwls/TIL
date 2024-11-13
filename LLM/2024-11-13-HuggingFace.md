### [ 오늘 공부한 내용 ]
오늘은 huggingface_hub 라이브러리에서 발생한 버전 호환성 문제를 해결하려고 시도했습니다. 주된 문제는 sentence_transformers와 huggingface_hub 간의 호환성 문제로, cached_download와 snapshot_download 함수가 제대로 동작하지 않는 오류를 해결하기 위한 과정이 있었습니다.

&nbsp;
## [ 한국어 임베딩 실습 코드 ]

1. 라이브러리 불러오기
```python
from sentence_transformers import SentenceTransformer
import numpy as np
```
* `sentence_transformers`는 문장 임베딩을 계산하는 데 사용되는 라이브러리입니다.
* `numpy`는 배열 처리를 위한 라이브러리로, 벡터나 행렬 계산을 효율적으로 할 수 있도록 도와줍니다.

&nbsp;
2. 모델 로드
```python
model = SentenceTransformer('intfloat/multilingual-e5-large')
```
* `SentenceTransformer`는 문장을 임베딩 벡터로 변환하는 데 사용되는 모델을 로드하는 클래스입니다.
* `'intfloat/multilingual-e5-large'`는 사용할 사전 학습된 모델을 지정하는 부분입니다. 이 모델은 다양한 언어에서 임베딩을 생성할 수 있는 다국어 모델입니다.

&nbsp;
3. 문장 정의
```python
sentences = [
    "참새는 짹짹하고 웁니다.",
    "LangChain과 Faiss를 활용한 예시입니다.",
    "자연어 처리를 위한 임베딩 모델 사용법을 배워봅시다.",
    "유사한 문장을 검색하는 방법을 살펴보겠습니다.",
    "강좌를 수강하시는 수강생 여러분 감사합니다!"
]
```
* `sentences`는 임베딩을 계산하려는 문장들의 리스트입니다. 각 문장은 한국어와 영어가 섞여 있습니다.

&nbsp;
4. 문장 임베딩 생성
```python
embeddings = model.encode(sentences)
```
* `model.encode(sentences)`는 모델을 사용하여 각 문장을 임베딩 벡터로 변환합니다.
* 이 함수는 각 문장을 고차원 벡터로 변환한 결과를 `embeddings`에 저장합니다.
* `embeddings`는 각 문장에 대한 벡터 표현을 담고 있는 배열입니다.

&nbsp;
5. 결과 출력
```python
print(embeddings.shape)
```
* `embeddings.shape`는 `embeddings` 배열의 크기(차원 수)를 출력합니다.
* `embeddings`는 `(n, m)` 형태의 배열이 됩니다. 여기서 n은 문장의 개수(이 예제에서는 5개), m은 각 문장의 임베딩 벡터 차원 수입니다. 이 값은 모델에 따라 다르지만, `intfloat/multilingual-e5-large` 모델의 경우 벡터 차원이 1024일 수 있습니다.

&nbsp;
### [ 문제가 있었던 부분 ]
* `huggingface_hub`의 버전 관리와 호환성 문제로 인해 `cached_download`가 제공되지 않는 최신 버전에서는 `sentence_transformers`가 제대로 동작하지 않았습니다.

* 여러 번 버전 설치를 시도했으나, 여전히 함수가 누락된 문제를 해결하는 데 시간이 걸렸습니다.

&nbsp;
### [ 해결한 방법 ]

* huggingface_hub에서 cached_download와 snapshot_download 두 가지 함수가 제공되지만, 최신 버전에서는 cached_download 함수가 삭제되었으며, snapshot_download만 제공됩니다.

-> `pip install huggingface_hub==0.10.0` 을 통해 두 개의 함수 모두 실행할 수 있었습니다.

&nbsp;
### [ 배운 점 ]
* huggingface_hub의 버전 0.10.0에서 cached_download와 snapshot_download를 모두 지원하므로, 필요한 기능에 맞춰 버전을 관리해야 한다.

* 최신 버전에서는 snapshot_download로 대체 가능하며, 기존의 cached_download 기능을 지원하는 버전으로 다운그레이드해야 하는 경우가 있다.