## [ AI 활용 ]
오늘은 **파이썬 패키지 및 가상환경 관리**와 **허깅 페이스의 활용법**을 공부했습니다.
### [ 파이썬 패키지 관리와 가상환경의 중요성 ]
* **패키지 버전 관리** : 패키지의 버전 차이에 따라 기능이 다를 수 있어 팀 프로젝트나 유지보수 시 특정 버전을 맞추는 것이 중요하다.

* **패키지 목록 확인** : pip list 명령어로 패키지 종류와 버전을 확인하고 공유할 수 있다.

* **가상환경 사용**: 각 프로젝트에 독립적인 환경을 구성해 패키지 충돌을 방지할 수 있다.

  * **conda 사용법**
    - 가상환경 생성: `conda create --name 환경이름`
    - 가상환경 활성화: `conda activate 환경이름`
    - 가상환경 비활성화: `conda deactivate`
  * **venv 사용법**
	
    - 가상환경 생성: `!python -m venv 환경이름`
	
    - 가상환경 활성화: `환경이름\\Scripts\\activate` 
    - 가상환경 비활성화: `deactivate`
 &nbsp;
 &nbsp;
### [ Hugging Face 이해하기 ]
* **정의** : NLP 중심의 다양한 AI 모델을 제공하는 플랫폼.

* **특징** : 강력한 Transformers 라이브러리, 모델 허브, 커뮤니티 중심

* **장점** : 사용이 쉬워 접근성이 높고, 다양한 모델을 무료로 제공하며, 커뮤니티가 활발하게 지원된다.

* **단점** : 리소스가 많이 필요하고 초기 설정이 복잡할 수 있으며, NLP 외 모델은 상대적으로 적다.
&nbsp;
 &nbsp;
 &nbsp;
### [ 코드 예제: 문장 생성하기 ]
```python
코드 복사
from transformers import GPT2Tokenizer, GPT2LMHeadModel
```
→ **GPT2Tokenizer**와 **GPT2LMHeadModel**을 불러와 문장을 분할하고 생성하는 데 사용
```python
코드 복사
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
```
→ **사전 학습 모델 'gpt2'**로 토큰화하고 모델을 준비

```python
코드 복사
text = "My name is"  # 입력 문장
encoded_input = tokenizer(text, return_tensors='pt') # 모델이 사용할 수 있는 토큰 형태로 변환

output = model.generate(encoded_input['input_ids'], max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_text
```
→ 주어진 입력 문장에 따라 언어 모델이 문장을 확장해 자동 생성. 결과는 사람이 읽을 수 있는 형태로 디코드 후 출력.
 &nbsp;
  &nbsp;
### [ 배운점 ]
* **효율적 가상환경 관리**는 프로젝트 간 패키지 충돌을 방지하며, 팀 작업 시 패키지 호환성 문제를 줄여준다.

* **Hugging Face의 NLP 모델**은 사용이 쉽고 다양하게 적용 가능하여, 복잡한 NLP 작업도 효율적으로 진행할 수 있음을 깨달았다.