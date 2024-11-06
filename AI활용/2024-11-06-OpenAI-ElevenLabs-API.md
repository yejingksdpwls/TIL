오늘은 OpenAI API를 이용한 챗봇 개발과 ElevenLabs API를 이용한 음성 생성 방법에 대해 공부했습니다. OpenAI API를 통해 변호사 역할을 수행하는 챗봇을 구현하고, ElevenLabs API를 활용해 텍스트를 음성으로 변환하는 실습을 진행했습니다.

### [ OpenAI API 이용한 간단 챗봇 코드 실습 ]

1. API 키 입력

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
from google.colab import files

files.upload()  # 파일 선택 창이 나타나면 .env 파일을 선택하여 업로드
load_dotenv()  # .env 파일 로드
api_key = os.getenv("API_KEY")  # 환경 변수 가져오기

client = OpenAI()
client.api_key = api_key
```

&nbsp;
2. 시스템 메시지 설정

```python
system_message = {'role': 'system', 'content': '너는 변호사야, 나에게 법률적인 상담을 해줘. 그리고 주의사항은 말하지마'}
```

→  챗봇의 역할을 정의하는 시스템 메시지입니다. 이 메시지를 통해 챗봇이 "법률 상담을 제공하는 변호사" 역할을 수행하도록 설정합니다.

&nbsp;
3. 반복적인 사용자 입력 처리

```python
messages = [system_message]

while True:
    user_input = input('사용자 전달 : ')
    if user_input == 'exit':
        print('즐거운 대화였습니다! 감사합니다!')
        break

```

→ `messages` 리스트에 `system_message`를 추가하여 대화를 시작합니다. 이 리스트는 대화의 문맥을 유지하는 데 사용되며, 모든 사용자 메시지와 챗봇의 응답이 이 리스트에 저장됩니다.

→ 사용자가 "exit"라고 입력할 때까지 대화를 반복합니다. "exit"를 입력하면 종료 메시지를 출력하고 프로그램을 종료합니다.

&nbsp;
4. 사용자 메시지 추가 및 챗봇 응답 생성

```python
messages.append({'role':'user', 'content': user_input})
completion = client.chat.completions.create(
    model='gpt-4o'
    messages = messages
)
```

→  사용자의 메시지를 `messages` 리스트에 추가합니다.

→ `client.chat.completions.create()` 함수를 사용해 OpenAI 모델 `gpt-4o`를 호출하여 챗봇의 응답을 생성합니다.

&nbsp;
5. 응답 출력 및 리스트에 추가

```python
reply = completion.choices[0].message.content
print('대답: ' + reply)
messages.append({'role':'assistant', 'content':reply})
```

→  생성된 응답을 `reply` 변수에 저장하고 화면에 출력합니다.

→ 챗봇의 응답도 `messages` 리스트에 추가하여 대화의 문맥을 유지합니다.

&nbsp;
## [ ELEVENLABS API 이용한 음성 생성 실습 코드 ]

&nbsp;
1. 필요한 모듈 임포트

```python
import os
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
```

- `os`: 환경 변수를 처리하기 위해 사용.
- `requests`: API 요청을 보내기 위해 사용.
- `pydub`: 오디오 데이터를 처리하고, 변환된 오디오를 재생하기 위해 사용.
- `io`: 바이트 데이터를 다루기 위한 모듈로, 메모리에서 직접 오디오를 다룹니다.

&nbsp;
2. 환경 변수 설정

```python
output_filename = "output_audio.mp3"
```

- `output_filename` 변수는 최종 음성 파일이 저장될 위치와 파일명을 지정합니다.

&nbsp;
3. API URL 및 헤더 설정

```python
url = "model url"
headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}
```

- `url`은 Eleven Labs의 텍스트-음성 변환 API 엔드포인트입니다.
- `headers`는 API 호출에 필요한 키와 JSON 형식을 지정합니다. 여기서 `api_key`는 API 액세스 키로 `.env` 파일에 저장하여 보안을 유지하는 것이 좋습니다.

&nbsp;
4. 텍스트 입력

```python
text = input("텍스트를 입력하세요: ")
```

- 입력된 텍스트가 음성으로 변환됩니다.

&nbsp;
5. 모델 및 음성 설정

```python
data = {
    "text": text,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.3,
        "similarity_boost": 1,
        "style": 1,
        "use_speaker_boost": True
    }
}
```

- `data`는 API에 전달할 텍스트와 모델 및 음성 설정을 포함합니다. `model_id`는 텍스트를 변환할 모델을 지정하며, `voice_settings`는 음성의 안정성, 유사도, 스타일 등을 설정합니다.

&nbsp;
6. API 요청

```python
response = requests.post(url, json=data, headers=headers, stream=True)
```

- `requests.post`로 API에 텍스트 변환 요청을 보냅니다. 성공하면 서버가 생성한 오디오 데이터를 반환합니다.

&nbsp;
7. 오디오 파일 생성 및 재생

```python
if response.status_code == 200:
    audio_content = b""
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            audio_content += chunk
    segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
    segment.export(output_filename, format="mp3")
    print(f"Success! Wrote audio to {output_filename}")
    play(segment)
else:
    print(f"Failed to save file: {response.status_code}")
```

- 응답 상태가 성공(`200 OK`)이면 API가 반환한 오디오 데이터를 반복문으로 수신하고, 오디오 데이터를 `AudioSegment` 객체로 변환합니다.
- 변환한 오디오 데이터를 `output_filename` 파일로 저장하고, `play` 함수를 통해 오디오를 재생합니다.

&nbsp;
### [ 배운 점 ]

* OpenAI API를 사용하여 챗봇의 대화 시스템을 구현하는 과정에서, 대화의 흐름을 관리하기 위해 시스템 메시지와 메시지 리스트를 적절히 활용하는 방법을 배웠습니다.

* ElevenLabs API를 통해 텍스트를 음성으로 변환하는 방법을 실습하면서, API 요청과 음성 설정을 통해 다양한 스타일과 톤의 음성을 만들 수 있음을 알게 되었습니다. 또한, 음성을 파일로 저장하고 재생하는 방법도 익혔습니다.