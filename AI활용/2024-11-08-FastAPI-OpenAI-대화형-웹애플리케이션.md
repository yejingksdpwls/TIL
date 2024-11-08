오늘은 FastAPI를 이용해 OpenAI API와 연동한 대화형 웹 애플리케이션을 구현하는 방법을 공부했습니다. 사용자와의 대화 흐름을 관리하며, OpenAI API를 호출해 자연스러운 답변을 생성해주는 기능을 추가했습니다. 또한, 템플릿을 사용해 대화 내역을 웹 페이지에 시각적으로 표현하는 방법을 학습했습니다.

&nbsp;

## [ 대화형 웹 애플리케이션 구현 ]

1. 라이브러리 및 API 설정
```python
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
import os
```
* `fastapi`: FastAPI 라이브러리로 웹 애플리케이션의 라우팅과 요청 처리를 담당.
* `Jinja2Templates`: 템플릿을 사용해 HTML 파일을 렌더링.
* `StaticFiles`: 정적 파일을 서빙하기 위해 사용.
* `OpenAI`: OpenAI의 API 호출을 위해 사용.
* `dotenv`: `.env` 파일에서 환경 변수를 로드해 API 키 보안을 유지.

&nbsp;
2. API 키 설정
```python
load_dotenv()  # 현재 디렉터리의 .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
```
* `.env` 파일에 저장된 `OPENAI_API_KEY`를 로드해 OpenAI 클라이언트를 설정. 이를 통해 API 키를 코드에 직접 입력하지 않고 안전하게 관리할 수 있음.

&nbsp;
3. FastAPI 및 템플릿 설정
```python
app = FastAPI()
templates = Jinja2Templates(directory='app/templates')
app.mount('/static', StaticFiles(directory='app/static'), name='static')
```
* FastAPI 애플리케이션 객체 생성.
* `app/templates` 디렉터리에서 Jinja2 템플릿 파일을 불러옴.
* `app/static` 디렉터리의 정적 파일을 서빙하도록 설정.

&nbsp;
4. 초기 시스템 메시지 설정 및 대화 흐름 관리
```python
system_message = {
    'role': 'system',
    'content' : '너는 환영 인사를 하는 인공지능이야, 농담을 넣어 재미있게해줘'
}
messages = [system_message]
```
* 초기 시스템 메시지를 정의하여 AI가 대화를 시작할 때의 기본 역할을 설정.
* 시스템 메시지를 포함한 `messages 리스트`로 대화 내역을 관리.

&nbsp;
5. 채팅 페이지 렌더링 (GET 요청)
```python
@app.get('/', response_class=HTMLResponse)
async def get_chat_page(request: Request):
    conversation_history = [msg for msg in messages if msg['role'] != 'system']
    return templates.TemplateResponse('index.html', {'request': request, 'conversation_history': conversation_history})
```
* 사용자에게 대화 페이지를 렌더링하여 보여줌.
* `messages 리스트`에서 시스템 메시지를 제외한 대화 내역을 템플릿에 전달.

&nbsp;
6. 사용자 메시지 처리 및 OpenAI API 호출 (POST 요청)
```python
@app.post('/chat', response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    global messages
    messages.append({'role': 'user', 'content': user_input})

    completion = client.chat.completions.create(
        model='gpt-4',
        messages = messages
    )
    assistant_reply = completion.choices[0].message.content
    messages.append({'role': 'assistant', 'content': assistant_reply})

    conversation_history = [msg for msg in messages if msg['role']!='system']
    return templates.TemplateResponse('index.html', {
        'request': request,
        'conversation_history': conversation_history
    })
```
* 사용자의 메시지를 `messages 리스트`에 추가한 후 OpenAI API를 호출하여 응답을 받음.
* AI의 응답을 `messages`에 추가하여 대화가 이어지도록 관리.
* 대화 내역을 다시 템플릿에 전달하여 HTML로 렌더링.

&nbsp;
### [ 배운점 ]
* FastAPI와 OpenAI API를 연동하여 웹 애플리케이션에서 대화형 기능을 구현할 수 있음을 확인했습니다.

* .env 파일을 통해 API 키를 안전하게 관리하는 방법과, 사용자와 AI의 대화 흐름을 리스트로 관리하여 웹 페이지에서 표현하는 방법을 익혔습니다.

* FastAPI의 템플릿 기능과 정적 파일 서빙을 통해 사용자 친화적인 대화형 UI를 구성할 수 있었습니다.