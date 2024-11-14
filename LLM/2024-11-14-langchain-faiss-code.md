### [ 오늘 공부한 내용 ]
오늘은 LangChain과 FAISS를 사용하여 벡터 기반 유사성 검색과 질의응답 시스템을 구현하는 방법을 학습했습니다. LangChain의 프롬프트 템플릿, 벡터 임베딩, FAISS 인덱스, RAG(Retrieval-Augmented Generation) 체인 등을 사용하여 유사 문서 검색과 질의응답 체인을 구성하는 방법을 익혔습니다.

&nbsp;
## [ LangChain 시스템 실습 코드 ]

1. OpenAI API 키 설정

```python
import os
from getpass import getpass

os.environ['OPENAI_API_KEY'] = getpass('OpenAI API key 입력: ')
```
* `os.environ`을 통해 OpenAI API 키를 환경 변수에 저장합니다. getpass는 터미널에서 보이지 않게 키 입력을 받습니다.

&nbsp;
2. 모델 초기화 및 메시지 생성

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model='gpt-4')
response = model.invoke([HumanMessage(content='안녕하세요, 무엇을 도와드릴까요?')])
```
* GPT-4 모델을 초기화하고 `HumanMessage`로 대화형 메시지를 생성하여 모델에 전달합니다.

&nbsp;
3. 프롬프트 템플릿 생성

```python
from langchain_core.prompts import ChatPromptTemplate

# 시스템 메시지 설정
system_template = "Translate the following sentence from English to {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])
```
* 메시지 템플릿을 설정하여 사용자가 지정한 `language`로 주어진 텍스트를 번역하도록 하는 시스템 메시지를 구성합니다.

&nbsp;
4. 체인 실행

```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

chain = prompt_template | model | parser
response = chain.invoke({"language": "Korean", "text": "Where is the library?"})
```
* 프롬프트, 모델, 출력 파서를 연결한 체인을 만들고 실행하여 응답을 반환합니다.

&nbsp;
5. OpenAI 임베딩 초기화

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```
* OpenAI 임베딩 모델을 초기화합니다. 이 임베딩을 문서 검색에 사용할 벡터로 변환합니다.

&nbsp;
6. FAISS 인덱스 생성

```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
```
* FAISS 라이브러리를 이용하여 벡터 인덱스를 생성하고, 문서를 임시 저장하기 위한 `InMemoryDocstore`와 함께 설정합니다.

&nbsp;
7. 문서 추가

```python
from langchain_core.documents import Document
from uuid import uuid4

documents = [
    Document(page_content="LangChain을 사용해 프로젝트를 구축하고 있습니다!", metadata={"source": "tweet"}),
    Document(page_content="내일 날씨는 맑고 따뜻할 예정입니다.", metadata={"source": "news"}),
    # 기타 문서들 추가
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```
다양한 문서를 벡터 저장소에 추가하여 각 문서에 고유 ID를 할당합니다.

&nbsp;
8. 유사성 검색

```python
results = vector_store.similarity_search("내일 날씨는 어떨까요?", k=2, filter={"source": "news"})
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```
* 입력 쿼리와 유사한 문서를 `source` 필터를 사용해 검색합니다. k=2는 가장 유사한 두 개의 문서를 반환하도록 설정합니다.

&nbsp;
9. 질의응답 체인 (RAG) 설정

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\n\nQuestion: {question}")
])

rag_chain_debug = {
    "context": retriever,
    "question": DebugPassThrough()
} | DebugPassThrough() | ContextToText() | contextual_prompt | model
```
* 사용자의 질문에 대한 답을 주어진 문맥(context)만 사용해 제공하는 RAG 체인을 구성합니다. `DebugPassThrough` 클래스는 중간 출력을 확인하는 데 사용됩니다.

&nbsp;
10. 인덱스 저장 및 로드

```python
vector_store.save_local("faiss_index")
new_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```
* FAISS 인덱스를 로컬 파일로 저장하고, 필요한 경우 다시 로드할 수 있습니다.

&nbsp;
11. 벡터 저장소 병합

```python
db1 = FAISS.from_texts(["문서 1 내용"], embeddings)
db2 = FAISS.from_texts(["문서 2 내용"], embeddings)
db1.merge_from(db2)
```
* 두 개의 벡터 저장소를 병합하여 데이터를 통합합니다.

&nbsp;
## [ 배운 점 ]
* LangChain을 활용해 GPT 모델과 벡터 임베딩을 쉽게 연결하고, 검색과 질의응답 시스템을 구성하는 방법을 익혔습니다.

* FAISS 인덱스와 RAG 체인을 통해 문서의 유사성을 기반으로 검색하고 그 결과를 활용해 답변을 생성하는 과정에 대해 이해할 수 있었습니다.

* LangChain과 FAISS를 활용한 프로젝트가 다양한 텍스트 기반 애플리케이션에 적용 가능하다는 가능성을 보았습니다.
