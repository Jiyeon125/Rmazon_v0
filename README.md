# Rmazon 상품 분석 시스템

**Rmazon 상품 분석 시스템**은 온라인 상품 판매자가 데이터에 기반한 합리적인 의사결정을 내릴 수 있도록 지원하는 웹 기반 분석 솔루션입니다. 방대한 상품 데이터를 분석하여 **시장 내 경쟁 환경 분석, 소비자 요구사항 파악, 최적 가격 전략 수립** 등 상품 기획 및 판매 전략의 전 과정에 필요한 핵심 인사이트를 제공합니다.

노션 기술 보고서 링크 : https://www.notion.so/Rmazon-_-21b6d98822c68078a325c147b8c5a4da?source=copy_link


---

##  주요 기능 (Core Features)

1.  **유사 상품 탐색 및 AI 리뷰 분석:**
    *   **3요소 가중 유사도 분석:** 사용자가 입력한 상품 정보를 바탕으로 **상품 설명의 의미적 유사성(TF-IDF), 가격 근접성, 카테고리 핵심 키워드 일치도**를 종합적으로 분석하여 가장 유사한 상품 목록을 제공합니다. 이 분석은 상품의 핵심 정보가 담긴 `about_product` 필드를 기반으로 이루어져 높은 정확도를 보장합니다.
    *   **AI 기반 리뷰 요약:** 경쟁 상품들의 모든 리뷰(`review_title` + `review_content`)를 취합하여 **긍정/부정 그룹으로 분류**한 뒤, 각 그룹에서 가장 빈도가 높은 키워드를 추출합니다. 이를 통해 소비자의 구체적인 반응과 그 원인을 직관적으로 파악할 수 있습니다.

2.  **판매 지표 예측:**
    *   **회귀 모델 기반 예측:** 특정 카테고리와 가격을 설정하면, `Ridge` 회귀 모델이 해당 조건에서 기대할 수 있는 **예상 별점과 리뷰 수**를 예측합니다.
    *   **시장 내 위치 분석:** 예측된 지표가 해당 시장(카테고리) 내에서 상위 몇 %에 위치하는지 백분위수와 함께 시각적인 분포도 그래프를 제공하여, 객관적인 시장 포지셔닝 분석을 지원합니다.

---

## 기술 스택 (Tech Stack)

| 구분         | 기술                                                               |
| :----------- | :----------------------------------------------------------------- |
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS, shadcn/ui, Recharts       |
| **Backend**  | FastAPI, Python, Pandas, NumPy, Scikit-learn, VaderSentiment |
| **Deployment** | Vercel (Frontend), Local Server (Backend)                          |

---

## 시스템 아키텍처 (System Architecture)

본 시스템은 최신 웹 기술 스택을 기반으로 프론트엔드와 백엔드가 명확하게 분리된 구조로 설계되었습니다. 


-   **프론트엔드 (Frontend):** Next.js 기반의 동적 사용자 인터페이스입니다. 사용자의 입력을 받아 백엔드 API와 비동기적으로 통신하고, 반환된 데이터를 다양한 
차트와 테이블 형태로 시각화합니다.
-   **백엔드 (Backend):** FastAPI를 활용한 고성능 Python 서버입니다. 데이터 로딩 및 전처리, 머신러닝 모델 서빙, 핵심 분석 로직 수행 및 REST API 제공 역할을 
담당합니다.


아래 다이어그램들은 시스템의 전체 구조와 핵심 기능의 데이터 흐름을 보여줍니다.

### 1. 전체 시스템 구성도

이 다이어그램은 사용자의 요청이 프론트엔드와 백엔드를 거쳐 다시 사용자에게 시각화된 결과로 반환되기까지의 전체적인 흐름을 나타냅니다. 각 컴포넌트가 어떤 기술로 이루어져 있고 어떤 핵심 역할을 수행하는지 한눈에 파악할 수 있습니다.

```mermaid
graph TD;
    subgraph "사용자 (Client)";
        User("사용자") --> Browser("웹 브라우저");
    end;

    subgraph "프론트엔드 (Next.js @ :3000)";
        FE["<b>Next.js / React</b><br/>유사 상품 검색 UI<br/>판매 지표 예측 UI<br/>Recharts 시각화"];
    end;

    subgraph "백엔드 (FastAPI @ :8000)";
        API["<b>FastAPI Server</b><br/>API 엔드포인트 제공<br/>(/search-similarity, /predict)"];
        Engine["<b>핵심 분석 엔진</b><br/>Pandas 데이터 처리<br/>Scikit-learn 유사도/예측<br/>VaderSentiment 감성 분석"];
        Models["<b>머신러닝 모델</b><br/>(서버 시작 시 학습)<br/>TF-IDF Vectorizer<br/>Ridge 회귀 모델"];
    end;
    
    subgraph "데이터 저장소";
         Data["<B>데이터 파일</B><br/>cleaned_amazon_0519.csv"];
    end;

    Browser -- "분석 요청" --> FE;
    FE -- "API 요청 (Fetch)" --> API;
    API -- "분석 로직 실행" --> Engine;
    Engine -- "데이터 조회" --> Data;
    Engine -- "학습된 모델 활용" --> Models;
    Models -- "데이터로 모델 학습" --> Data;
    Engine -- "분석 결과 반환" --> API;
    API -- "응답 (JSON)" --> FE;
    FE -- "결과 시각화" --> Browser;
```

### 2. '유사 상품 탐색' 기능 데이터 흐름 (시퀀스 다이어그램)

이 다이어그램은 `유사 상품 탐색` 기능이 실행될 때, 백엔드 내부에서 각 모듈과 데이터가 어떤 순서로 상호작용하는지를 상세하게 보여줍니다. 사용자의 요청이 여러 분석 단계를 거쳐 최종 결과로 조합되는 과정을 이해할 수 있습니다.

```mermaid
sequenceDiagram
    participant Client as 사용자/프론트엔드
    participant API as FastAPI 엔드포인트<br/>(/search-similarity)
    participant Engine as 분석 로직
    participant TFIDF_Model as TF-IDF 모델<br/>(Vectorizer + Matrix)
    participant df_products as 상품 데이터<br/>(Pandas DataFrame)
    participant df_reviews as 리뷰 데이터<br/>(Pandas DataFrame)
    participant Vader_Model as 감성 분석기<br/>(VaderSentiment)

    Client->>API: 유사 상품 분석 요청 (설명, 가격, 카테고리)
    
    activate API
    API->>Engine: 분석 로직 실행
    activate Engine
    
    Engine->>df_products: 카테고리로 1차 상품 필터링
    df_products-->>Engine: 필터링된 상품 목록
    
    Engine->>TFIDF_Model: 입력된 설명으로 텍스트 유사도 계산
    TFIDF_Model-->>Engine: 코사인 유사도 점수 배열
    
    Engine->>Engine: 유사도 점수 결합 및 상위 10개 선정
    
    loop for each of top 10 products
        Engine->>df_reviews: Product ID로 관련 리뷰 조회
        df_reviews-->>Engine: 해당 상품의 모든 리뷰 텍스트
        Engine->>Vader_Model: 리뷰 텍스트 감성 분석 요청
        Vader_Model-->>Engine: 긍/부정 비율 및 핵심 키워드
    end
    
    Engine->>Engine: 최종 결과 데이터 조합 (JSON 생성)
    Engine-->>API: 분석 완료된 데이터 반환
    deactivate Engine
    
    API-->>Client: 최종 결과 응답 (JSON)
    deactivate API
```

---

## 설치 및 실행 방법 (Installation & Usage)

### 사전 요구사항

-   Node.js (v18.x 이상 권장)
-   Python (v3.8 이상 권장)
-   `pip` 및 `venv`

### 1. 프로젝트 클론 및 의존성 설치

```bash
# 1. 프로젝트 저장소를 클론합니다.
git clone <your-repository-url>
cd <project-directory>

# 2. 프론트엔드 의존성을 설치합니다. (루트 디렉토리에서 실행)
# 일부 shadcn/ui 관련 패키지 버전 충돌이 있을 수 있으므로, --legacy-peer-deps 옵션을 권장합니다.
npm install --legacy-peer-deps

# 3. 백엔드 가상 환경 생성 및 의존성 설치
# (Windows PowerShell에서는 '&&' 대신 각 명령어를 순차적으로 실행하세요.)
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 서버 실행

**두 개의 터미널**을 열고 각각 다음 명령어를 실행해야 합니다.

1.  **백엔드 서버 실행 (API 서버):**
    *   `backend` 디렉토리에서 실행합니다.

    ```bash
    # (venv가 활성화된 상태에서 실행)
    # 다른 장치에서 접속하려면 --host 0.0.0.0 옵션을 추가하세요.
    uvicorn main:app --reload --host 0.0.0.0
    ```
    *   서버가 정상적으로 실행되면 `http://127.0.0.1:8000` (또는 `http://[YOUR_IP]:8000`)에서 API 서버가 동작합니다.

2.  **프론트엔드 서버 실행 (웹 애플리케이션):**
    *   프로젝트의 루트 디렉토리에서 실행합니다.

    ```bash
    npm run dev
    ```
    *   서버가 정상적으로 실행되면 브라우저에서 `http://localhost:3000`으로 접속하여 애플리케이션을 사용할 수 있습니다.
