from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import os

# --- Pydantic 모델 정의 ---
# 요청 본문의 데이터 구조를 정의합니다.
class PredictionRequest(BaseModel):
    price: float
    review_count: int
    category: str

# 응답 본문의 데이터 구조를 정의합니다.
class PredictionResponse(BaseModel):
    predicted_star: float

# --- FastAPI 애플리케이션 설정 ---
app = FastAPI()

# CORS 미들웨어 추가: Next.js 앱(http://localhost:3000)에서의 요청을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 머신러닝 모델 및 데이터 전처리기 ---
# 모델과 전처리 파이프라인을 저장할 변수입니다.
# 서버가 실행되는 동안 메모리에 유지됩니다.
ml_pipe = None

# --- 서버 시작 시 실행될 로직 ---
@app.on_event("startup")
def load_model_and_data():
    global ml_pipe
    
    # 1. 데이터 로드
    csv_path = os.path.join("data", "cleaned_amazon_0519.csv")
    df = pd.read_csv(csv_path)

    # 2. 올바른 컬럼 이름으로 필요한 컬럼만 선택하고 결측치 처리
    df = df[['discounted_price', 'rating_count', 'category_cleaned', 'rating']].dropna()
    df = df[df['rating_count'] > 0] # 리뷰 수가 0인 데이터는 제외

    # 3. 특성(X)과 타겟(y) 분리
    X = df[['discounted_price', 'rating_count', 'category_cleaned']]
    y = df['rating']

    # 4. 데이터 전처리 파이프라인 구축
    # 수치형 특성은 StandardScaler로, 범주형 특성은 OneHotEncoder로 변환합니다.
    numeric_features = ['discounted_price', 'rating_count']
    categorical_features = ['category_cleaned']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 5. 릿지 회귀 모델을 포함한 전체 파이프라인 생성
    ml_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', Ridge(alpha=1.0))])

    # 6. 모델 학습
    ml_pipe.fit(X, y)
    print("✅ Model training complete!")
    print(f"📈 Available categories: {X['category_cleaned'].unique().tolist()}")


# --- API 엔드포인트 정의 ---
@app.get("/")
def read_root():
    return {"message": "Rmazon-predictor API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_star_rating(request: PredictionRequest):
    # 1. 요청 데이터를 DataFrame으로 변환 (컬럼 이름 매칭)
    input_data_dict = {
        'discounted_price': request.price,
        'rating_count': request.review_count,
        'category_cleaned': request.category
    }
    input_data = pd.DataFrame([input_data_dict])
    
    # 2. 학습된 파이프라인을 사용해 예측 수행
    predicted_star = ml_pipe.predict(input_data)[0]
    
    # 3. 예측 결과를 소수점 2자리까지 반올림하여 반환
    return {"predicted_star": round(predicted_star, 2)} 