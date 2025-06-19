from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import os
import shutil

# --- Pydantic 모델 정의 ---
# 요청 본문의 데이터 구조를 정의합니다.
class PredictionRequest(BaseModel):
    price: float
    review_count: int
    category: str

# 응답 본문의 데이터 구조를 정의합니다.
class PredictionResponse(BaseModel):
    predicted_star: float

class SimilarityRequest(BaseModel):
    product_id: str

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

# --- 전역 변수: 모델, 데이터, 전처리기 ---
ml_pipe = None
tfidf_vectorizer = None
tfidf_matrix = None
df_products = pd.DataFrame()
DATA_FILE_PATH = os.path.join("backend", "data", "cleaned_amazon_0519.csv")

# --- 핵심 로직: 데이터 로딩 및 모델 학습 ---
def load_data_and_train_models():
    global ml_pipe, tfidf_vectorizer, tfidf_matrix, df_products
    
    if not os.path.exists(DATA_FILE_PATH):
        print(f"⚠️ 데이터 파일이 존재하지 않습니다: {DATA_FILE_PATH}")
        df_products = pd.DataFrame()
        return

    # 1. 데이터 로드
    df = pd.read_csv(DATA_FILE_PATH)
    
    # 2. 필수 컬럼 확인
    required_columns = ['product_id', 'product_name', 'review_title', 'review_content', 'discounted_price', 'rating_count', 'category_cleaned', 'rating']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")

    df = df[required_columns].dropna()
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
    df = df[df['rating_count'] > 0]
    df.reset_index(drop=True, inplace=True)
    df_products = df.copy()

    # 3. 릿지 회귀 모델 학습 (별점 예측용)
    X_ridge = df[['discounted_price', 'rating_count', 'category_cleaned']]
    y_ridge = df['rating']
    numeric_features = ['discounted_price', 'rating_count']
    categorical_features = ['category_cleaned']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    ml_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
    ml_pipe.fit(X_ridge, y_ridge)
    print("✅ Ridge Regression model training complete!")

    # 4. TF-IDF 모델 학습 (유사도 분석용)
    df_products['combined_text'] = df_products['review_title'].fillna('') + ' ' + df_products['review_content'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_products['combined_text'])
    print("✅ TF-IDF model training complete!")
    print(f"📈 Total {len(df_products)} products loaded and models trained.")


# --- 서버 시작 시 실행될 로직 ---
@app.on_event("startup")
def startup_event():
    try:
        load_data_and_train_models()
    except Exception as e:
        print(f"🚨 서버 시작 중 오류 발생: {e}")


# --- API 엔드포인트 정의 ---
@app.get("/")
def read_root():
    return {"message": "Rmazon predictor and similarity API is running!"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # 임시 파일로 저장
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 새 파일 검증
    try:
        df_new = pd.read_csv(temp_file_path)
        required_columns = ['product_id', 'product_name', 'review_title', 'review_content', 'discounted_price', 'rating_count', 'category_cleaned', 'rating']
        missing = [col for col in required_columns if col not in df_new.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"필수 컬럼 누락: {', '.join(missing)}")
        
        # 검증 통과 시, 기존 파일 덮어쓰기
        shutil.move(temp_file_path, DATA_FILE_PATH)
        
        # 데이터와 모델 다시 로드
        load_data_and_train_models()
        
        return {"message": "파일이 성공적으로 업로드 및 처리되었습니다.", "rows": len(df_new)}
    
    except Exception as e:
        os.remove(temp_file_path) # 실패 시 임시 파일 제거
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {e}")
    finally:
        await file.close()

@app.get("/products")
def get_products():
    if df_products.empty:
        return []
    # product_id와 product_name만 프론트엔드로 전송
    return df_products[['product_id', 'product_name']].to_dict('records')

@app.post("/search-similarity")
def search_similarity(request: SimilarityRequest):
    if df_products.empty or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="서버 데이터가 준비되지 않았습니다.")

    try:
        target_index = df_products.index[df_products['product_id'] == request.product_id].tolist()[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="해당 상품 ID를 찾을 수 없습니다.")

    # 유사도 계산
    cosine_similarities = cosine_similarity(tfidf_matrix[target_index], tfidf_matrix).flatten()
    
    # 상위 3개 (자기 자신 제외)
    similar_indices = cosine_similarities.argsort()[-4:-1][::-1]
    
    similar_products = df_products.iloc[similar_indices].copy()
    similar_products['similarity'] = cosine_similarities[similar_indices]
    
    return similar_products.to_dict('records')

@app.post("/predict", response_model=PredictionResponse)
def predict_star_rating(request: PredictionRequest):
    if ml_pipe is None:
        raise HTTPException(status_code=503, detail="예측 모델이 준비되지 않았습니다.")
        
    input_data_dict = {
        'discounted_price': request.price,
        'rating_count': request.review_count,
        'category_cleaned': request.category
    }
    input_data = pd.DataFrame([input_data_dict])
    
    predicted_star = ml_pipe.predict(input_data)[0]
    
    return {"predicted_star": round(predicted_star, 2)} 