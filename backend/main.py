from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
import os
import shutil
from typing import List, Optional, Dict

# --- Pydantic 모델 정의 ---
# 요청 본문의 데이터 구조를 정의합니다.
class Keyword(BaseModel):
    word: str
    count: int

class PredictionRequest(BaseModel):
    price: float
    review_count: int
    category: str

# 응답 본문의 데이터 구조를 정의합니다.
class PredictionResponse(BaseModel):
    predicted_star: float

class SimilarityRequest(BaseModel):
    description: str
    price: float
    discount_percentage: float
    category: str

class Product(BaseModel):
    product_id: str
    product_name: str

class ReviewAnalysis(BaseModel):
    overall_sentiment: str
    sentiment_distribution: Dict[str, int]
    top_keywords: List[Keyword]
    negative_concerns: List[str]
    summary: str
    review_count: int

class SimilarityResult(BaseModel):
    product_id: str
    product_name: str
    similarity: float
    discounted_price: float
    rating: float
    rating_count: int
    review_analysis: ReviewAnalysis

class PriceRangeResponse(BaseModel):
    min_price: float
    max_price: float

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
DATA_FILE_PATH = os.path.join("data", "cleaned_amazon_0519.csv")

# --- 핵심 로직: 데이터 로딩 및 모델 학습 ---
def load_data_and_train_models():
    global ml_pipe, tfidf_vectorizer, tfidf_matrix, df_products
    
    if not os.path.exists(DATA_FILE_PATH):
        print(f"⚠️ 데이터 파일이 존재하지 않습니다: {DATA_FILE_PATH}")
        df_products = pd.DataFrame()
        return

    df = pd.read_csv(DATA_FILE_PATH)
    
    required_columns = ['product_id', 'product_name', 'review_title', 'review_content', 'discounted_price', 'rating_count', 'category_cleaned', 'rating']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")

    # 🚨 데이터 클리닝 및 전처리 로직 개선
    df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)
    
    # 텍스트 컬럼의 NaN 값을 빈 문자열로 대체 (FutureWarning 수정)
    df['review_title'] = df['review_title'].fillna('')
    df['review_content'] = df['review_content'].fillna('')

    # 숫자형 컬럼 처리
    df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # 모델 학습에 필수적인 컬럼에 NaN이 있으면 해당 행 제거
    df.dropna(subset=['discounted_price', 'rating_count', 'rating', 'category_cleaned'], inplace=True)

    # rating_count가 0 이하인 데이터는 예측에 의미가 없으므로 제외
    df = df[df['rating_count'] > 0].copy()
    
    df.reset_index(drop=True, inplace=True)
    
    # TF-IDF용 텍스트 합치기 (클리닝 이후에 수행)
    df['combined_text'] = df['review_title'] + ' ' + df['review_content']

    df_products = df.copy()

    if df_products.empty:
        print("⚠️ 처리할 데이터가 없습니다. 모델 학습을 건너뜁니다.")
        return

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
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_products['combined_text'])
    print("✅ TF-IDF model training complete!")
    print(f"📈 Total {len(df_products)} unique products loaded and models trained.")


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

@app.get("/categories", response_model=List[str])
def get_categories():
    """사용 가능한 모든 카테고리 목록을 반환합니다."""
    if df_products.empty:
        return []
    return sorted(df_products['category_cleaned'].unique().tolist())

@app.get("/products", response_model=List[Product])
def get_products(category: Optional[str] = Query(None)):
    """
    상품 목록을 반환합니다.
    - category 쿼리 파라미터가 있으면 해당 카테고리의 상품만 필터링합니다.
    """
    if df_products.empty:
        return []
    
    if category:
        filtered_df = df_products[df_products['category_cleaned'] == category]
        return filtered_df[['product_id', 'product_name']].to_dict('records')
    
    # 카테고리가 없으면 전체 목록 반환 (관리용 또는 다른 용도로 유지)
    return df_products[['product_id', 'product_name']].to_dict('records')

# --- 고급 리뷰 분석 로직 (서버 사이드로 이동) ---
def advanced_review_analysis(reviews: List[str]) -> Dict:
    # 이 부분은 이전에 프론트엔드에 있던 로직을 가져온 것입니다.
    # 실제로는 더 정교한 NLP 라이브러리(spaCy, NLTK 등)를 사용해야 하지만,
    # 기존 기능 복원을 위해 동일한 로직을 사용합니다.
    
    # ... (여기에 감성분석, 키워드 추출 등 기존 로직 구현) ...
    # 간단한 구현 예시:
    positive_words = ['good', 'great', 'excellent', 'love', 'best']
    negative_words = ['bad', 'poor', 'terrible', 'hate', 'worst']
    
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    all_words = []
    
    for review in reviews:
        review_lower = review.lower()
        pos_count = sum(1 for word in positive_words if word in review_lower)
        neg_count = sum(1 for word in negative_words if word in review_lower)
        
        if pos_count > neg_count:
            sentiments['positive'] += 1
        elif neg_count > pos_count:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
        
        all_words.extend(review_lower.split())

    # 전체 감성
    overall = max(sentiments, key=sentiments.get)

    # 키워드 (간단한 빈도수 기반)
    from collections import Counter
    # 불용어 리스트 확장
    stop_words_list = list(text.ENGLISH_STOP_WORDS) + ['product', 'amazon', 'use', 'get', 'it', 'i']
    
    keywords_with_counts = [
        (word, count) for word, count in Counter(all_words).most_common(20) 
        if word.isalpha() and len(word) > 2 and word not in stop_words_list
    ]

    return {
        "overall_sentiment": overall,
        "sentiment_distribution": sentiments,
        "top_keywords": [{"word": w, "count": c} for w, c in keywords_with_counts[:5]], # 상위 5개만 선택
        "negative_concerns": [r for r in reviews if any(w in r.lower() for w in negative_words)][:2],
        "summary": f"전체적으로 {overall}적인 평가가 많습니다. 주요 키워드는 {', '.join([k[0] for k in keywords_with_counts[:5]])} 등입니다.",
        "review_count": len(reviews)
    }

def calculate_price_similarity(price1: float, price2: float) -> float:
   if price1 == 0 or price2 == 0: return 0
   diff = abs(price1 - price2)
   avg = (price1 + price2) / 2
   return max(0, 1 - diff / avg)

@app.get("/category-price-range", response_model=PriceRangeResponse)
def get_category_price_range(category: str = Query(...)):
    """특정 카테고리의 최소 및 최대 가격을 반환합니다."""
    if df_products.empty:
        raise HTTPException(status_code=503, detail="서버 데이터가 준비되지 않았습니다.")
    
    filtered_df = df_products[df_products['category_cleaned'] == category]
    
    if filtered_df.empty:
        # 해당 카테고리에 상품이 없으면 전체 데이터의 가격 범위를 기본값으로 제공
        min_price = df_products['discounted_price'].min()
        max_price = df_products['discounted_price'].max()
    else:
        min_price = filtered_df['discounted_price'].min()
        max_price = filtered_df['discounted_price'].max()

    return {"min_price": min_price, "max_price": max_price}

@app.post("/search-similarity", response_model=List[SimilarityResult])
def search_similarity(request: SimilarityRequest):
    if df_products.empty or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="서버 데이터가 준비되지 않았습니다.")

    # 1. 텍스트 유사도 계산
    input_vec = tfidf_vectorizer.transform([request.description])
    text_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # 2. 가격 및 할인율 유사도 계산
    request_discounted_price = request.price * (1 - request.discount_percentage / 100)
    price_similarities = df_products['discounted_price'].apply(lambda x: calculate_price_similarity(request_discounted_price, x))
    discount_similarities = df_products['discount_percentage'].apply(lambda x: 1 - abs(request.discount_percentage - x) / 100)

    # 3. 카테고리 일치 점수 계산 (매우 중요한 요소)
    category_match_score = (df_products['category_cleaned'] == request.category).astype(int)

    # 4. 최종 유사도 계산 (가중치 조정: 카테고리 일치에 높은 가중치 부여)
    df_products['similarity'] = (
        text_similarities * 0.4 + 
        price_similarities * 0.2 + 
        discount_similarities * 0.1 +
        category_match_score * 0.3  # 카테고리 가중치 추가
    )
    
    # 5. 상위 3개 상품 선정
    top_3_products = df_products.sort_values(by='similarity', ascending=False).head(3)

    # 6. 결과 목록 생성 (리뷰 분석 포함)
    results = []
    for _, product in top_3_products.iterrows():
        # 리뷰 데이터 추출
        reviews = (str(product.get('review_title', '')) + ',' + str(product.get('review_content', ''))).split(',')
        reviews = [r.strip() for r in reviews if r.strip()]
        
        # 리뷰 분석 실행
        review_analysis_data = advanced_review_analysis(reviews)
        
        results.append({
            "product_id": product['product_id'],
            "product_name": product['product_name'],
            "similarity": product['similarity'],
            "discounted_price": product['discounted_price'],
            "rating": product['rating'],
            "rating_count": product['rating_count'],
            "review_analysis": review_analysis_data,
        })
        
    return results

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