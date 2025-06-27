from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
import os
import shutil
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix
from collections import Counter

# --- 카테고리별 키워드 세트 ---
# 이전에 사용되던 카테고리별 키워드셋은 TECH_KEYWORDS 방식으로 대체되었습니다.
KEYWORD_SETS = {}

# 제품의 핵심 기능을 정의하는 키워드셋
# 이 키워드들은 제품의 종류를 명확히 구분하는 데 사용됨
TECH_KEYWORDS = [
    'usb-c', 'usb-a', 'hdmi', '8-pin', 'lightning', 'micro-usb', 'type-c',
    'bluetooth', 'wireless', 'wired', '5g', '4g', 'lte', 'wifi',
    'ssd', 'hdd', 'ddr4', 'ddr5', 'ram', 'gb', 'tb',
    'led', 'lcd', 'oled', 'qled', '4k', '8k', '1080p',
    'noise cancelling', 'waterproof', 'water resistant',
    'sata', 'nvme', 'm.2'
]

# --- Pydantic 모델 정의 ---
# API 요청/응답의 데이터 구조를 정의하여 유효성 검사 및 문서화를 자동화합니다.
class Keyword(BaseModel):
    """리뷰 분석 결과에서 사용될 키워드와 빈도수 모델"""
    word: str
    count: int

class PredictionRequest(BaseModel):
    """판매 지표 예측 요청 모델"""
    price: float
    category: str

class PredictionResponse(BaseModel):
    """판매 지표 예측 결과 응답 모델"""
    predicted_star: float
    predicted_review_count: float
    price_percentile: float
    review_count_percentile: float
    rating_percentile: float

class SimilarityRequest(BaseModel):
    """유사도 검색 요청 모델"""
    description: str
    price: float
    discount_percentage: float
    category: str

class Product(BaseModel):
    """기본 상품 정보 모델"""
    product_id: str
    product_name: str

class ReviewAnalysis(BaseModel):
    """리뷰 분석 결과 모델"""
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    top_positive_keywords: List[str]
    top_negative_keywords: List[str]

class ProductInfo(BaseModel):
    """상품의 종합 정보 모델"""
    product_id: str
    product_name: str
    category: str
    price: float
    review_count: int
    review_analysis: ReviewAnalysis

class SimilarProduct(ProductInfo):
    """유사도 점수가 포함된 상품 정보 모델 (현재는 직접 사용되지 않음)"""
    similarity: float

class SimilarityResult(BaseModel):
    """유사도 검색 결과로 반환될 각 상품의 상세 정보 모델"""
    product_id: str
    product_name: str
    category: str
    similarity: float # 최종 유사도 점수 (TF-IDF + 키워드 점수)
    discounted_price: float
    rating: float
    rating_count: int
    review_analysis: ReviewAnalysis

class PriceRangeResponse(BaseModel):
    """카테고리별 가격 범위 응답 모델"""
    min_price: float
    max_price: float

class DistributionBin(BaseModel):
    """분포도 차트의 각 막대를 나타내는 모델"""
    name: str
    count: int

class CategoryStatsResponse(BaseModel):
    """카테고리 통계 정보 응답 모델"""
    min_price: float
    max_price: float
    min_review_count: float
    max_review_count: float
    price_distribution: List[DistributionBin]
    review_count_distribution: List[DistributionBin]
    rating_distribution: List[DistributionBin]

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
ml_pipe: Optional[Pipeline] = None
review_count_pipe: Optional[Pipeline] = None # 리뷰 수 예측 모델
hierarchical_categories_data: Dict = {} # 계층적 카테고리 데이터
tfidf_vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix: Optional[csr_matrix] = None
df_products: pd.DataFrame = pd.DataFrame() # 상품 메타데이터 및 유사도 분석용
df_reviews: pd.DataFrame = pd.DataFrame() # 상품별 개별 리뷰 저장용
DATA_FILE_PATH = os.path.join("data", "cleaned_amazon_0519.csv")

# --- 핵심 로직: 데이터 로딩 및 모델 학습 ---
def load_data_and_train_models():
    """
    서버 시작 시 실행되는 핵심 함수.
    1. CSV 데이터 파일을 읽어들입니다.
    2. 데이터를 정제하고 기본 타입을 변환합니다.
    3. 'review_content'에 쉼표로 합쳐진 리뷰들을 분리하여 상품별 개별 리뷰(df_reviews)를 생성합니다.
       이 과정에서 더 정확한 분석을 위해 리뷰 제목('review_title')과 내용('review_content')을 결합합니다.
    4. 상품 메타데이터(df_products)를 생성합니다.
    5. df_products를 기반으로 3가지 머신러닝 모델을 학습합니다:
        - 별점 예측 모델 (ml_pipe)
        - 리뷰 수 예측 모델 (review_count_pipe)
        - 상품 설명 기반 TF-IDF 모델 (tfidf_vectorizer, tfidf_matrix)
    6. 프론트엔드에서 사용할 계층적 카테고리 데이터를 생성합니다.
    """
    global ml_pipe, review_count_pipe, tfidf_vectorizer, tfidf_matrix, df_products, df_reviews, hierarchical_categories_data
    
    if not os.path.exists(DATA_FILE_PATH):
        print(f"⚠️ 데이터 파일이 존재하지 않습니다: {DATA_FILE_PATH}")
        df_products, df_reviews = pd.DataFrame(), pd.DataFrame()
        return

    df_raw = pd.read_csv(DATA_FILE_PATH)
    
    required_columns = ['product_id', 'product_name', 'about_product', 'review_title', 'review_content', 'discounted_price', 'rating_count', 'category_cleaned', 'rating']
    if any(col not in df_raw.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df_raw.columns]
        raise ValueError(f"필수 컬럼 중 일부가 누락되었습니다: {', '.join(missing_cols)}")

    # --- 1. 기본 클리닝 및 타입 변환 ---
    df_raw['about_product'] = df_raw['about_product'].fillna('')
    df_raw['review_title'] = df_raw['review_title'].fillna('')
    df_raw['review_content'] = df_raw['review_content'].fillna('')
    for col in ['discounted_price', 'rating_count', 'rating']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw.dropna(subset=['product_id', 'discounted_price', 'rating_count', 'rating', 'category_cleaned'], inplace=True)

    # --- 2. 리뷰 분리 및 df_reviews 생성 (제목 + 내용 결합) ---
    # CSV의 한 셀에 모든 리뷰가 쉼표로 합쳐져 있는 문제를 해결하기 위해,
    # 각 상품의 모든 리뷰를 개별 행으로 분리하여 별도의 DataFrame(df_reviews)을 생성합니다.
    reviews_list = []
    for _, row in df_raw.iterrows():
        contents = [c.strip() for c in str(row['review_content']).split(',') if c.strip()]
        title = str(row['review_title']).strip()
        for content_part in contents:
            full_review_text = (title + ' ' + content_part).strip()
            reviews_list.append({'product_id': row['product_id'], 'review_text': full_review_text})
    df_reviews = pd.DataFrame(reviews_list)
    
    # --- 3. 유사도 분석 및 예측 모델 학습용 df_products 생성 ---
    # 리뷰 데이터를 제외한 순수 상품 정보(메타데이터)만으로 DataFrame을 구성하고, 상품 ID 기준 중복을 제거합니다.
    df_meta_cols = ['product_id', 'product_name', 'about_product', 'category_cleaned', 'discounted_price', 'rating_count', 'rating']
    df_products = df_raw[df_meta_cols].drop_duplicates(subset=['product_id']).copy() # type: ignore
    df_products.dropna(subset=['discounted_price', 'rating_count', 'rating', 'category_cleaned'], inplace=True)

    # --- 4. 모델 학습 ---
    # 가격(numeric)과 카테고리(categorical) 정보를 기반으로 별점과 리뷰 수를 예측하는 모델을 만듭니다.
    numeric_features = ['discounted_price']
    categorical_features = ['category_cleaned']
    X_features = df_products[numeric_features + categorical_features]
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    
    # 모델 1: 별점(rating) 예측
    y_rating = df_products['rating']
    ml_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
    ml_pipe.fit(X_features, y_rating)
    print("✅ Rating Prediction model training complete!")

    # 모델 2: 리뷰 수(rating_count) 예측
    y_review_count = df_products['rating_count']
    review_count_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
    review_count_pipe.fit(X_features, y_review_count)
    print("✅ Review Count Prediction model training complete!")

    # 모델 3: TF-IDF 모델 학습 (유사도 분석용)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_products['about_product']) # type: ignore
    print("✅ TF-IDF model (based on product description) training complete!")
    
    # --- 5. 계층적 카테고리 데이터 생성 ---
    # 프론트엔드에서 동적 카테고리 선택 UI를 구현하기 위해
    temp_hierarchical_data = {}
    for cat_string in df_products['category_cleaned'].unique():
        parts = cat_string.split(' | ')
        current_level = temp_hierarchical_data
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
    hierarchical_categories_data = temp_hierarchical_data
    print("✅ Hierarchical category data created!")
    print(f"📈 Total {len(df_products)} unique products and {len(df_reviews)} individual reviews loaded.")
    print(f"⭐ Rating range found in data: {df_products['rating'].min()} ~ {df_products['rating'].max()}")


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
    """API 서버의 루트 엔드포인트. 서버가 실행 중인지 확인하는 데 사용됩니다."""
    return {"message": "Rmazon predictor and similarity API is running!"}

@app.get("/categories", response_model=List[str])
def get_categories():
    """데이터셋에 있는 모든 최상위 카테고리 목록을 반환합니다."""
    if not hierarchical_categories_data:
        raise HTTPException(status_code=503, detail="카테고리 데이터가 아직 로드되지 않았습니다.")
    return list(hierarchical_categories_data.keys())

@app.get("/hierarchical-categories", response_model=Dict)
def get_hierarchical_categories():
    """전체 카테고리 구조를 계층적(JSON) 형태로 반환합니다."""
    if not hierarchical_categories_data:
        raise HTTPException(status_code=503, detail="카테고리 데이터가 아직 로드되지 않았습니다.")
    return hierarchical_categories_data

@app.get("/product-count", response_model=int)
def get_product_count(category: str = Query(..., description="상품 수를 조회할 전체 카테고리 경로")):
    """선택된 최종 카테고리에 해당하는 상품의 총 개수를 반환합니다."""
    if df_products.empty:
        return 0
    return int(df_products[df_products['category_cleaned'] == category].shape[0])

@app.get("/products", response_model=List[Product])
def get_products(category: Optional[str] = Query(None)):
    """특정 카테고리에 속한 상품 목록을 반환합니다."""
    if df_products.empty:
        return []
    target_df = df_products
    if category:
        target_df = df_products[df_products['category_cleaned'] == category]
    return target_df[['product_id', 'product_name']].to_dict(orient='records') # type: ignore

def advanced_review_analysis(reviews: List[str]) -> Dict[str, Any]:
    """
    VADER를 사용하여 리뷰 목록의 긍/부정/중립 비율을 분석하고,
    긍정/부정 리뷰에서 가장 빈도가 높은 키워드를 추출합니다.
    """
    if not reviews:
        return {"positive_percentage": 0, "negative_percentage": 0, "neutral_percentage": 100, "top_positive_keywords": [], "top_negative_keywords": []}
    
    analyzer = SentimentIntensityAnalyzer()
    positive_reviews_text, negative_reviews_text = [], []
    pos_count, neg_count, neu_count = 0, 0, 0

    for review in reviews:
        score = analyzer.polarity_scores(review)
        if score['compound'] >= 0.05:
            positive_reviews_text.append(review)
            pos_count += 1
        elif score['compound'] <= -0.05:
            negative_reviews_text.append(review)
            neg_count += 1
        else:
            neu_count += 1

    total = len(reviews)
    positive_percentage = (pos_count / total) * 100
    negative_percentage = (neg_count / total) * 100
    neutral_percentage = (neu_count / total) * 100

    custom_stop_words = text.ENGLISH_STOP_WORDS.union(['product', 'good', 'great', 'bad', 'price', 'quality', 'item', 'like', 'just', 'really', 'very', 'amazon', 'nice', 'best', 'time', 'use', 'working', 'not', 'don', 'doesn', 'is', 'are'])

    def extract_top_keywords(texts: List[str], stop_words, top_n=5) -> List[str]:
        if not texts: return []
        words = ' '.join(texts).lower().split()
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        return [word for word, count in Counter(filtered_words).most_common(top_n)]

    positive_keywords = extract_top_keywords(positive_reviews_text, custom_stop_words)
    negative_keywords = extract_top_keywords(negative_reviews_text, custom_stop_words)

    return {
        'positive_percentage': round(positive_percentage, 2),
        'negative_percentage': round(negative_percentage, 2),
        'neutral_percentage': round(neutral_percentage, 2),
        'top_positive_keywords': positive_keywords,
        'top_negative_keywords': negative_keywords
    }

@app.get("/category-price-range", response_model=PriceRangeResponse)
def get_category_price_range(category: str = Query(...)):
    """특정 카테고리의 최소/최대 가격을 반환합니다."""
    if df_products.empty:
        raise HTTPException(status_code=404, detail="상품 데이터가 없습니다.")
    filtered_df = df_products[df_products['category_cleaned'] == category]
    if filtered_df.empty:
        return PriceRangeResponse(min_price=0.0, max_price=0.0)
    min_price = filtered_df['discounted_price'].min()
    max_price = filtered_df['discounted_price'].max()
    return PriceRangeResponse(min_price=float(min_price), max_price=float(max_price))

@app.get("/category-stats", response_model=CategoryStatsResponse)
def get_category_stats(category: str = Query(..., description="통계 정보를 조회할 카테고리")):
    """특정 카테고리의 가격, 리뷰 수, 별점 분포 등 종합적인 통계 정보를 반환합니다."""
    if df_products.empty:
        raise HTTPException(status_code=503, detail="상품 데이터가 아직 로드되지 않았습니다.")
    
    category_df = df_products[df_products['category_cleaned'] == category]
    if category_df.empty:
        return CategoryStatsResponse(min_price=0, max_price=0, min_review_count=0, max_review_count=0, price_distribution=[], review_count_distribution=[], rating_distribution=[])

    def create_histogram(data: pd.Series, bins=10) -> List[DistributionBin]:
        if data.empty: return []
        counts, bin_edges = np.histogram(data.dropna(), bins=bins)
        return [DistributionBin(name=f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}", count=int(counts[i])) for i in range(len(counts))]

    def process_rating_distribution(data: pd.Series) -> List[DistributionBin]:
        if data.empty: return []
        rating_counts = (data.dropna() / 0.5).round() * 0.5
        rating_distribution = rating_counts.value_counts().sort_index()
        return [DistributionBin(name=f"{rating:.1f}", count=int(count)) for rating, count in rating_distribution.items()]

    price_dist = create_histogram(category_df['discounted_price']) # type: ignore
    review_count_dist = create_histogram(category_df['rating_count']) # type: ignore
    rating_dist = process_rating_distribution(category_df['rating']) # type: ignore

    return CategoryStatsResponse(
        min_price=float(category_df['discounted_price'].min()), # type: ignore
        max_price=float(category_df['discounted_price'].max()), # type: ignore
        min_review_count=float(category_df['rating_count'].min()), # type: ignore
        max_review_count=float(category_df['rating_count'].max()), # type: ignore
        price_distribution=price_dist,
        review_count_distribution=review_count_dist,
        rating_distribution=rating_dist
    )

@app.post("/search-similarity", response_model=List[SimilarityResult])
def search_similarity(request: SimilarityRequest):
    """
    사용자 입력을 기반으로 3가지 요소를 종합하여 유사 상품을 검색합니다.
    [현재 디버깅 중] - 텍스트 유사도(TF-IDF)만으로 점수를 계산하여 문제의 원인을 격리합니다.
    """
    if df_products.empty or tfidf_vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")

    # 1. 특정 카테고리의 상품만 필터링
    df_filtered = df_products[df_products['category_cleaned'] == request.category].copy()
    if df_filtered.empty:
        return []

    # --- 문제 격리를 위해 텍스트 유사도만 계산 ---
    filtered_indices = df_filtered.index.tolist()
    filtered_tfidf_matrix = tfidf_matrix[filtered_indices]

    user_desc_vector = tfidf_vectorizer.transform([request.description])
    text_similarities = cosine_similarity(user_desc_vector, filtered_tfidf_matrix).flatten()
    
    # 최종 유사도를 오직 텍스트 유사도만으로 설정
    df_filtered['similarity'] = text_similarities * 100

    # 상위 10개 결과 정렬 및 선택
    top_10_similar_products = df_filtered.sort_values(by='similarity', ascending=False).head(10) # type: ignore
    
    results = []
    for _, row in top_10_similar_products.iterrows():
        product_reviews = df_reviews[df_reviews['product_id'] == row['product_id']]['review_text'].tolist()
        review_analysis_data = advanced_review_analysis(product_reviews)
        
        results.append(SimilarityResult(
            product_id=str(row['product_id']),
            product_name=str(row['product_name']),
            category=str(row['category_cleaned']),
            similarity=round(float(row['similarity']), 2),
            discounted_price=float(row['discounted_price']),
            rating=float(row['rating']),
            rating_count=int(row['rating_count']),
            review_analysis=ReviewAnalysis(**review_analysis_data)
        ))
        
    return results

@app.post("/predict", response_model=PredictionResponse)
def predict_star_rating(request: PredictionRequest):
    """
    입력된 가격과 카테고리를 바탕으로 예상 별점과 리뷰 수를 예측합니다.
    또한 입력된 가격이 해당 카테고리 내에서 어느 정도 수준인지 백분위 정보를 함께 제공합니다.
    """
    if ml_pipe is None or df_products.empty or review_count_pipe is None:
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")
    
    try:
        input_data = pd.DataFrame([[request.price, request.category]], columns=['discounted_price', 'category_cleaned']) # type: ignore
        predicted_star = ml_pipe.predict(input_data)[0]
        predicted_review_count = review_count_pipe.predict(input_data)[0]

        category_df = df_products[df_products['category_cleaned'] == request.category]
        
        def calculate_percentile(series: pd.Series, score: float) -> float:
            if series.empty: return 0.0
            from scipy.stats import percentileofscore
            return float(percentileofscore(series.dropna(), score, kind='weak'))

        price_percentile = calculate_percentile(category_df['discounted_price'], request.price) # type: ignore
        review_count_percentile = calculate_percentile(category_df['rating_count'], float(predicted_review_count)) # type: ignore
        rating_percentile = calculate_percentile(category_df['rating'], float(predicted_star)) # type: ignore

        return PredictionResponse(
            predicted_star=round(float(predicted_star), 2),
            predicted_review_count=round(float(predicted_review_count), 0),
            price_percentile=round(price_percentile, 2),
            review_count_percentile=round(review_count_percentile, 2),
            rating_percentile=round(rating_percentile, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {e}") 