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

# --- 상수 정의 ---
TECH_KEYWORDS = [
    'usb-c', 'usb-a', 'hdmi', '8-pin', 'lightning', 'micro-usb', 'type-c',
    'bluetooth', 'wireless', 'wired', '5g', '4g', 'lte', 'wifi',
    'ssd', 'hdd', 'ddr4', 'ddr5', 'ram', 'gb', 'tb',
    'led', 'lcd', 'oled', 'qled', '4k', '8k', '1080p',
    'noise cancelling', 'waterproof', 'water resistant',
    'sata', 'nvme', 'm.2'
]
W_TFIDF = 0.5
W_PRICE = 0.3
W_KEYWORD = 0.2

# --- Pydantic 모델 정의 ---
class Keyword(BaseModel):
    word: str
    count: int

class PredictionRequest(BaseModel):
    price: float
    category: str

class PredictionResponse(BaseModel):
    predicted_star: float
    predicted_review_count: float
    price_percentile: float
    review_count_percentile: float
    rating_percentile: float

class SimilarityRequest(BaseModel):
    description: str
    price: float
    discount_percentage: float
    category: str

class Product(BaseModel):
    product_id: str
    product_name: str

class ReviewAnalysis(BaseModel):
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    top_positive_keywords: List[str]
    top_negative_keywords: List[str]

class SimilarityResult(BaseModel):
    product_id: str
    product_name: str
    category: str
    similarity: float
    discounted_price: float
    rating: float
    rating_count: int
    review_analysis: ReviewAnalysis

class PriceRangeResponse(BaseModel):
    min_price: float
    max_price: float

class DistributionBin(BaseModel):
    name: str
    count: int

class CategoryStatsResponse(BaseModel):
    min_price: float
    max_price: float
    min_review_count: float
    max_review_count: float
    price_distribution: List[DistributionBin]
    review_count_distribution: List[DistributionBin]
    rating_distribution: List[DistributionBin]

# --- FastAPI 애플리케이션 설정 ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 전역 변수 ---
ml_pipe: Optional[Pipeline] = None
review_count_pipe: Optional[Pipeline] = None
hierarchical_categories_data: Dict = {}
tfidf_vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix: Optional[csr_matrix] = None
df_products: pd.DataFrame = pd.DataFrame()
df_reviews: pd.DataFrame = pd.DataFrame()
DATA_FILE_PATH = os.path.join("data", "cleaned_amazon_0519.csv")

# --- 내부 헬퍼 함수 ---
def _extract_tech_keywords(text: str) -> set:
    if not isinstance(text, str): return set()
    return {kw for kw in TECH_KEYWORDS if kw in text.lower()}

def _jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2: return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def advanced_review_analysis(reviews: List[str]) -> Dict[str, Any]:
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

# --- 핵심 로직: 데이터 로딩 및 모델 학습 ---
def load_data_and_train_models():
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

    df_raw['about_product'] = df_raw['about_product'].fillna('')
    df_raw['review_title'] = df_raw['review_title'].fillna('')
    df_raw['review_content'] = df_raw['review_content'].fillna('')
    for col in ['discounted_price', 'rating_count', 'rating']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw.dropna(subset=['product_id', 'discounted_price', 'rating_count', 'rating', 'category_cleaned'], inplace=True)

    reviews_list = []
    for _, row in df_raw.iterrows():
        contents = [c.strip() for c in str(row['review_content']).split(',') if c.strip()]
        title = str(row['review_title']).strip()
        for content_part in contents:
            full_review_text = (title + ' ' + content_part).strip()
            reviews_list.append({'product_id': row['product_id'], 'review_text': full_review_text})
    df_reviews = pd.DataFrame(reviews_list)
    
    df_meta_cols = ['product_id', 'product_name', 'about_product', 'category_cleaned', 'discounted_price', 'rating_count', 'rating']
    df_products = df_raw[df_meta_cols].drop_duplicates(subset=['product_id']).copy() # type: ignore
    df_products.dropna(subset=['discounted_price', 'rating_count', 'rating', 'category_cleaned'], inplace=True)

    numeric_features = ['discounted_price']
    categorical_features = ['category_cleaned']
    X_features = df_products[numeric_features + categorical_features]
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    
    y_rating = df_products['rating']
    ml_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
    ml_pipe.fit(X_features, y_rating)
    print("✅ Rating Prediction model training complete!")

    y_review_count = df_products['rating_count']
    review_count_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
    review_count_pipe.fit(X_features, y_review_count)
    print("✅ Review Count Prediction model training complete!")

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_products['about_product']) # type: ignore
    print("✅ TF-IDF model (based on product description) training complete!")
    
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

@app.get("/categories", response_model=List[str])
def get_categories():
    if not hierarchical_categories_data:
        raise HTTPException(status_code=503, detail="카테고리 데이터가 아직 로드되지 않았습니다.")
    return list(hierarchical_categories_data.keys())

@app.get("/hierarchical-categories", response_model=Dict)
def get_hierarchical_categories():
    if not hierarchical_categories_data:
        raise HTTPException(status_code=503, detail="카테고리 데이터가 아직 로드되지 않았습니다.")
    return hierarchical_categories_data

@app.get("/product-count", response_model=int)
def get_product_count(category: str = Query(..., description="상품 수를 조회할 전체 카테고리 경로")):
    if df_products.empty: return 0
    return int(df_products[df_products['category_cleaned'] == category].shape[0])

@app.get("/category-price-range", response_model=PriceRangeResponse)
def get_category_price_range(category: str = Query(...)):
    if df_products.empty: raise HTTPException(status_code=404, detail="상품 데이터가 없습니다.")
    filtered_df = df_products[df_products['category_cleaned'] == category]
    if filtered_df.empty: return PriceRangeResponse(min_price=0.0, max_price=0.0)
    min_price = filtered_df['discounted_price'].min()
    max_price = filtered_df['discounted_price'].max()
    return PriceRangeResponse(min_price=float(min_price), max_price=float(max_price))

@app.get("/category-stats", response_model=CategoryStatsResponse)
def get_category_stats(category: str = Query(..., description="통계 정보를 조회할 카테고리")):
    if df_products.empty: raise HTTPException(status_code=503, detail="상품 데이터가 아직 로드되지 않았습니다.")
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
    if df_products.empty or tfidf_matrix is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다.")

    df_cat = df_products[df_products['category_cleaned'] == request.category].copy()
    if df_cat.empty: return []
    
    indices = df_cat.index.to_numpy()
    category_tfidf_matrix = tfidf_matrix[indices]
    desc_vector = tfidf_vectorizer.transform([request.description])
    s_tfidf = cosine_similarity(desc_vector, category_tfidf_matrix).flatten()

    user_price = request.price
    product_prices = df_cat['discounted_price']
    price_range = product_prices.max() - product_prices.min()
    s_price = (1 - (np.abs(product_prices - user_price) / price_range)) if price_range > 0 else pd.Series(1.0, index=df_cat.index)
    s_price[s_price < 0] = 0

    user_keywords = _extract_tech_keywords(request.description)
    s_keyword_series = df_cat['about_product'].apply(lambda p_text: _jaccard_similarity(user_keywords, _extract_tech_keywords(str(p_text)))) if user_keywords else pd.Series(0.0, index=df_cat.index)
    s_keyword = s_keyword_series.values

    final_score = (W_TFIDF * s_tfidf) + (W_PRICE * s_price.values) + (W_KEYWORD * s_keyword)
    df_cat['similarity'] = final_score
    
    df_top10 = df_cat.sort_values(by='similarity', ascending=False).head(10)
    
    results = []
    for _, row in df_top10.iterrows():
        product_reviews = df_reviews[df_reviews['product_id'] == row['product_id']]['review_text'].tolist()
        review_analysis_result_dict = advanced_review_analysis(product_reviews)
        review_analysis_result = ReviewAnalysis(**review_analysis_result_dict)
        results.append(SimilarityResult(
            product_id=str(row['product_id']),
            product_name=str(row['product_name']),
            category=str(row['category_cleaned']),
            similarity=float(row['similarity']),
            discounted_price=float(row['discounted_price']),
            rating=float(row['rating']),
            rating_count=int(row['rating_count']),
            review_analysis=review_analysis_result,
        ))
    return results

@app.post("/predict", response_model=PredictionResponse)
def predict_star_rating(request: PredictionRequest):
    if ml_pipe is None or df_products.empty or review_count_pipe is None:
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")
    
    try:
        input_data = pd.DataFrame([[request.price, request.category]], columns=['discounted_price', 'category_cleaned'])
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