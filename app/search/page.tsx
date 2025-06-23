"use client"

import { useState, useEffect, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { BarChart, Search, AlertCircle, Loader2, Star, ThumbsUp, ThumbsDown, MessageSquareQuote, Sparkles, Tag, ChevronsUpDown, PackageSearch } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"

// --- 데이터 타입 정의 ---

interface ReviewAnalysis {
  positive_percentage: number;
  negative_percentage: number;
  neutral_percentage: number;
  top_positive_keywords: string[];
  top_negative_keywords: string[];
}

interface SimilarityResult {
  product_id: string;
  product_name: string;
  category: string;
  similarity: number;
  discounted_price: number;
  rating: number;
  rating_count: number;
  review_analysis: ReviewAnalysis;
}

interface PriceRange {
  min_price: number;
  max_price: number;
}

interface HierarchicalCategories {
  [key: string]: HierarchicalCategories | {};
}

const API_BASE_URL = "http://127.0.0.1:8000";

// --- 헬퍼 함수 ---
const getSentimentSummary = (analysis: ReviewAnalysis): { text: string; color: string } => {
  const { positive_percentage, negative_percentage } = analysis;

  if (positive_percentage >= 70) return { text: "아주 긍정적", color: "text-green-700" };
  if (negative_percentage >= 60) return { text: "아주 부정적", color: "text-red-700" };
  if (positive_percentage >= 35 && negative_percentage >= 35) return { text: "평가가 극단적임", color: "text-yellow-700" };
  
  if (positive_percentage > negative_percentage * 1.5) return { text: "대체로 긍정적", color: "text-green-600" };
  if (negative_percentage > positive_percentage * 1.5) return { text: "대체로 부정적", color: "text-red-600" };
  
  return { text: "중립적", color: "text-gray-600" };
};

// --- 컴포넌트 ---

const SentimentIcon = ({ sentiment }: { sentiment: 'positive' | 'negative' | 'neutral' }) => {
  if (sentiment === 'positive') return <ThumbsUp className="w-5 h-5 text-green-500" />;
  if (sentiment === 'negative') return <ThumbsDown className="w-5 h-5 text-red-500" />;
  return <MessageSquareQuote className="w-5 h-5 text-gray-500" />;
};

export default function SearchPage() {
  // --- 상태 관리 ---
  const [description, setDescription] = useState("A high-quality, wireless headphone with noise-cancelling feature and long battery life. Comes with a carrying case.");
  
  // 카테고리 상태 (계층형으로 변경)
  const [hierarchicalCategories, setHierarchicalCategories] = useState<HierarchicalCategories>({});
  const [cat1, setCat1] = useState("");
  const [cat2, setCat2] = useState("");
  const [cat3, setCat3] = useState("");

  // 가격 상태
  const [priceRange, setPriceRange] = useState<PriceRange>({ min_price: 10, max_price: 10000 });
  const [price, setPrice] = useState(500);

  // 할인율 상태
  const [discount, setDiscount] = useState(15);
  
  const [isLoading, setIsLoading] = useState(false);
  const [isDataLoading, setIsDataLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SimilarityResult[]>([]);
  const [similarityWarning, setSimilarityWarning] = useState<string | null>(null);
  const [productCount, setProductCount] = useState<number | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  // --- 파생 상태 (Memoization) ---
  const cat1Options = useMemo(() => Object.keys(hierarchicalCategories), [hierarchicalCategories]);
  const cat2Options = useMemo(() => cat1 ? Object.keys((hierarchicalCategories as any)[cat1] || {}) : [], [hierarchicalCategories, cat1]);
  const cat3Options = useMemo(() => cat1 && cat2 ? Object.keys((hierarchicalCategories as any)[cat1]?.[cat2] || {}) : [], [hierarchicalCategories, cat1, cat2]);

  // 카테고리 선택이 완료되었는지, 그리고 완료된 카테고리 문자열이 무엇인지 동적으로 결정
  const { fullCategory, fullCategorySelected } = useMemo(() => {
    if (!cat1) return { fullCategory: null, fullCategorySelected: false };

    // 다음 레벨의 카테고리가 있는데 선택하지 않은 경우
    if (cat2Options.length > 0 && !cat2) return { fullCategory: null, fullCategorySelected: false };
    if (cat3Options.length > 0 && !cat3) return { fullCategory: null, fullCategorySelected: false };

    // 모든 하위 카테고리가 선택되었을 때, 전체 카테고리 경로를 생성
    const finalCategory = [cat1, cat2, cat3].filter(Boolean).join(" | ");
    return { fullCategory: finalCategory, fullCategorySelected: true };
  }, [cat1, cat2, cat3, cat2Options, cat3Options]);
  
  const discountedPrice = useMemo(() => price * (1 - discount / 100), [price, discount]);

  // --- 데이터 로딩 Effect ---
  // 1. 초기 계층형 카테고리 목록 로드
  useEffect(() => {
    const fetchHierarchicalCategories = async () => {
      try {
        setError(null);
        setIsDataLoading(true);
        const response = await fetch(`${API_BASE_URL}/hierarchical-categories`);
        if (!response.ok) throw new Error("카테고리 목록 로딩 실패");
        const data = await response.json();
        setHierarchicalCategories(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "카테고리 로딩 중 오류");
      } finally {
        setIsDataLoading(false);
      }
    };
    fetchHierarchicalCategories();
  }, []);

  // 2. 카테고리 선택 시 가격 범위 로드
  useEffect(() => {
    if (!fullCategory) return;

    const fetchPriceRange = async () => {
      try {
        setError(null);
        setIsDataLoading(true);
        const response = await fetch(`${API_BASE_URL}/category-price-range?category=${encodeURIComponent(fullCategory)}`);
        if (!response.ok) throw new Error("가격 범위 로딩 실패");
        const data: PriceRange = await response.json();
        
        if (data.min_price < data.max_price) {
            setPriceRange(data);
            setPrice(Math.round((data.min_price + data.max_price) / 2));
        } else { 
            const defaultRange = { min_price: 10, max_price: 10000 };
            setPriceRange(defaultRange);
            setPrice(Math.round((defaultRange.min_price + defaultRange.max_price) / 2));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "가격 범위 로딩 중 오류");
      } finally {
        setIsDataLoading(false);
      }
    };
    fetchPriceRange();
  }, [fullCategory]);

  // 3. 카테고리 선택 시 상품 수 로드
  useEffect(() => {
    if (!fullCategory) {
      setProductCount(null);
      return;
    }

    const fetchProductCount = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/product-count?category=${encodeURIComponent(fullCategory)}`);
        if (!response.ok) throw new Error("상품 수 로딩 실패");
        const count = await response.json();
        setProductCount(count);
      } catch (err) {
        // 에러를 표시하기보다 그냥 null로 설정하여 UI를 깔끔하게 유지
        setProductCount(null);
        console.error("Failed to fetch product count:", err);
      }
    };

    fetchProductCount();
  }, [fullCategory]);
  
  // 카테고리 변경 핸들러
  const handleCat1Change = (value: string) => {
    setCat1(value);
    setCat2("");
    setCat3("");
  };
  const handleCat2Change = (value: string) => {
    setCat2(value);
    setCat3("");
  };

  // --- 이벤트 핸들러 ---
  const handleSearch = async () => {
    if (!description.trim() || !fullCategory) {
      setError("상품 설명과 카테고리를 모두 입력해주세요.");
      return;
    }
    
    setHasSearched(true);
    setIsLoading(true);
    setError(null);
    setResults([]);
    setSimilarityWarning(null);

    try {
      const response = await fetch(`${API_BASE_URL}/search-similarity`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: description,
          price: price,
          discount_percentage: discount,
          category: fullCategory,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "유사도 분석에 실패했습니다.");
      }
      
      const data: SimilarityResult[] = await response.json();
      setResults(data);

      // 유사도 경고 체크
      const warnings: string[] = [];
      if (data.length > 0) {
        const maxSim = Math.max(...data.map(r => r.similarity));
        const avgSim = data.reduce((acc, r) => acc + r.similarity, 0) / data.length;
        
        if (maxSim < 0.6) {
          warnings.push("⚠️ 주의: 최고 유사도가 60% 미만입니다. 입력한 설명과 매우 유사한 제품이 거의 없으며, 유사 제품 목록의 정확도가 낮을 수 있습니다.");
        }
        if (avgSim < 0.5) {
          warnings.push("⚠️ 주의: 평균 유사도가 50% 미만입니다.입력한 설명이 다른 제품들과 전반적으로 크게 다르며, 입력 정보를 조정하여 더 정확한 결과를 얻을 수 있습니다.");
        }
      }
      setSimilarityWarning(warnings.join('|'));

    } catch (err) {
       setError(err instanceof Error ? err.message : "유사도 분석 중 오류 발생");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50/50">
      <header className="bg-white border-b p-4 shadow-sm sticky top-0 z-10">
        <h1 className="text-xl font-semibold">유사 상품 탐색</h1>
        <p className="text-sm text-muted-foreground">가상의 상품 정보를 입력하여 시장 내 유사 제품과 리뷰를 심층 분석합니다.</p>
      </header>

      <div className="flex-1 overflow-y-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel: Controls */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>가상 상품 정보 입력</CardTitle>
              <CardDescription>분석하고 싶은 가상 상품의 정보를 입력하세요.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>카테고리</Label>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <Select onValueChange={handleCat1Change} value={cat1} disabled={isDataLoading}>
                    <SelectTrigger><SelectValue placeholder="대분류" /></SelectTrigger>
                    <SelectContent>{cat1Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                  </Select>
                  <Select onValueChange={handleCat2Change} value={cat2} disabled={!cat1 || isDataLoading || cat2Options.length === 0}>
                    <SelectTrigger>
                      <SelectValue placeholder={cat1 && cat2Options.length === 0 ? "(하위 분류 없음)" : "중분류"} />
                    </SelectTrigger>
                    <SelectContent>{cat2Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                  </Select>
                  <Select onValueChange={setCat3} value={cat3} disabled={!cat2 || isDataLoading || cat3Options.length === 0}>
                    <SelectTrigger>
                      <SelectValue placeholder={
                        (cat1 && cat2Options.length === 0) 
                        ? "(하위 분류 없음)" 
                        : (cat2 && cat3Options.length === 0 ? "(하위 분류 없음)" : "소분류")
                      } />
                    </SelectTrigger>
                    <SelectContent>{cat3Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
                {fullCategorySelected && productCount !== null && (
                  <p className="text-xs text-gray-500 pt-1">
                    현재 카테고리 내 상품 수는 <span className="font-semibold text-gray-600">{productCount.toLocaleString()}</span>개입니다.
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">상품 설명</Label>
                <Textarea 
                  id="description" 
                  placeholder="예: 고품질 무선 노이즈캔슬링 헤드폰..." 
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={4}
                />
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <Label htmlFor="price">정가 (₹)</Label>
                    <span className="text-sm font-medium text-muted-foreground">₹{price.toLocaleString()}</span>
                </div>
                <Slider
                  id="price"
                  min={priceRange.min_price}
                  max={priceRange.max_price}
                  step={1}
                  value={[price]}
                  onValueChange={(value) => setPrice(value[0])}
                  disabled={isDataLoading || !fullCategorySelected}
                />
                <div className="flex justify-between text-xs text-muted-foreground px-1">
                  <span>최소: ₹{priceRange.min_price.toLocaleString()}</span>
                  <span>최대: ₹{priceRange.max_price.toLocaleString()}</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <Label htmlFor="discount">할인율 (%)</Label>
                    <span className="text-sm font-medium text-muted-foreground">{discount}%</span>
                </div>
                <Slider
                  id="discount"
                  min={0}
                  max={100}
                  step={1}
                  value={[discount]}
                  onValueChange={(value) => setDiscount(value[0])}
                />
                <p className="text-sm text-muted-foreground mt-1 pl-1">적용된 상품 가격 : <span className="font-bold text-gray-800">₹{discountedPrice.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</span></p>
              </div>
            </CardContent>
          </Card>
          
          <Button onClick={handleSearch} disabled={isLoading || isDataLoading || !fullCategorySelected} className="w-full py-6 text-lg">
            {isLoading || isDataLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Search className="mr-2 h-5 w-5" />}
            {isLoading ? "분석 중..." : (isDataLoading && !cat1 ? "카테고리 로딩 중..." : "유사 상품 분석")}
          </Button>

          {error && (
            <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              <p>{error}</p>
            </div>
          )}
        </div>
        
        {/* Right Panel: Results */}
        <div className="lg:col-span-2">
          {isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
              <Loader2 className="w-12 h-12 animate-spin text-purple-600 mb-4" />
              <p className="text-lg">유사 상품을 분석하고 있습니다...</p>
              <p>잠시만 기다려주세요.</p>
            </div>
          )}

          {!isLoading && error && (
            <div className="text-red-500 bg-red-100 p-4 rounded-md">
              <p>{error}</p>
            </div>
          )}

          {/* 경고 블록: 경고가 있을 때만 렌더링 */}
          {similarityWarning && (
            <div className="space-y-2">
              {similarityWarning.split('|').filter(w => w).map((warning, index) => (
                <div key={index} className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md" role="alert">
                  <p>{warning}</p>
                </div>
              ))}
            </div>
          )}

          {/* 1. 초기 상태 (검색 전) */}
          {!isLoading && !hasSearched && (
            <div className="flex flex-col items-center justify-center h-full text-center bg-white rounded-lg p-8 shadow-sm">
              <Sparkles className="w-16 h-16 text-purple-400 mb-4" />
              <h3 className="text-xl font-semibold mb-1">유사 상품 분석 시작하기</h3>
              <p className="text-gray-500 max-w-md">
                좌측 패널에서 가상 상품의 카테고리, 설명, 가격 등을 설정한 후 '유사 상품 분석' 버튼을 눌러주세요.
                입력된 정보와 유사한 실제 상품 목록 및 리뷰 분석 결과를 확인할 수 있습니다.
              </p>
            </div>
          )}

          {/* 2. 결과 없음 (검색 후) */}
          {!isLoading && hasSearched && results.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center bg-white rounded-lg p-8 shadow-sm">
              <PackageSearch className="w-16 h-16 text-gray-300 mb-4" />
              <h3 className="text-xl font-semibold mb-1">결과 없음</h3>
              <p className="text-gray-500">입력한 조건과 일치하는 상품을 찾지 못했습니다.</p>
            </div>
          )}

          {/* 3. 결과 카드 (검색 후) */}
          {!isLoading && hasSearched && results.length > 0 && (
            <div className="space-y-6">
              {results.map((result, index) => {
                const sentiment = getSentimentSummary(result.review_analysis);
                return (
                  <Card key={result.product_id} className="overflow-hidden">
                    <CardHeader className="bg-gray-50/70">
                      <div className="flex justify-between items-start">
                        <div>
                          <Badge variant={index === 0 ? "default" : "secondary"} className="mb-2">
                            {index === 0 ? "가장 유사한 상품" : `${index + 1}번째 유사 상품`}
                          </Badge>
                          <CardTitle className="text-lg">{result.product_name}</CardTitle>
                          <CardDescription className="flex items-center gap-2 pt-1">
                            <Tag size={14} /> {result.category}
                          </CardDescription>
                        </div>
                        <div className="text-right flex-shrink-0">
                          <p className="text-sm text-muted-foreground">유사도</p>
                          <p className="text-2xl font-bold text-purple-700">{(result.similarity * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-4 grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="space-y-4 md:col-span-1">
                        <h4 className="font-semibold">상품 기본 정보</h4>
                        <div className="flex justify-between items-center text-sm p-2 bg-gray-50 rounded-md">
                          <span className="text-muted-foreground">가격</span>
                          <span className="font-mono">₹{result.discounted_price.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center text-sm p-2 bg-gray-50 rounded-md">
                          <span className="text-muted-foreground">평점</span>
                          <div className="flex items-center gap-1">
                            <Star className="w-4 h-4 text-yellow-400" />
                            <span>{result.rating.toFixed(1)}</span>
                          </div>
                        </div>
                        <div className="flex justify-between items-center text-sm p-2 bg-gray-50 rounded-md">
                          <span className="text-muted-foreground">리뷰 수</span>
                          <span>{result.rating_count.toLocaleString()} 개</span>
                        </div>
                      </div>
                      <div className="space-y-4 md:col-span-2">
                        <h4 className="font-semibold">리뷰 종합 분석</h4>
                        <div className="w-full bg-gray-200 rounded-full h-2.5 flex overflow-hidden">
                          <div className="bg-green-500 h-2.5" style={{ width: `${result.review_analysis.positive_percentage}%` }}></div>
                          <div className="bg-gray-400 h-2.5" style={{ width: `${result.review_analysis.neutral_percentage}%` }}></div>
                          <div className="bg-red-500 h-2.5" style={{ width: `${result.review_analysis.negative_percentage}%` }}></div>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-green-600">긍정 {result.review_analysis.positive_percentage.toFixed(0)}%</span>
                          <span className="text-gray-500">중립 {result.review_analysis.neutral_percentage.toFixed(0)}%</span>
                          <span className="text-red-600">부정 {result.review_analysis.negative_percentage.toFixed(0)}%</span>
                        </div>
                        <div className="bg-blue-50/50 p-3 rounded-md border border-blue-100 mt-4 space-y-3">
                          <p className="text-sm font-bold flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-blue-500" />AI 리뷰 분석: <span className={sentiment.color}>{sentiment.text}</span></p>
                          <div>
                            <p className="text-sm font-medium">주요 긍정 키워드</p>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {result.review_analysis.top_positive_keywords.length > 0 ? (
                                result.review_analysis.top_positive_keywords.map(kw => <Badge key={kw} variant="outline" className="text-green-700 border-green-200">{kw}</Badge>)
                              ) : (
                                <span className="text-xs text-gray-500">관련 키워드 없음</span>
                              )}
                            </div>
                          </div>
                          <div>
                            <p className="text-sm font-medium">주요 부정 키워드</p>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {result.review_analysis.top_negative_keywords.length > 0 ? (
                                result.review_analysis.top_negative_keywords.map(kw => <Badge key={kw} variant="outline" className="border-red-300 text-red-700 bg-transparent hover:bg-red-50">{kw}</Badge>)
                              ) : (
                                <span className="text-xs text-gray-500">관련 키워드 없음</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 