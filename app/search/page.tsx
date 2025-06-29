"use client"

import { useState, useEffect, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { BarChart, Search, AlertCircle, Loader2, Star, ThumbsUp, ThumbsDown, MessageSquareQuote, Sparkles, Tag, PackageSearch, AlertTriangle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
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

const ReviewAnalysisCard = ({ analysis }: { analysis: ReviewAnalysis }) => {
  const sentiment = getSentimentSummary(analysis);

  return (
    <div className="space-y-4">
      <h4 className="font-semibold">리뷰 종합 분석</h4>
      <div className="w-full bg-gray-200 rounded-full h-2.5 flex overflow-hidden">
        <div className="bg-green-500 h-2.5" style={{ width: `${analysis.positive_percentage}%` }}></div>
        <div className="bg-gray-400 h-2.5" style={{ width: `${analysis.neutral_percentage}%` }}></div>
        <div className="bg-red-500 h-2.5" style={{ width: `${analysis.negative_percentage}%` }}></div>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-green-600">긍정 {analysis.positive_percentage.toFixed(0)}%</span>
        <span className="text-gray-500">중립 {analysis.neutral_percentage.toFixed(0)}%</span>
        <span className="text-red-600">부정 {analysis.negative_percentage.toFixed(0)}%</span>
      </div>
      <div className="bg-blue-50/50 p-3 rounded-md border border-blue-100 mt-4 space-y-3">
        <p className="text-sm font-bold flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-blue-500" />AI 리뷰 분석: <span className={sentiment.color}>{sentiment.text}</span></p>
        <div>
          <p className="text-sm font-medium">주요 긍정 키워드</p>
          <div className="flex flex-wrap gap-1 mt-1">
            {analysis.top_positive_keywords.length > 0 ? (
              analysis.top_positive_keywords.map(kw => <Badge key={kw} variant="outline" className="text-green-700 border-green-200">{kw}</Badge>)
            ) : (
              <span className="text-xs text-gray-500">관련 키워드 없음</span>
            )}
          </div>
        </div>
        <div>
          <p className="text-sm font-medium">주요 부정 키워드</p>
          <div className="flex flex-wrap gap-1 mt-1">
            {analysis.top_negative_keywords.length > 0 ? (
              analysis.top_negative_keywords.map(kw => <Badge key={kw} variant="outline" className="border-red-300 text-red-700 bg-transparent hover:bg-red-50">{kw}</Badge>)
            ) : (
              <span className="text-xs text-gray-500">관련 키워드 없음</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ResultsDisplay = ({ results }: { results: SimilarityResult[] }) => {
  if (!results || results.length === 0) return null;

  return (
    <div className="space-y-6">
      {results.map((result, index) => (
        <Card key={result.product_id} className="overflow-hidden transition-all hover:shadow-lg">
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
              <div className="text-right flex-shrink-0 ml-4">
                <div className="text-sm text-muted-foreground">유사도</div>
                <div className="text-2xl font-bold text-purple-600">{(result.similarity * 100).toFixed(1)}%</div>
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
                  <Star className="w-4 h-4 text-yellow-400 fill-current" />
                  <span>{result.rating.toFixed(1)}</span>
                </div>
              </div>
              <div className="flex justify-between items-center text-sm p-2 bg-gray-50 rounded-md">
                <span className="text-muted-foreground">리뷰 수</span>
                <span>{result.rating_count.toLocaleString()} 개</span>
              </div>
            </div>
            <div className="md:col-span-2">
              <ReviewAnalysisCard analysis={result.review_analysis} />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

const SearchPageContent = ({
  isLoading,
  error,
  similarityWarning,
  results,
  hasSearched
}: {
  isLoading: boolean;
  error: string | null;
  similarityWarning: string | null;
  results: SimilarityResult[];
  hasSearched: boolean;
}) => {
  if (!hasSearched) {
    return (
      <Card className="h-full flex flex-col items-center justify-center p-8 text-center bg-secondary/50 border-dashed">
        <CardHeader>
           <div className="mx-auto bg-primary/10 p-4 rounded-full">
             <Sparkles className="w-12 h-12 text-primary" />
           </div>
          <CardTitle className="mt-4">유사 상품 분석 시작하기</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground max-w-md">
            입력 패널에서 가상 상품의 카테고리, 설명, 가격 등을 설정한 후 '유사 상품 분석' 버튼을 눌러주세요.
            입력된 정보와 유사한 실제 상품 목록 및 리뷰 분석 결과를 확인할 수 있습니다.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 text-center p-8">
        <Loader2 className="w-12 h-12 animate-spin text-primary" />
        <h3 className="text-xl font-semibold">AI가 유사 상품을 분석하고 있습니다...</h3>
        <p className="text-muted-foreground">잠시만 기다려주세요. 데이터의 의미를 깊게 파고드는 중입니다.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
         <Card className="border-destructive bg-destructive/10">
          <CardHeader>
            <div className="flex items-center justify-center gap-2">
              <AlertCircle className="w-8 h-8 text-destructive" />
              <CardTitle className="text-destructive">오류 발생</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-lg">{error}</p>
            <p className="text-sm text-muted-foreground mt-2">
              백엔드 서버가 실행 중인지 확인하거나, 입력값을 조정한 후 다시 시도해 주세요.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (hasSearched && results.length === 0) {
    return (
      <div className="p-8 text-center">
        <Card className="bg-secondary/50">
           <CardHeader>
            <div className="flex items-center justify-center gap-3">
              <PackageSearch className="w-8 h-8 text-muted-foreground" />
              <CardTitle>검색 결과 없음</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
             <p>입력하신 조건과 일치하는 상품을 찾을 수 없습니다.</p>
             <p className="text-sm text-muted-foreground mt-2">
              상품 설명이나 카테고리를 변경하여 다시 시도해 보세요.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {similarityWarning && (
        <div className="space-y-2">
          {similarityWarning.split('|').map((warning, index) => (
            <div key={index} className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800 p-4 rounded-md flex items-start">
              <AlertTriangle className="w-5 h-5 mr-3 flex-shrink-0" />
              <p>{warning.trim().replace(/^⚠️ 주의: /, '')}</p>
            </div>
          ))}
        </div>
      )}
      <ResultsDisplay results={results} />
    </div>
  );
};

export default function SearchPage() {
  // --- 상태 관리 ---
  const [description, setDescription] = useState("A durable, high-speed USB-C to USB-A cable with a braided nylon exterior for enhanced longevity. Supports fast charging and data transfer, compatible with all modern devices.");
  
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
  const [searchId, setSearchId] = useState(0);

  // --- 파생 상태 (Memoization) ---
  const { fullCategory, fullCategorySelected } = useMemo(() => {
    if (!cat1 || Object.keys(hierarchicalCategories).length === 0) {
      return { fullCategory: null, fullCategorySelected: false };
    }
    
    const cat1Data = (hierarchicalCategories as any)[cat1];
    if (!cat1Data) return { fullCategory: [cat1].join(" | "), fullCategorySelected: true };

    const cat2Options = Object.keys(cat1Data);
    if (cat2Options.length > 0 && !cat2) return { fullCategory: null, fullCategorySelected: false };

    if (!cat2) {
      return { fullCategory: [cat1].join(" | "), fullCategorySelected: true };
    }

    const cat2Data = cat1Data[cat2];
    if (!cat2Data) return { fullCategory: [cat1, cat2].filter(Boolean).join(" | "), fullCategorySelected: true };

    const cat3Options = Object.keys(cat2Data);
    if (cat3Options.length > 0 && !cat3) return { fullCategory: null, fullCategorySelected: false };

    return { fullCategory: [cat1, cat2, cat3].filter(Boolean).join(" | "), fullCategorySelected: true };
  }, [cat1, cat2, cat3, hierarchicalCategories]);
  
  const cat1Options = useMemo(() => Object.keys(hierarchicalCategories), [hierarchicalCategories]);
  const cat2Options = useMemo(() => cat1 ? Object.keys((hierarchicalCategories as any)[cat1] || {}) : [], [hierarchicalCategories, cat1]);
  const cat3Options = useMemo(() => cat1 && cat2 ? Object.keys((hierarchicalCategories as any)[cat1]?.[cat2] || {}) : [], [hierarchicalCategories, cat1, cat2]);
  
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
    if (!description.trim() || !fullCategorySelected) {
      setError("상품 설명과 모든 카테고리를 선택해주세요.");
      return;
    }
    
    setSearchId(prevId => prevId + 1);
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

      // 유사도 경고 체크 (단위를 %로 수정)
      if (data.length > 0) {
        const warnings: string[] = [];
        const maxSim = Math.max(...data.map(r => r.similarity));
        const avgSim = data.reduce((acc, r) => acc + r.similarity, 0) / data.length;
        
        if (maxSim < 60) {
          warnings.push("⚠️ 주의: 최고 유사도가 60% 미만입니다. 입력한 설명과 매우 유사한 제품이 거의 없으며, 유사 제품 목록의 정확도가 낮을 수 있습니다.");
        }
        if (avgSim < 50) {
          warnings.push("⚠️ 주의: 평균 유사도가 50% 미만입니다. 입력한 설명이 다른 제품들과 전반적으로 크게 다르며, 입력 정보를 조정하여 더 정확한 결과를 얻을 수 있습니다.");
        }
        setSimilarityWarning(warnings.length > 0 ? warnings.join(' | ') : null);
      } else {
        setSimilarityWarning(null);
      }

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

      <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>가상 상품 정보 입력</CardTitle>
                <CardDescription>분석하고 싶은 가상 상품의 정보를 입력하세요.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>카테고리</Label>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    <Select onValueChange={handleCat1Change} value={cat1}>
                      <SelectTrigger disabled={isDataLoading}><SelectValue placeholder="대분류" /></SelectTrigger>
                      <SelectContent>
                        {cat1Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                      </SelectContent>
                    </Select>
                    <Select onValueChange={handleCat2Change} value={cat2} disabled={!cat1 || cat2Options.length === 0}>
                      <SelectTrigger disabled={isDataLoading}><SelectValue placeholder={cat2Options.length > 0 ? "중분류" : "(선택)"} /></SelectTrigger>
                      <SelectContent>
                        {cat2Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                      </SelectContent>
                    </Select>
                    <Select onValueChange={setCat3} value={cat3} disabled={!cat2 || cat3Options.length === 0}>
                      <SelectTrigger disabled={isDataLoading}><SelectValue placeholder={cat3Options.length > 0 ? "소분류" : "(선택)"} /></SelectTrigger>
                      <SelectContent>
                        {cat3Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  {productCount !== null && (
                    <p className="text-sm text-muted-foreground pt-1">
                      현재 카테고리 내 상품 수: <span className="font-semibold text-primary">{productCount}</span>개입니다.
                    </p>
                  )}
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">상품 설명</Label>
                  <Textarea
                    id="description"
                    placeholder="제품의 주요 기능, 특징, 재질 등을 입력하세요..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={5}
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label htmlFor="price-slider">정가 (₹)</Label>
                    <span className="font-semibold text-primary">₹{Math.round(price).toLocaleString()}</span>
                  </div>
                  <Slider
                    id="price-slider"
                    min={priceRange.min_price}
                    max={priceRange.max_price}
                    step={priceRange.max_price > priceRange.min_price ? (priceRange.max_price - priceRange.min_price) / 100 : 1}
                    value={[price]}
                    onValueChange={(value) => setPrice(value[0])}
                    disabled={isDataLoading}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>₹{Math.round(priceRange.min_price).toLocaleString()}</span>
                    <span>₹{Math.round(priceRange.max_price).toLocaleString()}</span>
                  </div>
                </div>
                <div className="space-y-2">
                   <div className="flex justify-between items-center">
                    <Label htmlFor="discount-slider">할인율 (%)</Label>
                    <span className="font-semibold text-primary">{discount}%</span>
                  </div>
                  <Slider
                    id="discount-slider"
                    min={0}
                    max={90}
                    step={5}
                    value={[discount]}
                    onValueChange={(value) => setDiscount(value[0])}
                  />
                  <div className="text-right text-sm">
                    <p>최종 가격: <span className="font-bold">₹{Math.round(price * (1 - discount/100)).toLocaleString()}</span></p>
                  </div>
                </div>
                <Button onClick={handleSearch} className="w-full" disabled={isLoading || !fullCategorySelected}>
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      분석 중...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-4 w-4" />
                      유사 상품 분석
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
          
          <div className="lg:col-span-2">
            <div key={searchId}>
              <SearchPageContent 
                isLoading={isLoading}
                error={error}
                similarityWarning={similarityWarning}
                results={results}
                hasSearched={hasSearched}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
} 