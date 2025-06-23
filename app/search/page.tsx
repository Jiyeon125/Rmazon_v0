"use client"

import { useState, useEffect, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { BarChart, Search, AlertCircle, Loader2, Star, ThumbsUp, ThumbsDown, MessageSquareQuote, Sparkles, Tag, ChevronsUpDown } from "lucide-react"
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
    const fullCategory = cat1 && cat2 && cat3 ? `${cat1} | ${cat2} | ${cat3}` : null;
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
  }, [cat1, cat2, cat3]);
  
  // 파생 상태 (Memoization)
  const cat1Options = useMemo(() => Object.keys(hierarchicalCategories), [hierarchicalCategories]);
  const cat2Options = useMemo(() => cat1 ? Object.keys((hierarchicalCategories as any)[cat1] || {}) : [], [hierarchicalCategories, cat1]);
  const cat3Options = useMemo(() => cat1 && cat2 ? Object.keys((hierarchicalCategories as any)[cat1]?.[cat2] || {}) : [], [hierarchicalCategories, cat1, cat2]);

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
    const fullCategory = cat1 && cat2 && cat3 ? `${cat1} | ${cat2} | ${cat3}` : null;
    if (!description.trim() || !fullCategory) {
      setError("상품 설명과 카테고리를 모두 입력해주세요.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResults([]);

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

    } catch (err) {
       setError(err instanceof Error ? err.message : "유사도 분석 중 오류 발생");
    } finally {
      setIsLoading(false);
    }
  };

  const fullCategorySelected = cat1 && cat2 && cat3;

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
                  <Select onValueChange={handleCat2Change} value={cat2} disabled={!cat1 || isDataLoading}>
                    <SelectTrigger><SelectValue placeholder="중분류" /></SelectTrigger>
                    <SelectContent>{cat2Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                  </Select>
                  <Select onValueChange={setCat3} value={cat3} disabled={!cat2 || isDataLoading}>
                    <SelectTrigger><SelectValue placeholder="소분류" /></SelectTrigger>
                    <SelectContent>{cat3Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
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

          {!isLoading && results.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full bg-white rounded-lg border-2 border-dashed">
              <Sparkles className="w-16 h-16 text-gray-300 mb-4" />
              <h3 className="text-xl font-semibold text-gray-600">분석 결과가 여기에 표시됩니다.</h3>
              <p className="text-gray-400 mt-1">좌측에서 상품 정보를 입력하고 분석 버튼을 눌러주세요.</p>
            </div>
          )}

          {!isLoading && results.length > 0 && (
            <div className="space-y-6">
              {results.map((result, index) => (
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
                  <CardContent className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-4">
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
                     <div className="space-y-4">
                        <h4 className="font-semibold">리뷰 종합 분석</h4>
                        <div className="space-y-2">
                          <Progress value={result.review_analysis.positive_percentage} className="h-2" />
                          <div className="flex justify-between text-xs text-muted-foreground">
                             <span>긍정 {result.review_analysis.positive_percentage.toFixed(0)}%</span>
                             <span>중립 {result.review_analysis.neutral_percentage.toFixed(0)}%</span>
                             <span>부정 {result.review_analysis.negative_percentage.toFixed(0)}%</span>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm font-medium">주요 긍정 키워드</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {result.review_analysis.top_positive_keywords.map(kw => <Badge key={kw} variant="outline" className="text-green-700 border-green-200">{kw}</Badge>)}
                          </div>
                        </div>
                        <div>
                          <p className="text-sm font-medium">주요 부정 키워드</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {result.review_analysis.top_negative_keywords.map(kw => <Badge key={kw} variant="destructive">{kw}</Badge>)}
                          </div>
                        </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 