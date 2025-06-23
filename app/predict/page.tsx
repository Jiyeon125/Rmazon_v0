"use client"

import { useState, useEffect, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart3, Star, Loader2, AlertCircle, PackageSearch, MessageSquare } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"

// --- 데이터 타입 정의 ---
interface DistributionBin {
  name: string;
  count: number;
}
interface CategoryStats {
  min_price: number;
  max_price: number;
  min_review_count: number;
  max_review_count: number;
  price_distribution: DistributionBin[];
  review_count_distribution: DistributionBin[];
  rating_distribution: DistributionBin[];
}
interface PredictionResult {
  predicted_star: number;
  predicted_review_count: number;
  price_percentile: number;
  review_count_percentile: number;
  rating_percentile: number;
}
interface HierarchicalCategories {
  [key: string]: HierarchicalCategories | {};
}

const API_BASE_URL = "http://127.0.0.1:8000";

// --- Helper Functions for Chart Data ---

// For binned data like price and review count
const processBinnedDistribution = (
  distribution: DistributionBin[], 
  userValue: number | null, 
  predictedValue: number | null
) => {
  if (!distribution || distribution.length === 0) return [];

  return distribution.map((bin, index) => {
    const [start, end] = bin.name.split('-').map(s => parseInt(s.replace(/,/g, '')));
    
    // 마지막 bin인 경우, end 값을 포함하여 확인
    const isLastBin = index === distribution.length - 1;

    const isUserInRange = userValue !== null && (
      isLastBin
        ? userValue >= start && userValue <= end
        : userValue >= start && userValue < end
    );

    const isPredictedInRange = predictedValue !== null && (
      isLastBin
        ? predictedValue >= start && predictedValue <= end
        : predictedValue >= start && predictedValue < end
    );

    return {
      ...bin,
      isUser: isUserInRange,
      isPredicted: isPredictedInRange,
    };
  });
};

// For single value data like rating
const processRatingDistribution = (
  distribution: DistributionBin[],
  predictedValue: number | null
) => {
  if (!distribution || distribution.length === 0) return [];

  return distribution.map(bin => {
    // Exact match for rating
    const isPredicted = predictedValue !== null && parseFloat(bin.name) === predictedValue;
    return {
      ...bin,
      isUser: false, // User doesn't input a rating
      isPredicted,
    };
  });
};

export default function PredictPage() {
  // 상태 관리
  const [price, setPrice] = useState("");
  const [hierarchicalCategories, setHierarchicalCategories] = useState<HierarchicalCategories>({});
  const [cat1, setCat1] = useState("");
  const [cat2, setCat2] = useState("");
  const [cat3, setCat3] = useState("");
  const [productCount, setProductCount] = useState<number | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [isMetaLoading, setIsMetaLoading] = useState(false); // 카테고리, 상품 수 로딩
  const [error, setError] = useState<string | null>(null);
  
  const [stats, setStats] = useState<CategoryStats | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

  // 계층형 카테고리 데이터 로드
  useEffect(() => {
    const fetchHierarchicalCategories = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/hierarchical-categories`);
        if (!response.ok) throw new Error("카테고리 목록을 불러오는 데 실패했습니다.");
        const data = await response.json();
        setHierarchicalCategories(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "카테고리 로드 중 오류 발생");
      }
    };
    fetchHierarchicalCategories();
  }, []);

  // 마지막 카테고리 선택 시 상품 수 및 통계 정보 로드
  useEffect(() => {
    if (!cat1 || !cat2 || !cat3) {
      setProductCount(null);
      setStats(null);
      return;
    }
    
    const fullCategory = `${cat1} | ${cat2} | ${cat3}`;

    const fetchMetadata = async () => {
      setIsMetaLoading(true);
      setError(null);
      setPredictionResult(null);
      try {
        // 상품 수 조회
        const countResponse = await fetch(`${API_BASE_URL}/product-count?category=${encodeURIComponent(fullCategory)}`);
        if (!countResponse.ok) throw new Error("상품 수를 불러오는 데 실패했습니다.");
        const countData = await countResponse.json();
        setProductCount(countData);

        // 통계 정보 조회
        const statsResponse = await fetch(`${API_BASE_URL}/category-stats?category=${encodeURIComponent(fullCategory)}`);
        if (!statsResponse.ok) throw new Error("카테고리 통계 정보를 불러오는 데 실패했습니다.");
        const statsData: CategoryStats = await statsResponse.json();
        setStats(statsData);
        // 기본 가격 설정
        setPrice(String(Math.round((statsData.min_price + statsData.max_price) / 2)));

      } catch (err) {
        setError(err instanceof Error ? err.message : "메타데이터 로드 중 오류 발생");
        setProductCount(null);
        setStats(null);
      } finally {
        setIsMetaLoading(false);
      }
    };

    fetchMetadata();
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

  // 예측 실행 핸들러
  const handlePredict = async () => {
    const fullCategory = `${cat1} | ${cat2} | ${cat3}`;
    if (!price || !fullCategory) {
      setError("모든 값을 입력해주세요.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ price: parseFloat(price), category: fullCategory }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "API 요청에 실패했습니다.");
      }

      const data: PredictionResult = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }
  
  const fullCategorySelected = cat1 && cat2 && cat3;

  // Processed data for charts using the helper function
  const processedPriceDistribution = useMemo(() => 
    stats ? processBinnedDistribution(stats.price_distribution, parseFloat(price), null) : [],
    [stats, price]
  );

  const processedReviewDistribution = useMemo(() =>
    stats && predictionResult ? processBinnedDistribution(stats.review_count_distribution, null, predictionResult.predicted_review_count) : (stats ? processBinnedDistribution(stats.review_count_distribution, null, null) : []),
    [stats, predictionResult]
  );

  const processedRatingDistribution = useMemo(() =>
    stats && predictionResult ? processRatingDistribution(stats.rating_distribution, predictionResult.predicted_star) : (stats ? processRatingDistribution(stats.rating_distribution, null) : []),
    [stats, predictionResult]
  );

  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        <Card>
          <CardHeader>
             <div className="flex items-center gap-3">
              <div className="p-3 bg-purple-100 rounded-full">
                <BarChart3 className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <CardTitle className="text-2xl font-bold">판매 지표 예측</CardTitle>
                <CardDescription>상품 정보를 입력하여 예상 평점과 리뷰 수를 확인해보세요.</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6 pt-6">
            <div className="space-y-2">
              <Label>1. 분석할 상품 카테고리</Label>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <Select onValueChange={handleCat1Change} value={cat1}>
                  <SelectTrigger><SelectValue placeholder="대분류" /></SelectTrigger>
                  <SelectContent>{cat1Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                </Select>
                <Select onValueChange={handleCat2Change} value={cat2} disabled={!cat1}>
                  <SelectTrigger><SelectValue placeholder="중분류" /></SelectTrigger>
                  <SelectContent>{cat2Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                </Select>
                <Select onValueChange={setCat3} value={cat3} disabled={!cat2}>
                  <SelectTrigger><SelectValue placeholder="소분류" /></SelectTrigger>
                  <SelectContent>{cat3Options.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                </Select>
              </div>
            </div>
            
            {isMetaLoading && (
              <div className="flex items-center justify-center p-4">
                <Loader2 className="h-6 w-6 animate-spin" /> 
                <span className="ml-2">카테고리 정보 로딩 중...</span>
              </div>
            )}

            {fullCategorySelected && !isMetaLoading && productCount !== null && (
               <div className="p-3 bg-gray-50 rounded-md text-center">
                  <p className="text-sm text-gray-600">선택한 카테고리의 상품 수</p>
                  <p className="text-xl font-bold text-purple-700">{productCount.toLocaleString()} 개</p>
                </div>
            )}
            
            {stats && !isMetaLoading && (
              <div className="grid w-full items-center gap-2">
                <Label htmlFor="price">2. 가격 (Price)</Label>
                <p className="text-xs text-muted-foreground">
                  이 카테고리의 가격 범위: ₹{stats.min_price.toLocaleString()} ~ ₹{stats.max_price.toLocaleString()}
                </p>
                <Input id="price" type="number" value={price} onChange={(e) => setPrice(e.target.value)} />
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button onClick={handlePredict} disabled={isLoading || isMetaLoading || !stats} className="w-full">
              {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Star className="mr-2 h-4 w-4" />}
              {isLoading ? "예측 중..." : "결과 분석 및 예측"}
            </Button>
          </CardFooter>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>종합 분석 결과</CardTitle>
            <CardDescription>입력한 정보를 바탕으로 시장 내 위치와 예상 지표를 확인합니다.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            {isLoading && (
              <div className="flex items-center justify-center p-4 h-64">
                <Loader2 className="h-8 w-8 animate-spin text-purple-600" />
                <span className="ml-3 text-lg">결과 분석 중...</span>
              </div>
            )}
            
            {error && (
              <div className="mt-4 p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                <p>{error}</p>
              </div>
            )}

            {!isLoading && !predictionResult && !error &&(
                 <div className="flex flex-col items-center justify-center text-center text-gray-500 h-64">
                    <PackageSearch size={48} className="mb-4" />
                    <h3 className="text-lg font-semibold">예측 결과 대기 중</h3>
                    <p className="text-sm">좌측에서 카테고리와 가격을 입력하고<br/>예측 버튼을 눌러주세요.</p>
                </div>
            )}

            {!isLoading && !error && predictionResult && (
              <>
                <div className="grid grid-cols-2 gap-4 text-center">
                   <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-muted-foreground flex items-center justify-center gap-1"><Star size={14}/> 예상 평점</p>
                    <p className="text-3xl font-bold text-purple-700">{predictionResult.predicted_star.toFixed(2)}</p>
                    <p className="text-xs text-muted-foreground">시장 내 상위 {(100 - predictionResult.rating_percentile).toFixed(1)}%</p>
                  </div>
                   <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-muted-foreground flex items-center justify-center gap-1"><MessageSquare size={14}/> 예상 리뷰 수</p>
                    <p className="text-3xl font-bold text-purple-700">{Math.round(predictionResult.predicted_review_count).toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">시장 내 상위 {(100 - predictionResult.review_count_percentile).toFixed(1)}%</p>
                  </div>
                </div>

                {stats && (
                  <div className="space-y-8 pt-4">
                    <h3 className="text-lg font-semibold border-b pb-2">시장 내 분포도 분석</h3>
                     <div className="space-y-2">
                      <h4 className="font-semibold">가격 분포</h4>
                      <p className="text-xs text-muted-foreground">입력하신 가격은 시장 내에서 상위 {predictionResult.price_percentile.toFixed(1)}%에 위치합니다.</p>
                      <div className="w-full h-40">
                        <ResponsiveContainer>
                          <BarChart data={processedPriceDistribution}>
                            <XAxis dataKey="name" fontSize={10} tick={{ fill: '#6b7280' }} />
                            <YAxis fontSize={10} tick={{ fill: '#6b7280' }} />
                            <Tooltip
                              contentStyle={{ fontSize: '12px', padding: '4px 8px', borderRadius: '0.5rem' }} 
                              labelStyle={{ fontWeight: 'bold' }}
                              formatter={(value, name, props) => [`${props.payload.isUser ? '나의 가격' : '상품 수'}: ${value}`, null]}
                            />
                            <Bar dataKey="count">
                              {processedPriceDistribution.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.isUser ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.2)"} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                     <div className="space-y-2">
                      <h4 className="font-semibold">리뷰 수 분포</h4>
                      <p className="text-xs text-muted-foreground">예상 리뷰 수는 시장 내에서 상위 {predictionResult.review_count_percentile.toFixed(1)}%에 위치합니다.</p>
                      <div className="w-full h-40">
                        <ResponsiveContainer>
                          <BarChart data={processedReviewDistribution}>
                            <XAxis dataKey="name" fontSize={10} tick={{ fill: '#6b7280' }} />
                            <YAxis fontSize={10} tick={{ fill: '#6b7280' }} />
                            <Tooltip
                              contentStyle={{ fontSize: '12px', padding: '4px 8px', borderRadius: '0.5rem' }} 
                              labelStyle={{ fontWeight: 'bold' }}
                               formatter={(value, name, props) => [`${props.payload.isPredicted ? '예상 리뷰 수' : '상품 수'}: ${value}`, null]}
                            />
                            <Bar dataKey="count">
                              {processedReviewDistribution.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.isPredicted ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.2)"} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                     <div className="space-y-2">
                      <h4 className="font-semibold">평점 분포</h4>
                      <p className="text-xs text-muted-foreground">예상 평점은 시장 내에서 상위 {predictionResult.rating_percentile.toFixed(1)}%에 위치합니다.</p>
                      <div className="w-full h-40">
                         <ResponsiveContainer>
                          <BarChart data={processedRatingDistribution}>
                            <XAxis dataKey="name" fontSize={10} tick={{ fill: '#6b7280' }} />
                            <YAxis fontSize={10} tick={{ fill: '#6b7280' }} />
                             <Tooltip
                              contentStyle={{ fontSize: '12px', padding: '4px 8px', borderRadius: '0.5rem' }} 
                              labelStyle={{ fontWeight: 'bold' }}
                               formatter={(value, name, props) => [`${props.payload.isPredicted ? '예상 평점' : '상품 수'}: ${value}`, null]}
                            />
                            <Bar dataKey="count">
                              {processedRatingDistribution.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.isPredicted ? "hsl(var(--primary))" : "hsl(var(--primary) / 0.2)"} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}