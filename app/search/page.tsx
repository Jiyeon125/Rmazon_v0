"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { BarChart, Search, AlertCircle, Loader2, Star, ThumbsUp, ThumbsDown, MessageSquareQuote, Sparkles, Tag } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"

// --- 데이터 타입 정의 ---

interface ReviewAnalysis {
  overall_sentiment: 'positive' | 'negative' | 'neutral';
  sentiment_distribution: { positive: number; neutral: number; negative: number; };
  top_keywords: { word: string; count: number }[];
  negative_concerns: string[];
  summary: string;
  review_count: number;
}

interface SimilarityResult {
  product_id: string;
  product_name: string;
  similarity: number;
  discounted_price: number;
  rating: number;
  rating_count: number;
  review_analysis: ReviewAnalysis;
}

// --- 컴포넌트 ---

const SentimentIcon = ({ sentiment }: { sentiment: ReviewAnalysis['overall_sentiment'] }) => {
  if (sentiment === 'positive') return <ThumbsUp className="w-5 h-5 text-green-500" />;
  if (sentiment === 'negative') return <ThumbsDown className="w-5 h-5 text-red-500" />;
  return <MessageSquareQuote className="w-5 h-5 text-gray-500" />;
};

export default function SearchPage() {
  // --- 상태 관리 ---
  const [description, setDescription] = useState("A high-quality, wireless headphone with noise-cancelling feature and long battery life. Comes with a carrying case.");
  const [price, setPrice] = useState("199.99");
  const [discount, setDiscount] = useState("15");
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SimilarityResult[]>([]);

  // --- 이벤트 핸들러 ---
  const handleSearch = async () => {
    const priceNum = parseFloat(price);
    const discountNum = parseFloat(discount);

    if (!description.trim() || isNaN(priceNum) || isNaN(discountNum)) {
      setError("모든 입력 필드를 올바르게 채워주세요.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/search-similarity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: description,
          price: priceNum,
          discount_percentage: discountNum,
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
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="description">상품 설명</Label>
                <Textarea 
                  id="description" 
                  placeholder="예: 고품질 무선 노이즈캔슬링 헤드폰..." 
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={5}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="price">정가 (₹)</Label>
                <Input 
                  id="price" 
                  type="number"
                  placeholder="예: 199.99" 
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="discount">할인율 (%)</Label>
                <Input 
                  id="discount" 
                  type="number"
                  placeholder="예: 15"
                  value={discount}
                  onChange={(e) => setDiscount(e.target.value)}
                />
              </div>
            </CardContent>
          </Card>
          
          <Button onClick={handleSearch} disabled={isLoading} className="w-full py-6 text-lg">
            {isLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Search className="mr-2 h-5 w-5" />}
            {isLoading ? "분석 중..." : "유사 상품 분석"}
          </Button>

          {error && (
            <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              <p>{error}</p>
            </div>
          )}
        </div>

        {/* Right Panel: Analysis Results */}
        <div className="lg:col-span-2">
           <Card className="h-full">
            <CardHeader>
              <div className="flex items-center gap-3">
                  <BarChart className="w-6 h-6 text-gray-500" />
                  <CardTitle>분석 결과</CardTitle>
              </div>
               <CardDescription>
                {results.length > 0 ? `입력한 상품과 유사한 상위 ${results.length}개 상품 분석` : "가상 상품 정보를 입력하고 분석을 실행하면 결과가 여기에 표시됩니다."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex flex-col items-center justify-center h-96 border-2 border-dashed rounded-lg bg-gray-50">
                  <Loader2 className="h-10 w-10 text-gray-400 animate-spin" />
                  <p className="mt-4 text-muted-foreground">유사 상품을 분석하고 있습니다...</p>
                </div>
              ) : results.length > 0 ? (
                <div className="space-y-6">
                  {results.map((product, index) => (
                    <Card key={product.product_id} className="p-4 overflow-hidden">
                      <div className="flex justify-between items-start mb-3">
                        <h3 className="font-semibold text-lg flex-1 pr-4">{index + 1}. {product.product_name}</h3>
                        <Badge variant="secondary" className="whitespace-nowrap text-base py-1 px-3">
                          유사도: {(product.similarity * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      
                      {/* --- 기본 정보 --- */}
                      <div className="grid grid-cols-3 gap-4 text-sm text-muted-foreground border-b pb-4 mb-4">
                         <div className="space-y-1">
                           <p className="font-medium text-gray-700">할인 적용가</p>
                           <p className="text-lg font-bold text-gray-900">₹{product.discounted_price.toLocaleString()}</p>
                         </div>
                         <div className="space-y-1">
                            <p className="font-medium text-gray-700">평점</p>
                            <p className="flex items-center gap-1 text-lg font-bold text-gray-900"><Star className="w-5 h-5 text-yellow-500 fill-yellow-400" />{product.rating.toFixed(1)}</p>
                         </div>
                         <div className="space-y-1">
                            <p className="font-medium text-gray-700">리뷰 수</p>
                            <p className="text-lg font-bold text-gray-900">{product.rating_count.toLocaleString()}</p>
                         </div>
                      </div>

                      {/* --- 고급 리뷰 분석 --- */}
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                           <SentimentIcon sentiment={product.review_analysis.overall_sentiment} />
                           <h4 className="font-semibold text-md">리뷰 종합: <span className={
                               product.review_analysis.overall_sentiment === 'positive' ? 'text-green-600' : 
                               product.review_analysis.overall_sentiment === 'negative' ? 'text-red-600' : 'text-gray-600'
                             }>{product.review_analysis.overall_sentiment}</span>
                           </h4>
                        </div>
                        
                        {/* 감성 분포 */}
                        <div>
                          <p className="text-sm font-medium mb-1">감성 분포 (총 {product.review_analysis.review_count}개)</p>
                          <div className="w-full bg-gray-200 rounded-full h-2.5 flex overflow-hidden">
                            <div className="bg-green-500 h-2.5" style={{ width: `${(product.review_analysis.sentiment_distribution.positive / product.review_analysis.review_count) * 100}%` }}></div>
                            <div className="bg-gray-400 h-2.5" style={{ width: `${(product.review_analysis.sentiment_distribution.neutral / product.review_analysis.review_count) * 100}%` }}></div>
                            <div className="bg-red-500 h-2.5" style={{ width: `${(product.review_analysis.sentiment_distribution.negative / product.review_analysis.review_count) * 100}%` }}></div>
                          </div>
                        </div>

                        {/* 주요 키워드 */}
                        <div>
                           <p className="text-sm font-medium mb-2 flex items-center gap-1.5"><Tag className="w-4 h-4" />주요 키워드</p>
                           <div className="flex flex-wrap gap-2">
                            {product.review_analysis.top_keywords.map(kw => (
                              <Badge key={kw.word} variant="outline">{kw.word} ({kw.count})</Badge>
                            ))}
                           </div>
                        </div>

                        {/* 부정적 우려사항 */}
                        {product.review_analysis.negative_concerns.length > 0 && (
                          <div>
                            <p className="text-sm font-medium mb-2 flex items-center gap-1.5"><ThumbsDown className="w-4 h-4" />부정적 우려사항</p>
                            <div className="space-y-2 text-sm bg-red-50/50 p-3 rounded-md border border-red-100">
                              {product.review_analysis.negative_concerns.map((concern, i) => (
                                <p key={i} className="text-red-900 line-clamp-2">" ... {concern} ... "</p>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {/* AI 종합 분석 */}
                        <div className="bg-blue-50/50 p-3 rounded-md border border-blue-100">
                           <p className="text-sm font-medium mb-1.5 flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-blue-500" />AI 종합 분석</p>
                           <p className="text-sm text-blue-900">{product.review_analysis.summary}</p>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-96 border-2 border-dashed rounded-lg bg-gray-50">
                  <Search className="h-10 w-10 text-gray-400" />
                  <p className="mt-4 text-muted-foreground text-center">분석할 가상 상품의 정보를 입력하고<br/>'유사 상품 분석' 버튼을 눌러주세요.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
} 