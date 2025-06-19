"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart, Search, AlertCircle, Loader2, Star } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"

interface Product {
  product_id: string;
  product_name: string;
}

interface SimilarityResult extends Product {
  similarity: number;
  discounted_price: number;
  rating: number;
  rating_count: number;
}

export default function SearchPage() {
  const [categories, setCategories] = useState<string[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetchingData, setIsFetchingData] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SimilarityResult[]>([]);
  const [categorySearchTerm, setCategorySearchTerm] = useState("");

  // 1. 초기 카테고리 목록 로드
  useEffect(() => {
    const fetchCategories = async () => {
      setIsFetchingData(true);
      setError(null);
      try {
        const response = await fetch("http://127.0.0.1:8000/categories");
        if (!response.ok) throw new Error("카테고리 목록 로딩 실패");
        const data = await response.json();
        setCategories(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "카테고리 로딩 중 오류");
      } finally {
        setIsFetchingData(false);
      }
    };
    fetchCategories();
  }, []);

  // 2. 카테고리 선택 시 상품 목록 로드
  useEffect(() => {
    if (!selectedCategory) {
      setProducts([]);
      setSelectedProductId(null);
      return;
    }

    const fetchProducts = async () => {
      setIsFetchingData(true);
      setError(null);
      setResults([]);
      try {
        const response = await fetch(`http://127.0.0.1:8000/products?category=${encodeURIComponent(selectedCategory)}`);
        if (!response.ok) throw new Error("상품 목록 로딩 실패");
        const data = await response.json();
        setProducts(data);
        setSelectedProductId(null); // 카테고리 변경 시 상품 선택 초기화
      } catch (err) {
        setError(err instanceof Error ? err.message : "상품 목록 로딩 중 오류");
      } finally {
        setIsFetchingData(false);
      }
    };
    fetchProducts();
  }, [selectedCategory]);

  const handleSearch = async () => {
    if (!selectedProductId) {
      setError("분석할 상품을 선택해주세요.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      const response = await fetch("http://127.0.0.1:8000/search-similarity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: selectedProductId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "유사도 분석에 실패했습니다.");
      }
      
      const data = await response.json();
      setResults(data);

    } catch (err) {
       setError(err instanceof Error ? err.message : "유사도 분석 중 오류 발생");
    } finally {
      setIsLoading(false);
    }
  };

  const filteredCategories = categories.filter(c => c.toLowerCase().includes(categorySearchTerm.toLowerCase()));
  const selectedProductName = products.find(p => p.product_id === selectedProductId)?.product_name || "선택된 상품 없음";

  return (
    <div className="flex flex-col h-full bg-gray-50/50">
      <header className="bg-white border-b p-4 shadow-sm sticky top-0 z-10">
        <h1 className="text-xl font-semibold">유사 상품 탐색</h1>
      </header>

      <div className="flex-1 overflow-y-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel: Controls */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>1. 카테고리 선택</CardTitle>
              <CardDescription>분석할 상품의 카테고리를 선택하세요.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Input
                placeholder="카테고리 검색..."
                value={categorySearchTerm}
                onChange={(e) => setCategorySearchTerm(e.target.value)}
                disabled={isFetchingData && categories.length === 0}
              />
              <div className="max-h-60 overflow-y-auto space-y-2 pr-2">
                {isFetchingData && categories.length === 0 ? (
                  <div className="flex items-center justify-center p-4">
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  </div>
                ) : filteredCategories.length > 0 ? (
                  filteredCategories.map((category) => (
                    <Button
                      key={category}
                      variant={selectedCategory === category ? "default" : "outline"}
                      className="w-full justify-start text-left h-auto whitespace-normal"
                      onClick={() => setSelectedCategory(category)}
                    >
                      {category}
                    </Button>
                  ))
                ) : (
                  <p className="text-muted-foreground text-sm text-center p-4">해당 카테고리가 없습니다.</p>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>2. 상품 선택</CardTitle>
              <CardDescription>분석할 기준 상품을 선택하세요.</CardDescription>
            </CardHeader>
            <CardContent>
              <Select
                onValueChange={setSelectedProductId}
                disabled={!selectedCategory || products.length === 0 || isFetchingData}
                value={selectedProductId || ""}
              >
                <SelectTrigger>
                  <SelectValue placeholder={!selectedCategory ? "카테고리를 먼저 선택하세요" : (isFetchingData ? "상품 로딩 중..." : "상품을 선택하세요...")} />
                </SelectTrigger>
                <SelectContent>
                  {products.map((product) => (
                    <SelectItem key={product.product_id} value={product.product_id}>
                      {product.product_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          <Button onClick={handleSearch} disabled={isLoading || isFetchingData || !selectedProductId} className="w-full py-6 text-lg">
            {isLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Search className="mr-2 h-5 w-5" />}
            {isLoading ? "분석 중..." : "유사도 분석 실행"}
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
                {results.length > 0 ? `'${selectedProductName}'와(과) 유사한 상위 ${results.length}개 상품` : "선택한 상품과 유사한 제품들의 분석 결과가 여기에 표시됩니다."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center h-64 border-2 border-dashed rounded-lg">
                  <Loader2 className="h-8 w-8 text-gray-400 animate-spin" />
                </div>
              ) : results.length > 0 ? (
                <div className="space-y-4">
                  {results.map((product, index) => (
                    <Card key={product.product_id} className="p-4">
                       <div className="flex justify-between items-start">
                         <h3 className="font-semibold text-md mb-2 flex-1 pr-4">{index + 1}. {product.product_name}</h3>
                         <Badge variant="secondary" className="whitespace-nowrap">
                           유사도: {(product.similarity * 100).toFixed(1)}%
                         </Badge>
                       </div>
                       <div className="grid grid-cols-3 gap-4 text-sm text-muted-foreground mt-2">
                          <div>
                            <p className="font-medium text-gray-700">가격</p>
                            <p>₹{product.discounted_price.toLocaleString()}</p>
                          </div>
                          <div>
                             <p className="font-medium text-gray-700">평점</p>
                             <p className="flex items-center gap-1"><Star className="w-4 h-4 text-yellow-500 fill-yellow-400" />{product.rating}</p>
                          </div>
                          <div>
                             <p className="font-medium text-gray-700">평점 수</p>
                             <p>{product.rating_count.toLocaleString()}</p>
                          </div>
                       </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-64 border-2 border-dashed rounded-lg">
                  <p className="text-muted-foreground">분석할 상품을 선택하고 분석을 실행해주세요.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
} 