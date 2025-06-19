"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart, Search, AlertCircle, Loader2 } from "lucide-react"

// 임시 상품 데이터 (나중에 API로 대체)
const tempProducts = [
  { id: "B0B942F24V", name: "Product A - Temporary" },
  { id: "B0B195C686", name: "Product B - Temporary" },
  { id: "B09W9F4332", name: "Product C - Temporary" },
]

export default function SearchPage() {
  const [products, setProducts] = useState<{ id: string, name: string }[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // TODO: 나중에 백엔드 API에서 상품 목록을 가져오는 로직 추가
  useEffect(() => {
    // 임시로 하드코딩된 데이터를 사용합니다.
    setProducts(tempProducts);
  }, []);

  const handleSearch = async () => {
    if (!selectedProduct) {
      setError("분석할 상품을 선택해주세요.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    // TODO: 유사도 분석 결과 상태 초기화

    console.log("Searching for product:", selectedProduct);
    // TODO: 백엔드에 유사도 분석 요청 보내는 로직 추가

    // 임시로 로딩 시뮬레이션
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-full">
      <header className="bg-white border-b p-4">
        <h1 className="text-xl font-semibold">유사 상품 탐색</h1>
      </header>

      <div className="flex-1 overflow-y-auto p-4 md:p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Panel: Controls */}
        <div className="md:col-span-1 flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>상품 선택</CardTitle>
              <CardDescription>분석할 기준 상품을 선택하세요.</CardDescription>
            </CardHeader>
            <CardContent>
              <Select onValueChange={setSelectedProduct}>
                <SelectTrigger>
                  <SelectValue placeholder="상품을 선택하세요..." />
                </SelectTrigger>
                <SelectContent>
                  {products.map((product) => (
                    <SelectItem key={product.id} value={product.id}>
                      {product.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          <Button onClick={handleSearch} disabled={isLoading} className="w-full">
            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
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
        <div className="md:col-span-2">
          <Card className="h-full">
            <CardHeader>
              <div className="flex items-center gap-3">
                  <BarChart className="w-6 h-6 text-gray-500" />
                  <CardTitle>분석 결과</CardTitle>
              </div>
              <CardDescription>
                선택한 상품과 유사한 제품들의 분석 결과가 여기에 표시됩니다.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center h-64 border-2 border-dashed rounded-lg">
                <p className="text-muted-foreground">분석할 상품을 선택하고 분석을 실행해주세요.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
} 