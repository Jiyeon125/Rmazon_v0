"use client"

import type React from "react"

import { useState, useMemo, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Search,
  ShoppingCart,
  AlertTriangle,
  Star,
  Upload,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  ThumbsUp,
  ThumbsDown,
  TrendingUp,
  MessageSquare,
  BarChart3,
  Info,
} from "lucide-react"

// 고급 텍스트 분석 함수들 import
import { advancedReviewAnalysis } from "@/lib/text-analysis"

// Simple text similarity function
function calculateTextSimilarity(text1: string, text2: string): number {
  const words1 = text1.toLowerCase().split(/\s+/)
  const words2 = text2.toLowerCase().split(/\s+/)

  const intersection = words1.filter((word) => words2.includes(word))
  const union = [...new Set([...words1, ...words2])]

  return intersection.length / union.length
}

// Price similarity function
function calculatePriceSimilarity(price1: number, price2: number): number {
  const diff = Math.abs(price1 - price2)
  const avg = (price1 + price2) / 2
  return Math.max(0, 1 - diff / avg)
}

//review_title 기반 리뷰 수 계산
function countReviewsByTitle(product: any): number {
  if (!product.review_title || typeof product.review_title !== "string") return 0;

  const parts = product.review_title.split(",").map((part: string) => part.trim()).filter((part: string) => part.length > 0);
  return parts.length;
}

export default function ProductExplorer() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("")
  const [productDesc, setProductDesc] = useState("")
  const [actualPrice, setActualPrice] = useState(3000)
  const [discountPct, setDiscountPct] = useState([20])
  const [results, setResults] = useState<any[]>([])
  const [warnings, setWarnings] = useState<string[]>([])
  const [isSearching, setIsSearching] = useState(false)

  // 새로운 상태들 추가
  const [uploadedData, setUploadedData] = useState<any[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [fileInfo, setFileInfo] = useState<{ name: string; size: number; rows: number } | null>(null)
  const [showAllCategories, setShowAllCategories] = useState(false)

  // 리뷰 요약 관련 상태
  const [reviewSummaries, setReviewSummaries] = useState<{ [key: string]: any }>({})
  const [loadingSummaries, setLoadingSummaries] = useState<{ [key: string]: boolean }>({})

  // 현재 사용할 데이터 결정
  const currentData = uploadedData.length > 0 ? uploadedData : []

  // Get unique categories
  const categories = useMemo(() => {
    return [...new Set(currentData.map((p) => p.category))].sort()
  }, [currentData])

  // Filter categories based on search term
  const filteredCategories = useMemo(() => {
    return categories.filter((cat) => cat.toLowerCase().includes(searchTerm.toLowerCase()))
  }, [categories, searchTerm])

  // Calculate discounted price
  const discountedPrice = Math.round(actualPrice * (1 - discountPct[0] / 100))

  // CSV 파싱 함수
  const parseCSV = useCallback((csvText: string) => {
    const lines = csvText.split("\n")
    if (lines.length < 2) return []

    const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""))
    const data = []

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim()
      if (!line) continue

      const values: string[] = []
      let current = ""
      let inQuotes = false

      for (let j = 0; j < line.length; j++) {
        const char = line[j]
        if (char === '"') {
          inQuotes = !inQuotes
        } else if (char === "," && !inQuotes) {
          values.push(current.trim())
          current = ""
        } else {
          current += char
        }
      }
      values.push(current.trim())

      if (values.length >= headers.length) {
        const row: any = {}
        headers.forEach((header, index) => {
          let value = values[index] || ""
          value = value.replace(/^"|"$/g, "") // Remove quotes

          // 숫자 변환
          if (header.includes("price") || header.includes("percentage") || header.includes("rating")) {
            const numValue = Number.parseFloat(value.replace(/[₹,]/g, ""))
            row[header] = isNaN(numValue) ? 0 : numValue
          } else {
            row[header] = value
          }
        })

        // actual_price 계산 (없는 경우)
        if (!row.actual_price && row.discounted_price && row.discount_percentage) {
          row.actual_price = Math.round(row.discounted_price / (1 - row.discount_percentage / 100))
        }

        data.push(row)
      }
    }

    return data
  }, [])

  // 파일 업로드 핸들러
  const handleFileUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return

      if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("CSV 파일만 업로드 가능합니다.")
        return
      }

      setIsUploading(true)
      setUploadProgress(0)

      try {
        const text = await file.text()
        setUploadProgress(50)

        const parsedData = parseCSV(text)
        setUploadProgress(80)

        if (parsedData.length === 0) {
          throw new Error("파일을 파싱할 수 없습니다.")
        }

        // 필수 컬럼 확인
        const requiredColumns = ["product_name", "category", "about_product"]
        const firstRow = parsedData[0]
        const missingColumns = requiredColumns.filter((col) => !(col in firstRow))

        if (missingColumns.length > 0) {
          throw new Error(`필수 컬럼이 누락되었습니다: ${missingColumns.join(", ")}`)
        }

        setUploadedData(parsedData)
        setFileInfo({
          name: file.name,
          size: file.size,
          rows: parsedData.length,
        })
        setUploadProgress(100)

        // 기존 검색 결과 초기화
        setResults([])
        setSelectedCategory("")

        // 리뷰 컬럼 확인 및 안내
        const hasReviewTitle = "review_title" in firstRow
        const hasReviewContent = "review_content" in firstRow

        console.log("리뷰 컬럼 확인:")
        console.log("- review_title:", hasReviewTitle)
        console.log("- review_content:", hasReviewContent)

        if (hasReviewTitle || hasReviewContent) {
          console.log("✅ 리뷰 데이터 발견! 고급 분석이 가능합니다.")
        } else {
          console.log("⚠️ 리뷰 컬럼(review_title, review_content)이 없습니다.")
        }
      } catch (error) {
        console.error("파일 업로드 오류:", error)
        alert(`파일 업로드 중 오류가 발생했습니다: ${error instanceof Error ? error.message : "알 수 없는 오류"}`)
      } finally {
        setIsUploading(false)
        setTimeout(() => setUploadProgress(0), 1000)
      }
    },
    [parseCSV],
  )

  const handleSearch = async () => {
    if (!selectedCategory) {
      alert("카테고리를 먼저 선택해 주세요.")
      return
    }

    setIsSearching(true)
    setWarnings([])

    // Filter products by category
    const categoryProducts = currentData.filter((p) => p.category === selectedCategory)

    if (categoryProducts.length < 3) {
      setWarnings(["선택한 카테고리 내 제품 수가 너무 적습니다. 다른 카테고리를 선택해 주세요."])
      setIsSearching(false)
      return
    }

    // Calculate similarities
    const productsWithSimilarity = categoryProducts.map((product) => {
      const textSim = calculateTextSimilarity(productDesc, product.about_product)
      const priceSim = calculatePriceSimilarity(actualPrice, product.actual_price)
      const discountSim = calculatePriceSimilarity(discountPct[0], product.discount_percentage)

      // Combined similarity score
      const totalSimilarity = textSim * 0.6 + priceSim * 0.3 + discountSim * 0.1

      return {
        ...product,
        similarity: totalSimilarity,
        textSimilarity: textSim,
        priceSimilarity: priceSim,
        reviewCountByTitle: countReviewsByTitle(product),
      }
    })

    // Sort by similarity and get top 3
    const topMatches = productsWithSimilarity.sort((a, b) => b.similarity - a.similarity).slice(0, 3)

    // Check for warnings
    const avgSimilarity = topMatches.reduce((sum, p) => sum + p.textSimilarity, 0) / topMatches.length
    const maxSimilarity = Math.max(...topMatches.map((p) => p.textSimilarity))

    const newWarnings = []
    if (avgSimilarity < 0.1) {
      newWarnings.push(
        "⚠️ 입력한 설명이 다른 제품들과 전반적으로 크게 다릅니다. 유사 제품 목록의 정확도가 낮을 수 있습니다. (평균 유사도 낮음)\n권장: 설명을 더 구체적으로 작성해 보세요.",
      )
    }
    if (maxSimilarity < 0.2) {
      newWarnings.push(
        "⚠️ 입력한 설명과 매우 유사한 제품이 거의 없습니다. 유사 제품 목록의 정확도가 낮을 수 있습니다. (최고 유사도 낮음)",
      )
    }

    setWarnings(newWarnings)
    setResults(topMatches)
    setIsSearching(false)
  }

  // CSV에서 실제 리뷰 데이터 추출 (review_title, review_content만 사용)
  const extractRealReviews = (product: any): string[] => {
    const reviews: string[] = []

    // review_title과 review_content만 사용
    if (product.review_title && typeof product.review_title === "string") {
      const title = product.review_title.trim()
      if (title && title.length > 5) {
        // 최소 5자 이상
        reviews.push(title)
      }
    }

    if (product.review_content && typeof product.review_content === "string") {
      const content = product.review_content.trim()
      if (content && content.length > 10) {
        // 최소 10자 이상
        reviews.push(content)
      }
    }

    // 제목과 내용을 합쳐서 하나의 완전한 리뷰로 만들기 (둘 다 있는 경우)
    if (
      product.review_title &&
      product.review_content &&
      typeof product.review_title === "string" &&
      typeof product.review_content === "string"
    ) {
      const title = product.review_title.trim()
      const content = product.review_content.trim()
      if (title && content && title.length > 5 && content.length > 10) {
        const combinedReview = `${title}. ${content}`
        reviews.push(combinedReview)
      }
    }

    console.log(`${product.product_name}에서 추출된 리뷰:`, reviews.length, "개")
    if (reviews.length > 0) {
      console.log("리뷰 샘플:", reviews[0].substring(0, 100) + "...")
    }

    return reviews.slice(0, 20) // 최대 20개 리뷰만 분석
  }

  // 고급 리뷰 분석 함수 (CSV 데이터만 사용)
  const generateAdvancedReviewSummary = async (product: any) => {
    const productKey = `${product.product_name}_${product.category}`

    if (reviewSummaries[productKey] || loadingSummaries[productKey]) {
      return // 이미 요약이 있거나 로딩 중이면 스킵
    }

    setLoadingSummaries((prev) => ({ ...prev, [productKey]: true }))

    // 시뮬레이션을 위한 지연
    await new Promise((resolve) => setTimeout(resolve, 2000))

    try {
      // CSV에서 실제 리뷰 데이터만 추출
      const realReviews = extractRealReviews(product)

      console.log(`${product.product_name}에서 ${realReviews.length}개의 리뷰를 발견했습니다.`)

      // 고급 텍스트 분석 실행
      const analysis: any = advancedReviewAnalysis(realReviews)

      // 실제 리뷰 사용 여부 표시
      analysis.isRealData = realReviews.length > 0
      analysis.reviewCount = realReviews.length

      setReviewSummaries((prev) => ({ ...prev, [productKey]: analysis }))
    } catch (error) {
      console.error("리뷰 분석 오류:", error)
      setReviewSummaries((prev) => ({
        ...prev,
        [productKey]: {
          overall_sentiment: "neutral",
          confidence: 0,
          sentiment_distribution: { positive: 0, negative: 0, neutral: 0 },
          top_keywords: [],
          topic_analysis: {},
          positive_highlights: [],
          negative_concerns: [],
          summary: "리뷰 분석 중 오류가 발생했습니다.",
          isRealData: false,
          reviewCount: 0,
          hasInsufficientData: true,
        },
      }))
    } finally {
      setLoadingSummaries((prev) => ({ ...prev, [productKey]: false }))
    }
  }


  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* 헤더 로고 */}
      <div className="w-full bg-white shadow-sm border-b mb-8">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <h1 className="text-3xl font-bold text-blue-600">Rmazon</h1>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-2">
              <ShoppingCart className="h-8 w-8 text-blue-600" />
              예비 판매자를 위한 시장 내 유사 상품 탐색기
            </h2>
            <p className="text-gray-600">Amazon 시장에서 유사한 제품을 찾아 경쟁력을 분석해보세요</p>
          </div>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                CSV 파일 업로드
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="csv-upload">Amazon 제품 데이터 CSV 파일 선택</Label>
                <Input
                  id="csv-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                  className="mt-1"
                />
              </div>

              {isUploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>파일 업로드 중...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} className="w-full" />
                </div>
              )}

              {fileInfo && (
                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>{fileInfo.name}</strong> 파일이 성공적으로 업로드되었습니다.
                    <br />총 <strong>{fileInfo.rows.toLocaleString()}</strong>개의 제품 데이터를 불러왔습니다. (
                    {(fileInfo.size / 1024 / 1024).toFixed(2)} MB)
                  </AlertDescription>
                </Alert>
              )}

              <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
                <h4 className="font-semibold mb-2">CSV 파일 형식 안내:</h4>
                <ul className="space-y-1 text-xs">
                  <li>
                    • <strong>필수 컬럼:</strong> product_name, category, about_product
                  </li>
                  <li>
                    • <strong>권장 컬럼:</strong> discounted_price, discount_percentage, rating, rating_count, img_link
                  </li>
                  <li>
                    • <strong>리뷰 컬럼:</strong> review_title, review_content (실제 리뷰 분석용)
                  </li>
                  <li>
                    • <strong>파일 크기:</strong> 최대 50MB
                  </li>
                  <li>
                    • <strong>인코딩:</strong> UTF-8 권장
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                상품 정보 입력
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Category Search */}
              <div className="space-y-2">
                <Label htmlFor="category-search">카테고리 검색</Label>
                <Input
                  id="category-search"
                  placeholder="카테고리를 검색하세요..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              {/* Category Selection */}
              {currentData.length === 0 ? (
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    먼저 CSV 파일을 업로드해주세요. 파일을 업로드하면 카테고리 선택이 가능합니다.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>카테고리 선택</Label>

                    {filteredCategories.length > 3 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowAllCategories(!showAllCategories)}
                        className="text-blue-600 hover:text-blue-800"
                      >
                        {showAllCategories ? (
                          <>
                            <ChevronUp className="h-4 w-4 mr-1" />
                            접기
                          </>
                        ) : (
                          <>
                            <ChevronDown className="h-4 w-4 mr-1" />
                            더보기 ({filteredCategories.length - 3}개)
                          </>
                        )}
                      </Button>
                    )}
                  </div>

                  {filteredCategories.length > 0 ? (
                    <>
                      <div className="flex flex-wrap gap-2">
                        {(showAllCategories ? filteredCategories : filteredCategories.slice(0, 3)).map((category) => (
                          <Badge
                            key={category}
                            variant={selectedCategory === category ? "default" : "outline"}
                            className="cursor-pointer hover:bg-blue-100 transition-colors"
                            onClick={() => setSelectedCategory(category)}
                          >
                            {category}
                          </Badge>
                        ))}
                      </div>
                      {filteredCategories.length > 3 && !showAllCategories && (
                        <p className="text-sm text-gray-500">
                          {filteredCategories.length - 3}개의 추가 카테고리가 있습니다. "더보기"를 클릭하세요.
                        </p>
                      )}
                    </>
                  ) : (
                    <div className="p-4 text-center text-gray-500 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
                      해당 카테고리가 존재하지 않습니다. 검색어를 확인해주세요.
                    </div>
                  )}
                </div>
              )}

              {/* Product Description */}
              <div className="space-y-2">
                <Label htmlFor="product-desc">상품 설명 입력</Label>
                <Textarea
                  id="product-desc"
                  placeholder="예시: Outdoor camping gear with solar panel"
                  value={productDesc}
                  onChange={(e) => setProductDesc(e.target.value)}
                  rows={3}
                />
              </div>

              {/* Price Input */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="actual-price">정가 (₹)</Label>
                  <Input
                    id="actual-price"
                    type="number"
                    min="0"
                    value={actualPrice}
                    onChange={(e) => setActualPrice(Number(e.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>할인율 (%): {discountPct[0]}%</Label>
                  <Slider value={discountPct} onValueChange={setDiscountPct} max={100} step={1} className="w-full" />
                </div>
              </div>

              {/* Calculated Discounted Price */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-lg font-semibold text-blue-800">
                  할인가 (자동 계산): ₹{discountedPrice.toLocaleString()}
                </p>
              </div>

              {/* Search Button */}
              <Button onClick={handleSearch} className="w-full" size="lg" disabled={isSearching}>
                {isSearching ? "탐색 중..." : "시장 내 유사 상품 탐색하기"}
              </Button>
            </CardContent>
          </Card>

          {/* Warnings */}
          {warnings.length > 0 && (
            <div className="mb-6 space-y-2">
              {warnings.map((warning, index) => (
                <Alert key={index} variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="whitespace-pre-line">{warning}</AlertDescription>
                </Alert>
              ))}
            </div>
          )}

          {/* Results */}
          {results.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">📋 유사한 상위 3개 제품</h2>

              {results.map((product, index) => (
                <Card key={`${product.product_name}_${product.category}_${index}`} className="overflow-hidden">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0">
                        <img
                          src={product.img_link || "/placeholder.svg"}
                          alt={product.product_name}
                          className="w-24 h-24 object-cover rounded-lg"
                        />
                      </div>

                      <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary">{index + 1}위</Badge>
                          <h3 className="text-lg font-semibold">{product.product_name}</h3>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="font-medium">유사도:</span>
                            <div className="text-blue-600 font-mono">{(product.similarity * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <span className="font-medium">정가:</span>
                            <div>₹{product.actual_price.toLocaleString()}</div>
                          </div>
                          <div>
                            <span className="font-medium">할인율:</span>
                            <div>{product.discount_percentage}%</div>
                          </div>
                          <div>
                            <span className="font-medium">할인가:</span>
                            <div className="text-green-600 font-semibold">
                              ₹{product.discounted_price.toLocaleString()}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <div className="flex items-center gap-1">
                            <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                            <span>{product.rating}</span>
                          </div>
                          <div className="space-y-1">
                            <div>총 별점 수: {product.rating_count.toLocaleString()}</div>
                            <div>총 리뷰 수: {product.reviewCountByTitle}</div>
                          </div>
                         {reviewSummaries[`${product.product_name}_${product.category}`] && (
                           <span className="text-sm text-gray-500 ml-2">
                              (분석된 리뷰: {reviewSummaries[`${product.product_name}_${product.category}`].reviewCount}개)
                           </span>
                         )}
                        </div>

                        <p className="text-sm text-gray-600 line-clamp-2">{product.about_product}</p>

                        {/* 고급 리뷰 분석 */}
                        <div className="mt-4 pt-4 border-t border-gray-200">
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                              <BarChart3 className="h-4 w-4 text-purple-600" />
                              고급 리뷰 분석
                              <Badge variant="outline" className="text-xs bg-purple-50 text-purple-700">
                                AI 분석
                              </Badge>
                            </h4>
                            {!reviewSummaries[`${product.product_name}_${product.category}`] &&
                              !loadingSummaries[`${product.product_name}_${product.category}`] && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => generateAdvancedReviewSummary(product)}
                                  className="text-xs"
                                >
                                  분석 시작
                                </Button>
                              )}
                          </div>

                          {loadingSummaries[`${product.product_name}_${product.category}`] && (
                            <div className="space-y-3">
                              <div className="flex items-center gap-2 text-sm text-gray-500 p-3 bg-purple-50 rounded-lg">
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-600"></div>
                                리뷰 데이터를 분석하고 있습니다...
                              </div>
                            </div>
                          )}

                          {reviewSummaries[`${product.product_name}_${product.category}`] && (() => {
                              const summary = reviewSummaries[`${product.product_name}_${product.category}`];
                              if (!summary) return null;
                              
                              return (
                                <div className="space-y-4 pt-4">
                                  {/* 데이터 소스 */}
                                  <div className="flex items-center gap-2 text-xs">
                                    <Badge
                                      variant={
                                        summary.isRealData && summary.reviewCount > 0
                                          ? "default"
                                          : "secondary"
                                      }
                                      className="text-xs"
                                    >
                                      {summary.isRealData && summary.reviewCount > 0
                                        ? "CSV 리뷰 데이터"
                                        : "리뷰 데이터 없음"}
                                    </Badge>
                                    <span className="text-gray-500">
                                      CSV에서 추출: {summary.reviewCount}개
                                    </span>
                                  </div>

                                  {/* 리뷰 데이터 부족 */}
                                  {summary.hasInsufficientData && (
                                    <Alert className="border-amber-200 bg-amber-50">
                                      <Info className="h-4 w-4 text-amber-600" />
                                      <AlertDescription className="text-amber-800 text-xs">
                                        리뷰 데이터가 충분하지 않아 분석이 정확하지 않을 수 있습니다. ({summary.reviewCount}개 리뷰)
                                      </AlertDescription>
                                    </Alert>
                                  )}

                                  {/* 감정 분포 */}
                                  <div className="p-4 bg-gray-50 rounded-lg">
                                    <h5 className="font-semibold text-sm mb-3 flex items-center gap-2">
                                      <TrendingUp className="h-4 w-4 text-gray-600" />
                                      감정 분포
                                    </h5>
                                    <div className="grid grid-cols-3 text-center">
                                      <div>
                                        <p className="text-2xl font-bold text-blue-600">{summary.sentiment_distribution.positive}</p>
                                        <p className="text-xs text-blue-500">긍정</p>
                                      </div>
                                      <div>
                                        <p className="text-2xl font-bold text-gray-500">{summary.sentiment_distribution.neutral}</p>
                                        <p className="text-xs text-gray-400">중립</p>
                                      </div>
                                      <div>
                                        <p className="text-2xl font-bold text-red-600">{summary.sentiment_distribution.negative}</p>
                                        <p className="text-xs text-red-500">부정</p>
                                      </div>
                                    </div>
                                  </div>

                                  {/* 주요 키워드 */}
                                  {summary.top_keywords.length > 0 && (
                                      <div className="p-4 bg-gray-50 rounded-lg">
                                          <h5 className="font-semibold text-sm mb-3 flex items-center gap-2">
                                              <MessageSquare className="h-4 w-4 text-gray-600" />
                                              주요 키워드
                                          </h5>
                                          <div className="flex flex-wrap gap-2">
                                              {summary.top_keywords.slice(0, 10).map((kw: { word: string; count: number }) => (
                                                  <Badge key={kw.word} variant="outline" className="bg-white">
                                                      {kw.word} <span className="text-gray-500 ml-1.5">({kw.count})</span>
                                                  </Badge>
                                              ))}
                                          </div>
                                      </div>
                                  )}

                                  {/* 긍정/부정 하이라이트 */}
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                      {summary.positive_highlights.length > 0 && (
                                          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                                              <h5 className="font-semibold text-sm mb-2 text-green-800 flex items-center gap-2">
                                                  <ThumbsUp className="h-4 w-4" />
                                                  긍정적 하이라이트
                                              </h5>
                                              <ul className="space-y-2 text-xs text-green-700">
                                                  {summary.positive_highlights.slice(0, 2).map((hl: string, i: number) => (
                                                      <li key={i} className="leading-relaxed">"{hl}"</li>
                                                  ))}
                                              </ul>
                                          </div>
                                      )}
                                      {summary.negative_concerns.length > 0 && (
                                          <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                                              <h5 className="font-semibold text-sm mb-2 text-red-800 flex items-center gap-2">
                                                  <ThumbsDown className="h-4 w-4" />
                                                  부정적 우려사항
                                              </h5>
                                              <ul className="space-y-2 text-xs text-red-700">
                                                  {summary.negative_concerns.slice(0, 2).map((nc: string, i: number) => (
                                                      <li key={i} className="leading-relaxed">"{nc}"</li>
                                                  ))}
                                              </ul>
                                          </div>
                                      )}
                                  </div>
                                  
                                  {/* AI 종합 분석 */}
                                  <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                                      <div className="flex items-start justify-between">
                                          <div>
                                              <h5 className="font-semibold text-sm text-purple-800 mb-2">
                                                  AI 종합 분석
                                              </h5>
                                              <p className="text-sm text-purple-700">{summary.summary}</p>
                                          </div>
                                          <Badge variant={
                                              summary.overall_sentiment === 'positive' ? 'default' :
                                              summary.overall_sentiment === 'negative' ? 'destructive' : 'secondary'
                                          } className="capitalize">
                                              {
                                                  summary.overall_sentiment === 'positive' ? '긍정적' :
                                                  summary.overall_sentiment === 'negative' ? '부정적' : '중립'
                                              }
                                          </Badge>
                                      </div>
                                  </div>

                                </div>
                              )
                          })()}
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
