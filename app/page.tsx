import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Search, Wand2, BarChart3 } from "lucide-react"

export default function LandingPage() {
  return (
    <div className="flex flex-col items-center justify-center text-center p-8">
      <div className="max-w-3xl">
        <h1 className="text-5xl font-extrabold text-gray-900 mb-4 tracking-tight">
          <span className="text-blue-600">Rmazon</span>에서<br/>
          성공적인 판매 전략을 세워보세요
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Amazon 시장의 경쟁력을 심층 분석하고, 데이터 기반의 판매 지표 예측으로
          <br/>
          여러분의 비즈니스 성공을 앞당깁니다.
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/search">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
              <Search className="mr-2 h-5 w-5" />
              유사 상품 탐색 시작하기
            </Button>
          </Link>
          <Link href="/predict">
            <Button size="lg" variant="outline">
              <Wand2 className="mr-2 h-5 w-5" />
              판매 지표 예측하기
            </Button>
          </Link>
        </div>
      </div>

      <div className="mt-20 grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl w-full">
        <div className="p-8 bg-white rounded-xl shadow-md border flex flex-col items-start">
          <div className="bg-blue-100 p-3 rounded-full mb-4">
            <Search className="h-7 w-7 text-blue-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">유사 상품 탐색</h2>
          <p className="text-gray-600 text-left">
            내 상품과 유사한 경쟁 제품을 찾아내고, 가격, 할인율, 고객 리뷰 등 상세 데이터를 비교 분석하여 시장 내 포지셔닝 전략을 수립할 수 있습니다.
          </p>
        </div>
        <div className="p-8 bg-white rounded-xl shadow-md border flex flex-col items-start">
          <div className="bg-purple-100 p-3 rounded-full mb-4">
            <BarChart3 className="h-7 w-7 text-purple-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">판매 지표 예측</h2>
          <p className="text-gray-600 text-left">
            회귀 분석 모델을 기반으로 상품의 별점을 예측합니다. 상품 특징과 시장 데이터를 활용하여 예상되는 고객 반응을 미리 파악하고 상품성을 개선하세요. (준비 중)
          </p>
        </div>
      </div>
    </div>
  )
}
