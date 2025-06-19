import { BarChart3 } from "lucide-react";

const PredictPage = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-gray-50">
      <div className="p-6 bg-purple-100 rounded-full">
        <BarChart3 className="w-12 h-12 text-purple-600" />
      </div>
      <h1 className="mt-6 text-4xl font-bold text-gray-800">
        판매 지표 예측
      </h1>
      <p className="mt-4 text-lg text-gray-600 max-w-md">
        이 기능은 현재 준비 중입니다. 릿지 회귀 모델을 사용하여 상품의 예상 별점을 분석하는 기능이 곧 제공될 예정입니다.
      </p>
    </div>
  );
};

export default PredictPage; 