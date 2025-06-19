"use client"

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, FileText, Loader2, AlertCircle, CheckCircle } from "lucide-react";

export default function AdminPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
      setError(null);
      setSuccess(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("업로드할 CSV 파일을 선택해주세요.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // 나중에 실제 백엔드 엔드포인트로 교체될 부분입니다.
      // const response = await fetch("http://127.0.0.1:8000/upload-csv", {
      //   method: "POST",
      //   body: formData,
      // });

      // 임시로 성공을 시뮬레이션합니다.
      await new Promise(resolve => setTimeout(resolve, 1500));

      // if (!response.ok) {
      //   throw new Error("파일 업로드에 실패했습니다. 파일 형식과 내용을 확인해주세요.");
      // }

      setSuccess("파일이 성공적으로 업로드되어 데이터가 업데이트되었습니다.");
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="max-w-3xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl font-bold">데이터 관리</CardTitle>
            <CardDescription>
              새로운 CSV 파일을 업로드하여 유사 상품 분석 및 별점 예측에 사용될 데이터를 교체합니다.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="csv-upload">CSV 파일 업로드</Label>
              <div className="flex items-center gap-4">
                <Input id="csv-upload" type="file" accept=".csv" onChange={handleFileChange} className="flex-1" />
                <Button onClick={handleUpload} disabled={isLoading || !file}>
                  {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Upload className="mr-2 h-4 w-4" />}
                  {isLoading ? "업로드 중..." : "업로드"}
                </Button>
              </div>
               {file && <p className="text-sm text-muted-foreground">선택된 파일: {file.name}</p>}
            </div>

            {error && (
              <div className="p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                <p>{error}</p>
              </div>
            )}
            {success && (
              <div className="p-3 bg-green-50 text-green-700 border border-green-200 rounded-md flex items-center gap-2">
                <CheckCircle className="h-5 w-5" />
                <p>{success}</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="mt-8">
          <CardHeader>
            <div className="flex items-center gap-3">
              <FileText className="w-6 h-6 text-blue-600" />
              <CardTitle>CSV 파일 요구사항</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              업로드하는 CSV 파일은 다음 필수 컬럼들을 모두 포함해야 합니다. 컬럼 순서는 상관 없으나, 이름은 정확히 일치해야 합니다.
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">product_name</code>
                <span className="text-gray-500">- 상품명</span>
              </li>
              <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">review_title</code>
                <span className="text-gray-500">- 리뷰 제목</span>
              </li>
              <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">review_content</code>
                <span className="text-gray-500">- 리뷰 내용</span>
              </li>
              <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">discounted_price</code>
                <span className="text-gray-500">- 할인가</span>
              </li>
               <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">rating_count</code>
                <span className="text-gray-500">- 평점 개수</span>
              </li>
               <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">category_cleaned</code>
                <span className="text-gray-500">- 카테고리</span>
              </li>
               <li className="flex items-center gap-2">
                <code className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">rating</code>
                <span className="text-gray-500">- 평점</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 