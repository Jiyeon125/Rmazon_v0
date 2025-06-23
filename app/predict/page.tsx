"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { BarChart3, Star, Loader2, AlertCircle } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

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
  predicted_review_count: number; //예측 결과에 리뷰 수 추가
  price_percentile: number;
  review_count_percentile: number;
  rating_percentile: number;
}

const categories = [
  "Computers&Accessories | Accessories&Peripherals | Cables&Accessories",
  "Computers&Accessories | NetworkingDevices | NetworkAdapters",
  "Electronics | HomeTheater,TV&Video | Accessories",
  "Electronics | HomeTheater,TV&Video | Televisions",
  "Electronics | HomeAudio | Accessories",
  "Electronics | HomeTheater,TV&Video | Projectors",
  "Electronics | HomeTheater,TV&Video | SatelliteEquipment",
  "Electronics | HomeAudio | MediaStreamingDevices",
  "Electronics | HomeTheater,TV&Video | AVReceivers&Amplifiers",
  "Electronics | HomeAudio | Speakers",
  "Electronics | WearableTechnology | SmartWatches",
  "Electronics | Mobiles&Accessories | MobileAccessories",
  "Electronics | Mobiles&Accessories | Smartphones&BasicMobiles",
  "Electronics | Accessories | MemoryCards",
  "Electronics | Headphones,Earbuds&Accessories | Headphones",
  "Computers&Accessories | Accessories&Peripherals | LaptopAccessories",
  "Electronics | Headphones,Earbuds&Accessories | Adapters",
  "Computers&Accessories | ExternalDevices&DataStorage | PenDrives",
  "Computers&Accessories | Accessories&Peripherals | Keyboards,Mice&InputDevices",
  "MusicalInstruments | Microphones | Condenser",
  "Electronics | GeneralPurposeBatteries&BatteryChargers | DisposableBatteries",
  "OfficeProducts | OfficePaperProducts | Paper",
  "Home&Kitchen | CraftMaterials | Scrapbooking",
  "Computers&Accessories | ExternalDevices&DataStorage | ExternalHardDisks",
  "Electronics | Cameras&Photography | VideoCameras",
  "Electronics | Cameras&Photography | Accessories",
  "OfficeProducts | OfficeElectronics | Calculators",
  "Computers&Accessories | NetworkingDevices | Repeaters&Extenders",
  "Computers&Accessories | Printers,Inks&Accessories | Inks,Toners&Cartridges",
  "Computers&Accessories | Accessories&Peripherals | PCGamingPeripherals",
  "Home&Kitchen | CraftMaterials | PaintingMaterials",
  "Computers&Accessories | Accessories&Peripherals | HardDiskBags",
  "Electronics | Cameras&Photography | Flashes",
  "Computers&Accessories | NetworkingDevices",
  "Computers&Accessories | NetworkingDevices | Routers",
  "Electronics | GeneralPurposeBatteries&BatteryChargers",
  "Electronics | GeneralPurposeBatteries&BatteryChargers | RechargeableBatteries",
  "Computers&Accessories | Accessories&Peripherals | Adapters",
  "Computers&Accessories | Monitors",
  "Computers&Accessories | Accessories&Peripherals | USBGadgets",
  "Electronics | Cameras&Photography | SecurityCameras",
  "Computers&Accessories | Accessories&Peripherals | TabletAccessories",
  "Computers&Accessories | Accessories&Peripherals | USBHubs",
  "Computers&Accessories | Accessories&Peripherals | Audio&VideoAccessories",
  "Computers&Accessories | ExternalDevices&DataStorage | ExternalMemoryCardReaders",
  "Computers&Accessories | Components | Memory",
  "Computers&Accessories | Accessories&Peripherals | UninterruptedPowerSupplies",
  "Electronics | Headphones,Earbuds&Accessories | Cases",
  "HomeImprovement | Electrical | Adapters&Multi-Outlets",
  "Computers&Accessories | Components | InternalSolidStateDrives",
  "Computers&Accessories | NetworkingDevices | DataCards&Dongles",
  "Home&Kitchen | CraftMaterials | DrawingMaterials",
  "Computers&Accessories | Components | InternalHardDrives",
  "Computers&Accessories | Printers,Inks&Accessories | Printers",
  "Electronics | Headphones,Earbuds&Accessories | Earpads",
  "Toys&Games | Arts&Crafts | Drawing&PaintingSupplies",
  "Computers&Accessories | ExternalDevices&DataStorage | ExternalSolidStateDrives",
  "Electronics | PowerAccessories | SurgeProtectors",
  "Computers&Accessories | Tablets",
  "HomeImprovement | Electrical | CordManagement",
  "Computers&Accessories | Accessories&Peripherals | HardDriveAccessories",
  "Computers&Accessories | Laptops | TraditionalLaptops",
  "Home&Kitchen | Kitchen&HomeAppliances | SmallKitchenAppliances",
  "Home&Kitchen | Heating,Cooling&AirQuality | RoomHeaters",
  "Home&Kitchen | Kitchen&HomeAppliances | Vacuum,Cleaning&Ironing",
  "Home&Kitchen | Kitchen&Dining | KitchenTools",
  "Home&Kitchen | Heating,Cooling&AirQuality | WaterHeaters&Geysers",
  "Home&Kitchen | HomeStorage&Organization | LaundryOrganization",
  "Home&Kitchen | Heating,Cooling&AirQuality | Fans",
  "Home&Kitchen | Kitchen&HomeAppliances | Coffee,Tea&Espresso",
  "Home&Kitchen | Kitchen&HomeAppliances | WaterPurifiers&Accessories",
  "Car&Motorbike | CarAccessories | InteriorAccessories",
  "Home&Kitchen | Heating,Cooling&AirQuality | AirPurifiers",
  "Home&Kitchen | Kitchen&HomeAppliances | SewingMachines&Accessories",
  "Health&PersonalCare | HomeMedicalSupplies&Equipment | HealthMonitors",
  "Home&Kitchen | Heating,Cooling&AirQuality | Humidifiers",
  "Home&Kitchen | Heating,Cooling&AirQuality | AirConditioners",
  "Home&Kitchen | Heating,Cooling&AirQuality | Parts&Accessories",
];

// 헬퍼 함수를 컴포넌트 외부로 이동하여 구문 오류를 해결합니다.
const safeParseFloat = (str: string) => parseFloat(str.replace(/,/g, ""));

export default function PredictPage() {
  const [price, setPrice] = useState("");
  const [category, setCategory] = useState<string>("");

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [stats, setStats] = useState<CategoryStats | null>(null);
  const [isStatsLoading, setIsStatsLoading] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);

  useEffect(() => {
    if (!category) {
      setStats(null);
      return;
    }

    const fetchStats = async () => {
      setIsStatsLoading(true);
      setPredictionResult(null);
      setError(null);
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/category-stats?category=${encodeURIComponent(
            category
          )}`
        );
        if (!response.ok) {
          throw new Error("카테고리 통계 정보를 불러오는 데 실패했습니다.");
        }
        const data: CategoryStats = await response.json();
        setStats(data);
        setPrice(String(Math.round((data.min_price + data.max_price) / 2)));
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
        );
        setStats(null);
      } finally {
        setIsStatsLoading(false);
      }
    };
    fetchStats();
  }, [category]);

  const handlePredict = async () => {
    if (!price || !category) {
      setError("모든 값을 입력해주세요.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ price: parseFloat(price), category }),
      });

      if (!response.ok) {
        throw new Error(
          "API 요청에 실패했습니다. 백엔드 서버 상태를 확인해주세요."
        );
      }

      const data: PredictionResult = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="p-3 bg-purple-100 rounded-full">
                <BarChart3 className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <CardTitle className="text-2xl font-bold">
                  상품 판매 지표 예측
                </CardTitle>
                <CardDescription>
                  카테고리와 가격을 기반으로 예상 별점과 리뷰 수를 확인해보세요.
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6 pt-6">
            <div className="grid w-full items-center gap-2">
              <Label htmlFor="category">1. 분석할 상품 카테고리</Label>
              <Select onValueChange={setCategory} value={category}>
                <SelectTrigger>
                  <SelectValue placeholder="카테고리를 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {categories.map((cat) => (
                    <SelectItem key={cat} value={cat}>
                      {cat}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {isStatsLoading ? (
              <div className="flex items-center justify-center p-4">
                <Loader2 className="h-6 w-6 animate-spin" />
                <span className="ml-2">통계 정보 로딩 중...</span>
              </div>
            ) : (
              stats && (
                <div className="grid w-full items-center gap-2">
                  <Label htmlFor="price">2. 가격 (Price)</Label>
                  <p className="text-xs text-muted-foreground">
                    이 카테고리의 가격 범위: ₹{stats.min_price.toLocaleString()}{" "}
                    ~ ₹{stats.max_price.toLocaleString()}
                  </p>
                  <Input
                    id="price"
                    type="number"
                    value={price}
                    onChange={(e) => setPrice(e.target.value)}
                  />
                </div>
              )
            )}
          </CardContent>
          <CardFooter>
            <Button
              onClick={handlePredict}
              disabled={isLoading || isStatsLoading || !stats}
              className="w-full"
            >
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Star className="mr-2 h-4 w-4" />
              )}
              {isLoading ? "예측 중..." : "결과 분석 및 예측"}
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>종합 분석 결과</CardTitle>
            <CardDescription>
              입력한 정보를 바탕으로 시장 내 위치와 예상 별점을 확인합니다.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            {isLoading && (
              <div className="flex items-center justify-center p-4">
                <Loader2 className="h-6 w-6 animate-spin" />
                <span className="ml-2">예측 중...</span>
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                <p>{error}</p>
              </div>
            )}

            {!isLoading && !error && predictionResult && (
              <>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-sm text-muted-foreground">
                      예상 리뷰 수
                    </p>
                    <p className="text-2xl font-bold">
                      {predictionResult.predicted_review_count.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">예상 별점</p>
                    <p className="text-2xl font-bold">
                      {predictionResult.predicted_star.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">가격 위치</p>
                    <p className="text-2xl font-bold">
                      상위{" "}
                      {(100 - predictionResult.price_percentile).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">
                      리뷰 수 위치
                    </p>
                    <p className="text-2xl font-bold">
                      상위{" "}
                      {(100 - predictionResult.review_count_percentile).toFixed(
                        1
                      )}
                      %
                    </p>
                  </div>
                </div>

                {stats && (
                  <div className="space-y-8">
                    <h3 className="text-lg font-semibold border-b pb-2">
                      시장 내 분포도 분석
                    </h3>
                    <div>
                      <Label>가격 분포도</Label>
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={stats.price_distribution}>
                          <XAxis
                            dataKey="name"
                            angle={-45}
                            textAnchor="end"
                            height={70}
                            interval={0}
                            fontSize={12}
                          />
                          <YAxis allowDecimals={false} />
                          <Tooltip
                            cursor={{ fill: "rgba(200, 200, 200, 0.3)" }}
                          />
                          <Bar dataKey="count" name="상품 수">
                            {stats.price_distribution.map((entry, index) => {
                              const [start, end] = entry.name
                                .split("-")
                                .map(safeParseFloat);
                              return (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={
                                    parseFloat(price) >= start &&
                                    parseFloat(price) < end
                                      ? "#8884d8"
                                      : "#d1d5db"
                                  }
                                />
                              );
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div>
                      <Label>리뷰 수 분포도</Label>
                      {predictionResult && (
                        <p className="text-xs text-muted-foreground">
                          예측된 리뷰 수는 이 시장에서 상위{" "}
                          {(
                            100 - predictionResult.review_count_percentile
                          ).toFixed(1)}
                          % 수준입니다.
                        </p>
                      )}
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={stats.review_count_distribution}>
                          <XAxis
                            dataKey="name"
                            angle={-45}
                            textAnchor="end"
                            height={70}
                            interval={0}
                            fontSize={12}
                          />
                          <YAxis allowDecimals={false} />
                          <Tooltip
                            cursor={{ fill: "rgba(200, 200, 200, 0.3)" }}
                          />
                          <Bar dataKey="count" name="상품 수">
                            {stats.review_count_distribution.map(
                              (entry, index) => {
                                const [start, end] = entry.name
                                  .split("-")
                                  .map(safeParseFloat);
                                return (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={
                                      predictionResult.predicted_review_count >=
                                        start &&
                                      predictionResult.predicted_review_count <
                                        end
                                        ? "#82ca9d"
                                        : "#d1d5db"
                                    }
                                  />
                                );
                              }
                            )}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div>
                      <Label>예상 별점 분포도</Label>
                      <p className="text-xs text-muted-foreground">
                        예측된 별점은 이 시장에서 상위{" "}
                        {(100 - predictionResult.rating_percentile).toFixed(1)}%
                        수준입니다.
                      </p>
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={stats.rating_distribution}>
                          <XAxis
                            dataKey="name"
                            angle={-45}
                            textAnchor="end"
                            height={70}
                            interval={0}
                            fontSize={12}
                          />
                          <YAxis allowDecimals={false} />
                          <Tooltip
                            cursor={{ fill: "rgba(200, 200, 200, 0.3)" }}
                          />
                          <Bar dataKey="count" name="상품 수">
                            {stats.rating_distribution.map((entry, index) => {
                              const [start, end] = entry.name
                                .split("-")
                                .map(safeParseFloat);
                              return (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={
                                    predictionResult.predicted_star >= start &&
                                    predictionResult.predicted_star < end
                                      ? "#ffc658"
                                      : "#d1d5db"
                                  }
                                />
                              );
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </>
            )}

            {!isLoading && !error && !predictionResult && (
              <div className="text-center text-muted-foreground py-10">
                카테고리를 선택하고 값을 입력한 후, 예측 버튼을 눌러주세요.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
