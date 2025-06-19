"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart3, Star, Loader2, AlertCircle } from "lucide-react"

// 백엔드에서 제공된 카테고리 목록
const categories = [
    'Computers&Accessories | Accessories&Peripherals | Cables&Accessories', 'Computers&Accessories | NetworkingDevices | NetworkAdapters', 
    'Electronics | HomeTheater,TV&Video | Accessories', 'Electronics | HomeTheater,TV&Video | Televisions', 
    'Electronics | HomeAudio | Accessories', 'Electronics | HomeTheater,TV&Video | Projectors', 
    'Electronics | HomeTheater,TV&Video | SatelliteEquipment', 'Electronics | HomeAudio | MediaStreamingDevices', 
    'Electronics | HomeTheater,TV&Video | AVReceivers&Amplifiers', 'Electronics | HomeAudio | Speakers', 
    'Electronics | WearableTechnology | SmartWatches', 'Electronics | Mobiles&Accessories | MobileAccessories', 
    'Electronics | Mobiles&Accessories | Smartphones&BasicMobiles', 'Electronics | Accessories | MemoryCards', 
    'Electronics | Headphones,Earbuds&Accessories | Headphones', 'Computers&Accessories | Accessories&Peripherals | LaptopAccessories', 
    'Electronics | Headphones,Earbuds&Accessories | Adapters', 'Computers&Accessories | ExternalDevices&DataStorage | PenDrives', 
    'Computers&Accessories | Accessories&Peripherals | Keyboards,Mice&InputDevices', 'MusicalInstruments | Microphones | Condenser', 
    'Electronics | GeneralPurposeBatteries&BatteryChargers | DisposableBatteries', 'OfficeProducts | OfficePaperProducts | Paper', 
    'Home&Kitchen | CraftMaterials | Scrapbooking', 'Computers&Accessories | ExternalDevices&DataStorage | ExternalHardDisks', 
    'Electronics | Cameras&Photography | VideoCameras', 'Electronics | Cameras&Photography | Accessories', 
    'OfficeProducts | OfficeElectronics | Calculators', 'Computers&Accessories | NetworkingDevices | Repeaters&Extenders', 
    'Computers&Accessories | Printers,Inks&Accessories | Inks,Toners&Cartridges', 'Computers&Accessories | Accessories&Peripherals | PCGamingPeripherals', 
    'Home&Kitchen | CraftMaterials | PaintingMaterials', 'Computers&Accessories | Accessories&Peripherals | HardDiskBags', 
    'Electronics | Cameras&Photography | Flashes', 'Computers&Accessories | NetworkingDevices', 
    'Computers&Accessories | NetworkingDevices | Routers', 'Electronics | GeneralPurposeBatteries&BatteryChargers', 
    'Electronics | GeneralPurposeBatteries&BatteryChargers | RechargeableBatteries', 'Computers&Accessories | Accessories&Peripherals | Adapters', 
    'Computers&Accessories | Monitors', 'Computers&Accessories | Accessories&Peripherals | USBGadgets', 
    'Electronics | Cameras&Photography | SecurityCameras', 'Computers&Accessories | Accessories&Peripherals | TabletAccessories', 
    'Computers&Accessories | Accessories&Peripherals | USBHubs', 'Computers&Accessories | Accessories&Peripherals | Audio&VideoAccessories', 
    'Computers&Accessories | ExternalDevices&DataStorage | ExternalMemoryCardReaders', 'Computers&Accessories | Components | Memory', 
    'Computers&Accessories | Accessories&Peripherals | UninterruptedPowerSupplies', 'Electronics | Headphones,Earbuds&Accessories | Cases', 
    'HomeImprovement | Electrical | Adapters&Multi-Outlets', 'Computers&Accessories | Components | InternalSolidStateDrives', 
    'Computers&Accessories | NetworkingDevices | DataCards&Dongles', 'Home&Kitchen | CraftMaterials | DrawingMaterials', 
    'Computers&Accessories | Components | InternalHardDrives', 'Computers&Accessories | Printers,Inks&Accessories | Printers', 
    'Electronics | Headphones,Earbuds&Accessories | Earpads', 'Toys&Games | Arts&Crafts | Drawing&PaintingSupplies', 
    'Computers&Accessories | ExternalDevices&DataStorage | ExternalSolidStateDrives', 'Electronics | PowerAccessories | SurgeProtectors', 
    'Computers&Accessories | Tablets', 'HomeImprovement | Electrical | CordManagement', 
    'Computers&Accessories | Accessories&Peripherals | HardDriveAccessories', 'Computers&Accessories | Laptops | TraditionalLaptops', 
    'Home&Kitchen | Kitchen&HomeAppliances | SmallKitchenAppliances', 'Home&Kitchen | Heating,Cooling&AirQuality | RoomHeaters', 
    'Home&Kitchen | Kitchen&HomeAppliances | Vacuum,Cleaning&Ironing', 'Home&Kitchen | Kitchen&Dining | KitchenTools', 
    'Home&Kitchen | Heating,Cooling&AirQuality | WaterHeaters&Geysers', 'Home&Kitchen | HomeStorage&Organization | LaundryOrganization', 
    'Home&Kitchen | Heating,Cooling&AirQuality | Fans', 'Home&Kitchen | Kitchen&HomeAppliances | Coffee,Tea&Espresso', 
    'Home&Kitchen | Kitchen&HomeAppliances | WaterPurifiers&Accessories', 'Car&Motorbike | CarAccessories | InteriorAccessories', 
    'Home&Kitchen | Heating,Cooling&AirQuality | AirPurifiers', 'Home&Kitchen | Kitchen&HomeAppliances | SewingMachines&Accessories', 
    'Health&PersonalCare | HomeMedicalSupplies&Equipment | HealthMonitors', 'Home&Kitchen | Heating,Cooling&AirQuality | Humidifiers', 
    'Home&Kitchen | Heating,Cooling&AirQuality | AirConditioners', 'Home&Kitchen | Heating,Cooling&AirQuality | Parts&Accessories'
  ]

export default function PredictPage() {
  const [price, setPrice] = useState("")
  const [reviewCount, setReviewCount] = useState("")
  const [category, setCategory] = useState("")
  const [predictedStar, setPredictedStar] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handlePredict = async () => {
    if (!price || !reviewCount || !category) {
      setError("모든 값을 입력해주세요.")
      return
    }
    
    setIsLoading(true)
    setError(null)
    setPredictedStar(null)

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          price: parseFloat(price),
          review_count: parseInt(reviewCount),
          category: category,
        }),
      })

      if (!response.ok) {
        throw new Error("API 요청에 실패했습니다. 백엔드 서버 상태를 확인해주세요.")
      }

      const data = await response.json()
      setPredictedStar(data.predicted_star)
    } catch (err) {
      setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto p-4 md:p-8 flex justify-center items-start">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-100 rounded-full">
              <BarChart3 className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <CardTitle className="text-2xl font-bold">상품 별점 예측</CardTitle>
              <CardDescription>상품 정보를 입력하여 예상 별점을 확인해보세요.</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid w-full items-center gap-2">
            <Label htmlFor="price">가격 (Price)</Label>
            <Input
              id="price"
              type="number"
              placeholder="예: 29.99"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
          <div className="grid w-full items-center gap-2">
            <Label htmlFor="reviewCount">리뷰 수 (Review Count)</Label>
            <Input
              id="reviewCount"
              type="number"
              placeholder="예: 500"
              value={reviewCount}
              onChange={(e) => setReviewCount(e.target.value)}
            />
          </div>
          <div className="grid w-full items-center gap-2">
            <Label htmlFor="category">카테고리 (Category)</Label>
            <Select onValueChange={setCategory} value={category}>
              <SelectTrigger>
                <SelectValue placeholder="카테고리를 선택하세요" />
              </SelectTrigger>
              <SelectContent>
                {categories.map((cat) => (
                  <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
        <CardFooter className="flex flex-col items-stretch">
          <Button onClick={handlePredict} disabled={isLoading} className="w-full">
            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Star className="mr-2 h-4 w-4" />}
            {isLoading ? "예측 중..." : "예측하기"}
          </Button>
          
          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 border border-red-200 rounded-md flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              <p>{error}</p>
            </div>
          )}

          {predictedStar !== null && !isLoading && (
            <div className="mt-6 text-center">
              <p className="text-lg text-gray-600">예상 별점</p>
              <p className="text-5xl font-extrabold text-blue-600 flex items-center justify-center gap-2">
                <Star className="w-10 h-10 text-yellow-400 fill-current" />
                {predictedStar.toFixed(2)}
              </p>
            </div>
          )}
        </CardFooter>
      </Card>
    </div>
  )
} 