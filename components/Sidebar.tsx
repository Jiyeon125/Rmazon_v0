"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Search, Wand2 } from "lucide-react"

const Sidebar = () => {
  const pathname = usePathname()

  const navItems = [
    { href: "/search", label: "유사 상품 탐색", icon: Search },
    { href: "/predict", label: "판매 지표 예측", icon: Wand2 },
  ]

  return (
    <aside className="w-60 flex-shrink-0 border-r bg-white flex flex-col shadow-md">
      <div className="h-16 flex items-center justify-center border-b">
        <Link href="/" className="text-2xl font-bold text-blue-600 hover:opacity-80 transition-opacity">
          Rmazon
        </Link>
      </div>
      <nav className="flex-1 px-4 py-4 space-y-2">
        {navItems.map((item) => {
          const isActive = pathname.startsWith(item.href)
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-gray-700 transition-all hover:bg-blue-50 hover:text-blue-600 ${
                isActive ? "bg-blue-100 text-blue-700 font-semibold" : ""
              }`}
            >
              <item.icon className="h-5 w-5" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}

export default Sidebar 