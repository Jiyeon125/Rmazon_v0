// 감정 분석을 위한 키워드 사전
const POSITIVE_KEYWORDS = [
  "excellent",
  "great",
  "amazing",
  "perfect",
  "love",
  "recommend",
  "satisfied",
  "quality",
  "fast",
  "good",
  "best",
  "awesome",
  "fantastic",
  "wonderful",
  "outstanding",
  "superb",
  "brilliant",
  "impressive",
  "reliable",
  "durable",
  "comfortable",
  "easy",
  "beautiful",
  "stylish",
  "elegant",
  "smooth",
  "efficient",
]

const NEGATIVE_KEYWORDS = [
  "terrible",
  "awful",
  "bad",
  "worst",
  "hate",
  "disappointed",
  "poor",
  "slow",
  "broken",
  "defective",
  "useless",
  "waste",
  "horrible",
  "disgusting",
  "annoying",
  "frustrating",
  "cheap",
  "flimsy",
  "uncomfortable",
  "difficult",
  "complicated",
  "ugly",
  "expensive",
  "overpriced",
  "unreliable",
  "fragile",
  "noisy",
]

const STOP_WORDS = [
  "the",
  "a",
  "an",
  "and",
  "or",
  "but",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "with",
  "by",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "must",
  "this",
  "that",
  "these",
  "those",
  "i",
  "you",
  "he",
  "she",
  "it",
  "we",
  "they",
  "me",
  "him",
  "her",
  "us",
  "them",
]

// 감정 분석 함수
export function analyzeSentiment(reviewText: string) {
  const words = reviewText.toLowerCase().split(/\s+/)

  let positiveScore = 0
  let negativeScore = 0

  words.forEach((word) => {
    // 정확한 매칭과 부분 매칭 모두 고려
    if (POSITIVE_KEYWORDS.some((keyword) => word.includes(keyword) || keyword.includes(word))) {
      positiveScore++
    }
    if (NEGATIVE_KEYWORDS.some((keyword) => word.includes(keyword) || keyword.includes(word))) {
      negativeScore++
    }
  })

  const totalScore = positiveScore + negativeScore
  const sentiment = totalScore === 0 ? "neutral" : positiveScore > negativeScore ? "positive" : "negative"

  return {
    positive: positiveScore,
    negative: negativeScore,
    sentiment,
    confidence: totalScore > 0 ? Math.abs(positiveScore - negativeScore) / totalScore : 0,
  }
}

// 키워드 빈도 분석
export function extractKeywords(reviews: string[]) {
  const allText = reviews.join(" ").toLowerCase()
  const words = allText.split(/\s+/)

  // 불용어 제거 및 정제
  const filteredWords = words.filter(
    (word) => word.length > 2 && !STOP_WORDS.includes(word) && /^[a-zA-Z]+$/.test(word), // 영문자만
  )

  // 빈도 계산
  const frequency: { [key: string]: number } = {}
  filteredWords.forEach((word) => {
    frequency[word] = (frequency[word] || 0) + 1
  })

  // 상위 키워드 반환
  return Object.entries(frequency)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 15)
    .map(([word, count]) => ({ word, count }))
}

// 부정적 우려사항 추출
export function extractNegativeConcerns(reviews: string[]) {
  return reviews
    .map((review) => ({ review, sentiment: analyzeSentiment(review) }))
    .filter(({ sentiment }) => sentiment.sentiment === "negative" && sentiment.confidence > 0.3)
    .sort((a, b) => b.sentiment.negative - a.sentiment.negative)
    .slice(0, 3)
    .map(({ review }) => (review.length > 120 ? review.substring(0, 120) + "..." : review))
}

// 종합 리뷰 분석 (CSV 데이터만 사용)
export function advancedReviewAnalysis(reviews: string[]) {
  if (reviews.length === 0) {
    return {
      overall_sentiment: "neutral" as const,
      confidence: 0,
      sentiment_distribution: { positive: 0, negative: 0, neutral: 0 },
      top_keywords: [],
      negative_concerns: [],
      summary: "분석할 리뷰가 없습니다.",
      hasInsufficientData: true,
    }
  }

  const sentimentResults = reviews.map(analyzeSentiment)

  const positiveReviews = sentimentResults.filter((r) => r.sentiment === "positive").length
  const negativeReviews = sentimentResults.filter((r) => r.sentiment === "negative").length
  const neutralReviews = sentimentResults.filter((r) => r.sentiment === "neutral").length

  const totalReviews = reviews.length
  const positivePercentage = totalReviews > 0 ? (positiveReviews / totalReviews) * 100 : 0

  let overallSentiment: "positive" | "negative" | "neutral"
  let summary: string
  
  if (positivePercentage >= 70) {
    overallSentiment = "positive"
    summary = `전체 리뷰의 ${positivePercentage.toFixed(0)}%가 긍정적입니다. 고객들이 전반적으로 매우 만족하는 제품입니다.`
  } else if (positivePercentage >= 40) {
    overallSentiment = "neutral"
    summary = `전체 리뷰의 ${positivePercentage.toFixed(0)}%가 긍정적입니다. 긍정적인 평가가 있으나, 일부 개선이 필요한 점도 보입니다.`
  } else {
    overallSentiment = "negative"
    summary = `전체 리뷰의 ${positivePercentage.toFixed(0)}%가 긍정적입니다. 일부 부정적인 평가가 있어 구매 시 주의가 필요합니다.`
  }
  
  const topKeywords = extractKeywords(reviews)
  const negativeConcerns = extractNegativeConcerns(reviews)
  
  // 데이터 부족 여부 (리뷰 5개 미만)
  const hasInsufficientData = reviews.length < 5;

  return {
    overall_sentiment: overallSentiment,
    confidence: sentimentResults.reduce((acc, r) => acc + r.confidence, 0) / totalReviews,
    sentiment_distribution: {
      positive: positiveReviews,
      negative: negativeReviews,
      neutral: neutralReviews,
    },
    top_keywords: topKeywords,
    negative_concerns: negativeConcerns,
    summary: summary,
    hasInsufficientData: hasInsufficientData
  }
}
