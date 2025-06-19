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

// 주제별 키워드 분류
const TOPIC_KEYWORDS = {
  quality: ["quality", "build", "material", "durable", "sturdy", "solid", "construction", "craftsmanship"],
  price: ["price", "cost", "expensive", "cheap", "value", "money", "worth", "affordable", "budget"],
  shipping: ["shipping", "delivery", "fast", "slow", "arrived", "package", "packaging", "box"],
  service: ["service", "support", "help", "response", "customer", "staff", "team", "assistance"],
  design: ["design", "look", "appearance", "color", "style", "beautiful", "attractive", "aesthetic"],
  performance: ["performance", "speed", "efficiency", "power", "battery", "function", "work", "operation"],
  usability: ["easy", "difficult", "simple", "complex", "user", "interface", "setup", "install"],
}

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

// 주제별 분류 분석
export function categorizeReviews(reviews: string[]) {
  const topics: { [key: string]: { positive: number; negative: number; mentions: string[] } } = {}

  Object.keys(TOPIC_KEYWORDS).forEach((topic) => {
    topics[topic] = {
      positive: 0,
      negative: 0,
      mentions: [],
    }
  })

  reviews.forEach((review) => {
    const sentiment = analyzeSentiment(review)
    const words = review.toLowerCase().split(/\s+/)

    Object.entries(TOPIC_KEYWORDS).forEach(([topic, keywords]) => {
      const hasKeyword = keywords.some((keyword) =>
        words.some((word) => word.includes(keyword) || keyword.includes(word)),
      )

      if (hasKeyword) {
        if (sentiment.sentiment === "positive") {
          topics[topic].positive++
        } else if (sentiment.sentiment === "negative") {
          topics[topic].negative++
        }

        // 관련 문장 추출 (최대 100자)
        const sentence = review.substring(0, 100) + (review.length > 100 ? "..." : "")
        if (topics[topic].mentions.length < 3) {
          topics[topic].mentions.push(sentence)
        }
      }
    })
  })

  return topics
}

// 긍정적 하이라이트 추출
export function extractPositiveHighlights(reviews: string[]) {
  return reviews
    .map((review) => ({ review, sentiment: analyzeSentiment(review) }))
    .filter(({ sentiment }) => sentiment.sentiment === "positive" && sentiment.confidence > 0.3)
    .sort((a, b) => b.sentiment.positive - a.sentiment.positive)
    .slice(0, 3)
    .map(({ review }) => (review.length > 120 ? review.substring(0, 120) + "..." : review))
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
      topic_analysis: {},
      positive_highlights: [],
      negative_concerns: [],
      summary: "분석할 리뷰가 없습니다.",
      hasInsufficientData: true,
    }
  }

  // 1. 감정 분석
  const sentiments = reviews.map((review) => analyzeSentiment(review))
  const positiveCount = sentiments.filter((s) => s.sentiment === "positive").length
  const negativeCount = sentiments.filter((s) => s.sentiment === "negative").length
  const neutralCount = sentiments.filter((s) => s.sentiment === "neutral").length

  // 2. 키워드 추출
  const keywords = extractKeywords(reviews)

  // 3. 주제별 분류
  const topics = categorizeReviews(reviews)

  // 4. 하이라이트 추출
  const positiveHighlights = extractPositiveHighlights(reviews)
  const negativeConcerns = extractNegativeConcerns(reviews)

  // 5. 전체 감정 판단
  const totalReviews = reviews.length
  const overallSentiment =
    positiveCount > negativeCount ? "positive" : negativeCount > positiveCount ? "negative" : "neutral"

  const confidence = totalReviews > 0 ? Math.abs(positiveCount - negativeCount) / totalReviews : 0

  // 6. 요약 생성
  const positivePercentage = Math.round((positiveCount / totalReviews) * 100)
  const negativePercentage = Math.round((negativeCount / totalReviews) * 100)

  let summary = ""
  if (overallSentiment === "positive") {
    summary = `전체 리뷰의 ${positivePercentage}%가 긍정적입니다. 고객들이 전반적으로 만족하는 제품입니다.`
  } else if (overallSentiment === "negative") {
    summary = `전체 리뷰의 ${negativePercentage}%가 부정적입니다. 구매 전 신중한 검토가 필요합니다.`
  } else {
    summary = `긍정적 의견과 부정적 의견이 비슷한 수준입니다. 개인의 선호도에 따라 만족도가 달라질 수 있습니다.`
  }

  return {
    overall_sentiment: overallSentiment,
    confidence,
    sentiment_distribution: {
      positive: positiveCount,
      negative: negativeCount,
      neutral: neutralCount,
    },
    top_keywords: keywords,
    topic_analysis: topics,
    positive_highlights: positiveHighlights,
    negative_concerns: negativeConcerns,
    summary,
    hasInsufficientData: totalReviews <= 5, // 5개 이하면 데이터 부족으로 표시
  }
}
