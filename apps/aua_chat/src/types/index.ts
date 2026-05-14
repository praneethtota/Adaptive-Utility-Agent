export interface Session {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: number
  domain?: string
  routing_mode?: string
  u_score?: number
  latency_ms?: number
}

export interface RouteDebug {
  domain: string
  routing_mode: string
  domain_distribution: Record<string, number>
  u_score: number
  confidence: number
  latency_ms: number
  contradictions_detected: number
  specialist_responses?: Record<string, string>
  welfare_scores?: Record<string, number>  // VCG mode only
}

export interface StreamEvent {
  type: 'start' | 'route' | 'specialist_start' | 'chunk' | 'specialist_done' | 'done' | 'error'
  text?: string
  domain?: string
  routing_mode?: string
  specialist?: string
  response?: string
  u_score?: number
  latency_ms?: number
  domain_distribution?: Record<string, number>
  session_id?: string
  message_id?: string
  message?: string
}
