/**
 * AI 网关占位实现
 * - 负责模型路由/融合：主模型 + 安全审查模型 + 回归风险模型 + 轻量补全 rerank
 * - 提供统一 REST/gRPC 接口给 v1/v2/v3 域
 * - 所有跨域调用需 mTLS + ACL，仅传递必要元数据
 */
export interface AiRequest {
  task: "review" | "diagnosis" | "generation";
  payload: unknown;
  sensitivity: "low" | "medium" | "high";
  versionScope: "v1" | "v2" | "v3";
}

export interface AiResponse {
  decision: string;
  modelUsed: string;
  riskScore: number;
  traceId?: string;
  blocked?: boolean;
  reason?: string;
}

const HIGH_RISK_THRESHOLD = 0.2;

/**
 * 路由策略入口（示例）
 * - 高敏任务需强制安全审查模型
 * - 返回是否阻断、使用模型与风险评分
 */
export function routeRequest(req: AiRequest): AiResponse {
  // 输入验证
  if (!req || typeof req !== 'object') {
    return {
      decision: "error",
      modelUsed: "none",
      riskScore: 1.0,
      blocked: true,
      reason: "无效请求"
    };
  }

  if (!req.task || !['review', 'diagnosis', 'generation'].includes(req.task)) {
    return {
      decision: "error",
      modelUsed: "none",
      riskScore: 1.0,
      blocked: true,
      reason: "无效任务类型"
    };
  }

  if (!req.sensitivity || !['low', 'medium', 'high'].includes(req.sensitivity)) {
    return {
      decision: "error",
      modelUsed: "none",
      riskScore: 1.0,
      blocked: true,
      reason: "无效敏感度级别"
    };
  }

  const riskScore = req.sensitivity === "high" ? 0.25 : req.sensitivity === "medium" ? 0.1 : 0.05;
  
  if (riskScore >= HIGH_RISK_THRESHOLD) {
    return {
      decision: "blocked",
      modelUsed: "safety-guard",
      riskScore,
      blocked: true,
      reason: "高敏任务需安全审查",
      traceId: `trace-${Date.now()}`
    };
  }

  return {
    decision: "placeholder-response",
    modelUsed: "primary-model",
    riskScore,
    traceId: `trace-${Date.now()}`,
    blocked: false
  };
}

