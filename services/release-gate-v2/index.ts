/**
 * v2 发布闸门占位实现
 * - 聚合三重 AI 签名、预生产报告，决定是否放量
 * - 驱动 Argo Rollouts 执行蓝绿/金丝雀与分钟级回滚
 */
export interface Signatures {
  v1: boolean;
  v3: boolean;
  v2: boolean;
}

export interface PreprodReport {
  p99DeltaPct: number;
  errorRateDeltaPct: number;
  allKnownIssuesClosed: boolean;
  loadTestPassed?: boolean;
  scenarioSimPassed?: boolean;
}

export interface GateThresholds {
  maxP99RegressionPct: number;
  maxErrorRateRegressionPct: number;
  requireIssuesClosed: boolean;
  requirePreprodPass: boolean;
}

const defaultThresholds: GateThresholds = {
  maxP99RegressionPct: 0,
  maxErrorRateRegressionPct: 0,
  requireIssuesClosed: true,
  requirePreprodPass: true
};

export interface GateResult {
  approved: boolean;
  reasons: string[];
}

/**
 * 根据三重签名与预生产报告判断是否放量。
 * @param sigs 三重 AI 签名结果
 * @param report 预生产压测与场景模拟报告
 * @param thresholds 可配置阈值，默认不接受性能回退与错误率上升
 * @returns GateResult，包含是否通过与未通过原因
 */
export function approveRelease(
  sigs: Signatures,
  report: PreprodReport,
  thresholds: GateThresholds = defaultThresholds
): GateResult {
  const reasons: string[] = [];

  // 输入验证
  if (!sigs || typeof sigs !== 'object') {
    reasons.push("签名数据无效");
    return { approved: false, reasons };
  }

  if (!report || typeof report !== 'object') {
    reasons.push("预生产报告无效");
    return { approved: false, reasons };
  }

  if (!thresholds || typeof thresholds !== 'object') {
    thresholds = defaultThresholds;
  }

  // 验证数值有效性
  if (typeof report.p99DeltaPct !== 'number' || isNaN(report.p99DeltaPct)) {
    reasons.push("p99DeltaPct 无效");
  }

  if (typeof report.errorRateDeltaPct !== 'number' || isNaN(report.errorRateDeltaPct)) {
    reasons.push("errorRateDeltaPct 无效");
  }

  if (reasons.length > 0) {
    return { approved: false, reasons };
  }

  // 三重签名检查
  if (!(sigs.v1 && sigs.v3 && sigs.v2)) {
    reasons.push("三重签名未全部通过");
  }

  // 性能回退检查
  if (report.p99DeltaPct > thresholds.maxP99RegressionPct) {
    reasons.push(`p99 回退 ${report.p99DeltaPct.toFixed(2)}% 超过阈值 ${thresholds.maxP99RegressionPct}%`);
  }

  if (report.errorRateDeltaPct > thresholds.maxErrorRateRegressionPct) {
    reasons.push(`错误率回退 ${report.errorRateDeltaPct.toFixed(2)}% 超过阈值 ${thresholds.maxErrorRateRegressionPct}%`);
  }

  // 已知问题检查
  if (thresholds.requireIssuesClosed && !report.allKnownIssuesClosed) {
    reasons.push("仍有已知问题未关闭");
  }

  // 预生产测试检查
  if (thresholds.requirePreprodPass) {
    if (report.loadTestPassed === false) {
      reasons.push("预生产压测未通过");
    }
    if (report.scenarioSimPassed === false) {
      reasons.push("用户场景模拟未通过");
    }
  }

  return {
    approved: reasons.length === 0,
    reasons
  };
}

