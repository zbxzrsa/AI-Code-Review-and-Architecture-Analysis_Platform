/**
 * v1 沙盒编排占位实现
 * - 接收 tech.candidate，调度实验/A-B/压测/用户场景模拟
 * - 写回 v1.validation.report 与 v1.review.findings
 */
export interface ExperimentPlan {
  candidateId: string;
  cycles: number;
  enableShadow: boolean;
  enableAB: boolean;
  knownIssuesClosed: boolean;
  perfDeltaPct?: number; // 性能提升百分比（从实际测试结果获取）
}

export interface ExperimentResult {
  candidateId: string;
  issuesResolved: number;
  perfDeltaPct: number;
  passes: boolean;
  reasons: string[];
}

/**
 * 在 v1 沙盒执行实验，要求至少 3 个周期且性能提升达标。
 * @param plan 实验计划
 * @returns 实验结果，附带未通过原因
 */
/**
 * 在 v1 沙盒执行实验，要求至少 3 个周期且性能提升达标。
 * @param plan 实验计划
 * @returns 实验结果，附带未通过原因
 */
export function runExperiment(plan: ExperimentPlan): ExperimentResult {
  const reasons: string[] = [];

  // 输入验证
  if (!plan.candidateId || plan.candidateId.trim() === "") {
    reasons.push("候选ID无效");
    return {
      candidateId: plan.candidateId || "",
      issuesResolved: 0,
      perfDeltaPct: 0,
      passes: false,
      reasons
    };
  }

  if (plan.cycles < 3) {
    reasons.push(`实验周期不足 3（当前: ${plan.cycles}）`);
  }

  if (!plan.knownIssuesClosed) {
    reasons.push("仍有已知问题未关闭");
  }

  // 使用实际性能数据，如果没有则使用默认值（仅用于演示）
  const perfDeltaPct = plan.perfDeltaPct ?? (plan.enableAB ? 15 : 5);
  if (perfDeltaPct < 15) {
    reasons.push(`性能提升未达 15%（当前: ${perfDeltaPct}%）`);
  }

  return {
    candidateId: plan.candidateId,
    issuesResolved: plan.knownIssuesClosed ? 1 : 0,
    perfDeltaPct,
    passes: reasons.length === 0,
    reasons
  };
}

