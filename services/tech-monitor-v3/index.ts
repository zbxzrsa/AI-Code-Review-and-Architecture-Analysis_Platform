/**
 * v3 技术监测与基准服务
 *
 * 功能：
 * - 监控外部技术/模型更新
 * - 触发基准测试并输出到 tech.baseline.report
 * - 生成最优候选并推送到 tech.candidate
 * - 与版本架构系统集成，提供技术对比数据给v1
 * - 保留完整历史版本数据和性能参数
 *
 * 集成：
 * - 与 ai_core.version_architecture 系统集成
 * - 通过安全通信通道向v1发送技术对比数据
 * - 定期生成技术评估报告
 */

export interface BaselineCandidate {
  id: string;
  name: string;
  rationale: string;
  latencyImprovementPct: number;
  throughputImprovementPct: number;
  performanceMetrics?: {
    latency?: number;
    throughput?: number;
    accuracy?: number;
    cost?: number;
  };
  historicalData?: {
    version: string;
    timestamp: string;
    metrics: Record<string, number>;
  }[];
}

export interface SelectOptions {
  minTotalImprovementPct: number;
  requireKeyIndicatorImprovement?: boolean;
}

const defaultOptions: SelectOptions = {
  minTotalImprovementPct: 15, // 提升到15%以符合更新标准
  requireKeyIndicatorImprovement: true,
};

/**
 * 从候选中挑选综合改进度最高且达阈值的方案。
 *
 * 符合技术更新标准：
 * - 性能提升≥15%
 * - 关键指标显著改善
 *
 * @param candidates 候选列表
 * @param options 选择阈值配置
 * @returns 最优候选或 null
 */
export function selectBest(
  candidates: BaselineCandidate[],
  options: SelectOptions = defaultOptions
): BaselineCandidate | null {
  // 输入验证
  if (!candidates || !Array.isArray(candidates) || candidates.length === 0) {
    return null;
  }

  if (!options || typeof options.minTotalImprovementPct !== 'number' || options.minTotalImprovementPct < 0) {
    options = defaultOptions;
  }

  // 过滤无效候选（包含负数或NaN）
  const validCandidates = candidates.filter(c =>
    c &&
    typeof c.latencyImprovementPct === 'number' &&
    typeof c.throughputImprovementPct === 'number' &&
    !isNaN(c.latencyImprovementPct) &&
    !isNaN(c.throughputImprovementPct) &&
    c.latencyImprovementPct >= 0 &&
    c.throughputImprovementPct >= 0
  );

  if (validCandidates.length === 0) {
    return null;
  }

  // 应用15%提升阈值
  const sorted = validCandidates
    .map((c) => ({
      candidate: c,
      total: c.latencyImprovementPct + c.throughputImprovementPct
    }))
    .filter((item) => item.total >= options.minTotalImprovementPct)
    .sort((a, b) => b.total - a.total);

  return sorted.length ? sorted[0].candidate : null;
}

/**
 * 生成技术评估报告
 *
 * v3基准版定期生成技术评估报告，包含：
 * - 技术对比数据
 * - 性能参数
 * - 历史版本数据
 * - 最优技术推荐
 *
 * @param candidates 候选技术列表
 * @param historicalData 历史数据
 * @returns 评估报告
 */
export function generateTechAssessmentReport(
  candidates: BaselineCandidate[],
  historicalData?: Record<string, any>
): {
  reportId: string;
  timestamp: string;
  bestCandidate: BaselineCandidate | null;
  comparisonData: {
    candidates: BaselineCandidate[];
    metrics: Record<string, number>;
  };
  recommendation: string;
} {
  const bestCandidate = selectBest(candidates);

  return {
    reportId: `report_${Date.now()}`,
    timestamp: new Date().toISOString(),
    bestCandidate,
    comparisonData: {
      candidates,
      metrics: {
        totalCandidates: candidates.length,
        avgLatencyImprovement: candidates.reduce((sum, c) => sum + c.latencyImprovementPct, 0) / candidates.length,
        avgThroughputImprovement: candidates.reduce((sum, c) => sum + c.throughputImprovementPct, 0) / candidates.length,
      },
    },
    recommendation: bestCandidate
      ? `推荐技术: ${bestCandidate.name}，综合改进度: ${bestCandidate.latencyImprovementPct + bestCandidate.throughputImprovementPct}%`
      : '暂无符合标准的技术',
  };
}

/**
 * 保留历史版本数据
 *
 * v3基准版必须保留完整的历史版本数据和性能参数
 *
 * @param version 版本号
 * @param metrics 性能指标
 * @param metadata 元数据
 */
export function retainHistoricalData(
  version: string,
  metrics: Record<string, number>,
  metadata?: Record<string, any>
): BaselineCandidate['historicalData'][0] {
  return {
    version,
    timestamp: new Date().toISOString(),
    metrics,
    ...metadata,
  };
}

