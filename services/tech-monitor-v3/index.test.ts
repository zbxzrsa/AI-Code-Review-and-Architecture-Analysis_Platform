/**
 * 最小单测样板：验证基准候选选择逻辑。
 */
import assert from "node:assert";
import { selectBest } from "./index";

const best = selectBest([
  { id: "a", name: "A", rationale: "", latencyImprovementPct: 4, throughputImprovementPct: 5 },
  { id: "b", name: "B", rationale: "", latencyImprovementPct: 8, throughputImprovementPct: 5 },
  { id: "c", name: "C", rationale: "", latencyImprovementPct: 1, throughputImprovementPct: 2 }
]);

assert.equal(best?.id, "b", "应选择综合提升最高的候选");

const none = selectBest(
  [{ id: "d", name: "D", rationale: "", latencyImprovementPct: 2, throughputImprovementPct: 1 }],
  { minTotalImprovementPct: 10 }
);

assert.equal(none, null, "未达阈值应返回 null");

