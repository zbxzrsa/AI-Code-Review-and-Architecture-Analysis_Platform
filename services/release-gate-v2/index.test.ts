/**
 * 最小单测样板：验证发布闸门决策逻辑。
 */
import assert from "node:assert";
import { approveRelease, GateResult } from "./index";

const ok = approveRelease(
  { v1: true, v3: true, v2: true },
  {
    p99DeltaPct: -5,
    errorRateDeltaPct: -2,
    allKnownIssuesClosed: true,
    loadTestPassed: true,
    scenarioSimPassed: true
  }
);

assert.ok(ok.approved, "应通过：性能未回退且三签齐全");

const fail: GateResult = approveRelease(
  { v1: true, v3: false, v2: true },
  {
    p99DeltaPct: 10,
    errorRateDeltaPct: 3,
    allKnownIssuesClosed: false,
    loadTestPassed: false,
    scenarioSimPassed: true
  }
);

assert.ok(!fail.approved, "应拒绝：签名缺失与性能/问题未达标");
assert.ok(
  fail.reasons.some((r) => r.includes("三重签名")),
  "应包含签名未通过的原因"
);

