/**
 * 最小单测样板：验证 v1 沙盒实验判定逻辑。
 */
import assert from "node:assert";
import { runExperiment } from "./index";

const pass = runExperiment({
  candidateId: "c1",
  cycles: 3,
  enableShadow: true,
  enableAB: true,
  knownIssuesClosed: true
});

assert.ok(pass.passes, "应通过：周期>=3 且已关闭已知问题且性能提升达标");

const fail = runExperiment({
  candidateId: "c2",
  cycles: 2,
  enableShadow: false,
  enableAB: false,
  knownIssuesClosed: false
});

assert.ok(!fail.passes, "应拒绝：周期不足且性能未达标且有已知问题");
assert.ok(fail.reasons.length >= 2, "应返回失败原因");

