/**
 * 最小单测样板：验证路由决策逻辑。
 */
import assert from "node:assert";
import { decideRoute } from "./index";

const exp = decideRoute({ path: "/foo", isExperimental: true });
assert.equal(exp.target, "v1");
assert.ok(exp.featureFlag);

const baseline = decideRoute({ path: "/baseline/metrics" });
assert.equal(baseline.target, "v3");

const legacy = decideRoute({ path: "/api", clientVersion: "v1.9.0" });
assert.equal(legacy.target, "v1");

const stable = decideRoute({ path: "/api/users", clientVersion: "v2.4.0" });
assert.equal(stable.target, "v2");

