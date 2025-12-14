import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 10,
  duration: "2m",
  thresholds: {
    http_req_duration: ["p(99)<500"],
    http_req_failed: ["rate<0.01"]
  }
};

/**
 * Shadow 流量压测：命中 v1 沙盒与 v2 稳定，比较响应差异。
 */
export default function run() {
  const resV2 = http.get("https://gateway.example.com/api");
  const resV1 = http.get("https://gateway.example.com/experimental");
  check(resV2, { "v2 ok": (r) => r.status === 200 });
  check(resV1, { "v1 ok": (r) => r.status === 200 });
  sleep(1);
}

