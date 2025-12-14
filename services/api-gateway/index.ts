/**
 * API 网关占位实现
 * - 统一对外 API，保持向后兼容
 * - 路由到 v1/v2/v3 对应服务域
 * - 提供特性开关/灰度支持
 */
export interface RouteDecision {
  target: "v1" | "v2" | "v3";
  featureFlag?: string;
  reason: string;
}

export interface RequestContext {
  path: string;
  /**
   * 标记是否为实验/灰度请求（如 header/flag 指定）。
   */
  isExperimental?: boolean;
  /**
   * 客户端声明的 API 版本，用于兼容性判断。
   */
  clientVersion?: string;
}

const BASELINE_PREFIX = "/baseline";

/**
 * 基于请求上下文做版本路由决策，保持向后兼容。
 * @param ctx 请求上下文
 * @returns 路由决策：目标版本、特性开关、原因
 */
export function decideRoute(ctx: RequestContext): RouteDecision {
  const { path, isExperimental, clientVersion } = ctx;

  if (isExperimental) {
    return { target: "v1", featureFlag: "exp-enabled", reason: "实验流量" };
  }

  if (path.startsWith(BASELINE_PREFIX)) {
    return { target: "v3", reason: "基准/对比路径" };
  }

  if (clientVersion && clientVersion.startsWith("v3")) {
    return { target: "v3", reason: "客户端声明基线版本" };
  }

  if (clientVersion && clientVersion.startsWith("v1")) {
    return { target: "v1", featureFlag: "legacy-dev", reason: "客户端要求 v1" };
  }

  return { target: "v2", reason: "默认稳定版" };
}

