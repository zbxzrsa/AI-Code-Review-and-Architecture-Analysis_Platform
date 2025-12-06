#!/bin/bash
# Check Rollout Health
# Verifies SLOs during gray-scale rollout phases
#
# Usage: ./check_rollout_health.sh <rollout-name> <namespace>

set -e

ROLLOUT_NAME=${1:-"vcai-rollout"}
NAMESPACE=${2:-"platform-v2-stable"}
PROMETHEUS_URL=${PROMETHEUS_URL:-"http://prometheus.platform-monitoring.svc:9090"}

# Thresholds
P95_LATENCY_THRESHOLD_MS=3000
ERROR_RATE_THRESHOLD=0.02
MIN_SUCCESS_RATE=0.98

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Rollout Health Check                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ Rollout: $ROLLOUT_NAME"
echo "║ Namespace: $NAMESPACE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to query Prometheus
query_prometheus() {
    local query="$1"
    local result
    result=$(curl -s "${PROMETHEUS_URL}/api/v1/query" \
        --data-urlencode "query=${query}" | \
        jq -r '.data.result[0].value[1] // "NaN"')
    echo "$result"
    return 0
}

# Get rollout status
echo "📊 Checking rollout status..."
ROLLOUT_STATUS=$(kubectl argo rollouts status "$ROLLOUT_NAME" -n "$NAMESPACE" --timeout=5s 2>/dev/null || echo "Unknown")
echo "   Status: $ROLLOUT_STATUS"

# Get current canary weight
CANARY_WEIGHT=$(kubectl get rollout "$ROLLOUT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.canary.weights.canary}' 2>/dev/null || echo "0")
echo "   Canary Weight: ${CANARY_WEIGHT}%"
echo ""

# Check P95 Latency
echo "⏱️  Checking P95 Latency..."
LATENCY_QUERY="histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{namespace=\"${NAMESPACE}\",rollout_type=\"canary\"}[5m])) by (le)) * 1000"
P95_LATENCY=$(query_prometheus "$LATENCY_QUERY")

if [[ "$P95_LATENCY" == "NaN" ]] || [[ -z "$P95_LATENCY" ]]; then
    echo "   ⚠️  No latency data available (may be early in rollout)"
    LATENCY_OK=true
else
    P95_LATENCY_INT=$(printf "%.0f" "$P95_LATENCY")
    if [[ "$P95_LATENCY_INT" -le "$P95_LATENCY_THRESHOLD_MS" ]]; then
        echo "   ✅ P95 Latency: ${P95_LATENCY_INT}ms (threshold: ${P95_LATENCY_THRESHOLD_MS}ms)"
        LATENCY_OK=true
    else
        echo "   ❌ P95 Latency: ${P95_LATENCY_INT}ms EXCEEDS threshold: ${P95_LATENCY_THRESHOLD_MS}ms" >&2
        LATENCY_OK=false
    fi
fi
echo ""

# Check Error Rate
echo "🔴 Checking Error Rate..."
ERROR_QUERY="sum(rate(http_requests_total{namespace=\"${NAMESPACE}\",rollout_type=\"canary\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{namespace=\"${NAMESPACE}\",rollout_type=\"canary\"}[5m]))"
ERROR_RATE=$(query_prometheus "$ERROR_QUERY")

if [[ "$ERROR_RATE" == "NaN" ]] || [[ -z "$ERROR_RATE" ]]; then
    echo "   ⚠️  No error rate data available"
    ERROR_OK=true
else
    ERROR_RATE_PCT=$(echo "$ERROR_RATE * 100" | bc -l 2>/dev/null || echo "0")
    THRESHOLD_PCT=$(echo "$ERROR_RATE_THRESHOLD * 100" | bc -l)
    
    if (( $(echo "$ERROR_RATE <= $ERROR_RATE_THRESHOLD" | bc -l) )); then
        printf "   ✅ Error Rate: %.2f%% (threshold: %.2f%%)\n" "$ERROR_RATE_PCT" "$THRESHOLD_PCT"
        ERROR_OK=true
    else
        printf "   ❌ Error Rate: %.2f%% EXCEEDS threshold: %.2f%%\n" "$ERROR_RATE_PCT" "$THRESHOLD_PCT" >&2
        ERROR_OK=false
    fi
fi
echo ""

# Check Success Rate
echo "✅ Checking Success Rate..."
SUCCESS_QUERY="sum(rate(http_requests_total{namespace=\"${NAMESPACE}\",rollout_type=\"canary\",status=~\"2..\"}[5m])) / sum(rate(http_requests_total{namespace=\"${NAMESPACE}\",rollout_type=\"canary\"}[5m]))"
SUCCESS_RATE=$(query_prometheus "$SUCCESS_QUERY")

if [[ "$SUCCESS_RATE" == "NaN" ]] || [[ -z "$SUCCESS_RATE" ]]; then
    echo "   ⚠️  No success rate data available"
    SUCCESS_OK=true
else
    SUCCESS_RATE_PCT=$(echo "$SUCCESS_RATE * 100" | bc -l 2>/dev/null || echo "100")
    MIN_SUCCESS_PCT=$(echo "$MIN_SUCCESS_RATE * 100" | bc -l)
    
    if (( $(echo "$SUCCESS_RATE >= $MIN_SUCCESS_RATE" | bc -l) )); then
        printf "   ✅ Success Rate: %.2f%% (minimum: %.2f%%)\n" "$SUCCESS_RATE_PCT" "$MIN_SUCCESS_PCT"
        SUCCESS_OK=true
    else
        printf "   ❌ Success Rate: %.2f%% BELOW minimum: %.2f%%\n" "$SUCCESS_RATE_PCT" "$MIN_SUCCESS_PCT"
        SUCCESS_OK=false
    fi
fi
echo ""

# Check Request Volume
echo "📈 Checking Request Volume..."
VOLUME_QUERY="sum(rate(http_requests_total{namespace=\"${NAMESPACE}\",rollout_type=\"canary\"}[5m]))"
REQUEST_VOLUME=$(query_prometheus "$VOLUME_QUERY")

if [[ "$REQUEST_VOLUME" == "NaN" ]] || [[ -z "$REQUEST_VOLUME" ]]; then
    echo "   ⚠️  No request volume data"
else
    printf "   📊 Current RPS: %.2f\n" "$REQUEST_VOLUME"
fi
echo ""

# Check Pod Health
echo "🏥 Checking Pod Health..."
READY_PODS=$(kubectl get pods -n "$NAMESPACE" -l "app=$ROLLOUT_NAME,rollout-type=canary" -o json 2>/dev/null | \
    jq '[.items[] | select(.status.phase=="Running" and (.status.conditions[] | select(.type=="Ready" and .status=="True")))] | length')
TOTAL_PODS=$(kubectl get pods -n "$NAMESPACE" -l "app=$ROLLOUT_NAME,rollout-type=canary" -o json 2>/dev/null | \
    jq '.items | length')

echo "   Ready Pods: ${READY_PODS:-0}/${TOTAL_PODS:-0}"

if [[ "${READY_PODS:-0}" -gt 0 ]]; then
    POD_HEALTH_OK=true
    echo "   ✅ Pods are healthy"
else
    if [[ "$CANARY_WEIGHT" -gt 0 ]]; then
        POD_HEALTH_OK=false
        echo "   ❌ No ready pods for canary"
    else
        POD_HEALTH_OK=true
        echo "   ⚠️  Canary not yet deployed"
    fi
fi
echo ""

# Check Argo Analysis
echo "🔬 Checking Argo Analysis Runs..."
ANALYSIS_STATUS=$(kubectl get analysisrun -n "$NAMESPACE" -l "rollout-name=$ROLLOUT_NAME" \
    -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "None")
echo "   Analysis Status: $ANALYSIS_STATUS"

case "$ANALYSIS_STATUS" in
    "Successful")
        echo "   ✅ Analysis passed"
        ANALYSIS_OK=true
        ;;
    "Failed")
        echo "   ❌ Analysis failed"
        ANALYSIS_OK=false
        ;;
    "Running")
        echo "   ⏳ Analysis in progress"
        ANALYSIS_OK=true
        ;;
    *)
        echo "   ⚠️  No analysis running"
        ANALYSIS_OK=true
        ;;
esac
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      HEALTH SUMMARY                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"

OVERALL_HEALTHY=true

if [[ "$LATENCY_OK" == true ]]; then
    echo "║ ✅ Latency Check      : PASSED                              ║"
else
    echo "║ ❌ Latency Check      : FAILED                              ║" >&2
    OVERALL_HEALTHY=false
fi

if [[ "$ERROR_OK" == true ]]; then
    echo "║ ✅ Error Rate Check   : PASSED                              ║"
else
    echo "║ ❌ Error Rate Check   : FAILED                              ║" >&2
    OVERALL_HEALTHY=false
fi

if [[ "$SUCCESS_OK" == true ]]; then
    echo "║ ✅ Success Rate Check : PASSED                              ║"
else
    echo "║ ❌ Success Rate Check : FAILED                              ║"
    OVERALL_HEALTHY=false
fi

if [[ "$POD_HEALTH_OK" == true ]]; then
    echo "║ ✅ Pod Health Check   : PASSED                              ║"
else
    echo "║ ❌ Pod Health Check   : FAILED                              ║"
    OVERALL_HEALTHY=false
fi

if [[ "$ANALYSIS_OK" == true ]]; then
    echo "║ ✅ Analysis Check     : PASSED                              ║"
else
    echo "║ ❌ Analysis Check     : FAILED                              ║"
    OVERALL_HEALTHY=false
fi

echo "╠══════════════════════════════════════════════════════════════╣"

if [[ "$OVERALL_HEALTHY" == true ]]; then
    echo "║ 🎉 OVERALL STATUS     : HEALTHY                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "║ 🚨 OVERALL STATUS     : UNHEALTHY                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "⚠️  Rollout health check failed. Consider:"
    echo "   1. Pausing the rollout: kubectl argo rollouts pause $ROLLOUT_NAME -n $NAMESPACE"
    echo "   2. Aborting the rollout: kubectl argo rollouts abort $ROLLOUT_NAME -n $NAMESPACE"
    echo "   3. Reviewing logs: kubectl logs -n $NAMESPACE -l rollout-type=canary"
    exit 1
fi
