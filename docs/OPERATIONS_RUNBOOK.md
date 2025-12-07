# Operations Runbook

## AI Code Review Platform - Self-Healing System

**Version:** 1.0.0  
**Last Updated:** December 7, 2024  
**On-Call:** team@ai-code-review.dev

---

## Quick Reference

### Emergency Contacts

- **On-Call Engineer:** PagerDuty rotation
- **Team Lead:** team@ai-code-review.dev
- **Security:** security@ai-code-review.dev

### Critical URLs

- **Production:** https://api.coderev.example.com
- **Monitoring:** https://grafana.coderev.example.com
- **Status Page:** https://status.coderev.example.com
- **Docs:** https://docs.coderev.example.com

### Quick Commands

```bash
# System health
curl https://api.coderev.example.com/healthz

# Self-healing status
curl https://api.coderev.example.com/api/admin/self-healing/stats

# Trigger manual repair
python scripts/manual_repair.py --action restart_service --service analysis

# Generate health report
python scripts/version_health_report.py --daily

# Rollback deployment
./scripts/rollback.sh --to v2.0.5
```

---

## Self-Healing System Operations

### 1. Monitoring Dashboard

**Grafana Dashboards:**

- System Overview: http://localhost:3001/d/system-overview
- Self-Healing: http://localhost:3001/d/self-healing
- Version Health: http://localhost:3001/d/version-health

**Key Panels:**

- System Health Score (0-100)
- Active Issues Count
- Repair Success Rate
- Alert Timeline
- Resource Usage
- Error Rate by Type

### 2. Alert Response

#### Critical Alerts (PagerDuty + Slack)

**Alert: "System Health Critical"**

- **Trigger:** Health score < 50
- **Response Time:** Immediate (< 2 min)
- **Actions:**
  1. Check Grafana dashboard
  2. Review recent deployments
  3. Check error logs
  4. Consider rollback
  5. Escalate if not resolved in 15min

**Alert: "High Error Rate"**

- **Trigger:** Error rate > 5%
- **Response Time:** Immediate
- **Actions:**
  1. Check error logs for patterns
  2. Identify affected endpoints
  3. Check recent changes
  4. Rollback if deployment-related
  5. Scale up if load-related

**Alert: "Memory Critical"**

- **Trigger:** Memory > 95%
- **Response Time:** Immediate
- **Actions:**
  1. Check memory usage by service
  2. Look for memory leaks
  3. Restart affected service
  4. Monitor for recurrence
  5. Investigate root cause

#### Warning Alerts (Slack + Email)

**Alert: "Response Time Degraded"**

- **Trigger:** p95 > 2s
- **Response Time:** < 15 min
- **Actions:**
  1. Check system load
  2. Review slow query logs
  3. Check cache hit rate
  4. Consider scaling up
  5. Optimize if needed

**Alert: "Circuit Breaker Opened"**

- **Trigger:** External service failures
- **Response Time:** < 30 min
- **Actions:**
  1. Identify affected service
  2. Check service status
  3. Verify auto-recovery
  4. Manual intervention if needed
  5. Update runbook

### 3. Auto-Repair Actions

#### Restart Service

**When:** Service crash, memory leak, deadlock
**Duration:** < 30 seconds
**Verification:**

```bash
# Check service status
kubectl get pods -n platform-v2

# Verify health
curl http://service-url/healthz
```

#### Scale Up

**When:** High CPU/memory, queue overflow
**Duration:** 30-60 seconds
**Verification:**

```bash
# Check replica count
kubectl get deployment -n platform-v2

# Monitor metrics
watch -n 1 'curl -s http://localhost:9090/api/v1/query?query=cpu_usage'
```

#### Rollback Version

**When:** Deployment failure, high error rate
**Duration:** < 5 minutes
**Verification:**

```bash
# Check deployed version
kubectl get deployment analysis-service -o yaml | grep image

# Verify health
./scripts/health_check.sh
```

#### Clear Cache

**When:** Cache corruption, memory pressure
**Duration:** < 10 seconds
**Verification:**

```bash
# Check cache size
redis-cli INFO memory

# Verify cache hit rate
curl http://localhost:9090/api/v1/query?query=cache_hit_rate
```

#### Drain Queue

**When:** Queue overflow, backpressure
**Duration:** < 60 seconds
**Verification:**

```bash
# Check queue size
redis-cli LLEN analysis_queue

# Monitor processing rate
watch -n 1 'redis-cli LLEN analysis_queue'
```

### 4. Manual Intervention

#### When Auto-Repair Fails

**Scenario:** Repair attempted but failed

**Actions:**

1. Check repair logs:

   ```bash
   kubectl logs -n platform-v2 -l app=self-healing --tail=100
   ```

2. Review repair history:

   ```bash
   curl http://localhost:8000/api/admin/self-healing/repairs
   ```

3. Identify failure reason

4. Execute manual repair:

   ```bash
   python scripts/manual_repair.py --action <action> --service <service>
   ```

5. Verify recovery

6. Update runbook if new scenario

#### When Multiple Systems Affected

**Scenario:** Cascading failures across services

**Actions:**

1. **Immediate:** Stop auto-repair to prevent chaos

   ```bash
   curl -X POST http://localhost:8000/api/admin/self-healing/disable
   ```

2. **Assess:** Identify root cause

   - Check recent deployments
   - Review system logs
   - Check external dependencies

3. **Isolate:** Disable affected services

   ```bash
   kubectl scale deployment <service> --replicas=0
   ```

4. **Fix:** Apply targeted fix

   - Rollback deployment
   - Fix configuration
   - Restart services

5. **Verify:** Test recovery

   ```bash
   ./scripts/health_check.sh --comprehensive
   ```

6. **Re-enable:** Turn auto-repair back on
   ```bash
   curl -X POST http://localhost:8000/api/admin/self-healing/enable
   ```

---

## Version Management Operations

### Daily Health Report

**Schedule:** Every day at 9:00 AM

**Command:**

```bash
python scripts/version_health_report.py --daily --output reports/health-$(date +%Y%m%d).txt
```

**Review:**

1. Check all versions are healthy
2. Review error rates
3. Check user counts
4. Verify uptime targets
5. Escalate if issues

### Version Migration

**Before Migration:**

1. Review migration guide
2. Test in staging
3. Backup current version
4. Notify users
5. Prepare rollback plan

**Execute Migration:**

```bash
# Migrate specific module
python scripts/migrate_version.py --from v2.0.0 --to v2.1.0 --module auth

# Migrate all modules
python scripts/migrate_version.py --from v2.0.0 --to v2.1.0 --all

# Dry run first
python scripts/migrate_version.py --from v2.0.0 --to v2.1.0 --all --dry-run
```

**After Migration:**

1. Run health checks
2. Verify functionality
3. Monitor for 24 hours
4. Collect user feedback
5. Document lessons learned

### Spiral Iteration

**Phase 1: Introduce to V1**

```bash
# Deploy new feature to V1
kubectl apply -f kubernetes/deployments/v1-new-feature.yaml

# Monitor V1 metrics
watch -n 5 'curl -s http://v1-api/metrics | grep error_rate'
```

**Phase 2: Verify in V3**

```bash
# Run negative tests in V3
pytest tests/v3/test_negative_scenarios.py

# Check compatibility
python scripts/check_compatibility.py --version v3 --feature new_feature
```

**Phase 3: Promote to V2**

```bash
# Check quality gates
python scripts/check_quality_gates.py --feature new_feature

# If passed, promote
python scripts/promote_feature.py --from v1 --to v2 --feature new_feature
```

**Phase 4: Optimize**

```bash
# Collect feedback
python scripts/collect_feedback.py --version v2 --feature new_feature

# Generate optimization report
python scripts/optimize_feature.py --feature new_feature
```

---

## Troubleshooting

### Common Issues

#### Issue: Self-Healing Not Working

**Symptoms:**

- No auto-repairs triggered
- Alerts not generated
- Health checks not running

**Diagnosis:**

```bash
# Check orchestrator status
curl http://localhost:8000/api/admin/self-healing/status

# Check logs
kubectl logs -n platform-v2 -l app=self-healing

# Verify configuration
cat /etc/self-healing/config.yaml
```

**Resolution:**

1. Restart self-healing service
2. Verify configuration
3. Check permissions
4. Review logs for errors

#### Issue: High Repair Failure Rate

**Symptoms:**

- Repair success rate < 80%
- Multiple repair attempts
- Manual interventions increasing

**Diagnosis:**

```bash
# Check repair statistics
curl http://localhost:8000/api/admin/self-healing/repairs/stats

# Review failed repairs
curl http://localhost:8000/api/admin/self-healing/repairs?status=failed
```

**Resolution:**

1. Identify failing repair types
2. Check repair logs
3. Verify system resources
4. Update repair strategies
5. Test in staging

#### Issue: Alert Fatigue

**Symptoms:**

- Too many alerts
- High deduplication rate
- Ignored alerts

**Diagnosis:**

```bash
# Check alert statistics
curl http://localhost:8000/api/admin/alerts/stats

# Review alert frequency
curl http://localhost:8000/api/admin/alerts?last=24h
```

**Resolution:**

1. Review alert thresholds
2. Increase dedup window
3. Adjust severity levels
4. Add alert grouping
5. Update routing rules

---

## Maintenance Tasks

### Daily

- [ ] Review health reports
- [ ] Check self-healing stats
- [ ] Verify alert delivery
- [ ] Review error logs
- [ ] Check resource usage

### Weekly

- [ ] Analyze repair patterns
- [ ] Review false positives
- [ ] Update thresholds
- [ ] Test rollback procedures
- [ ] Update documentation

### Monthly

- [ ] Comprehensive health review
- [ ] Chaos engineering tests
- [ ] Performance benchmarks
- [ ] Security scans
- [ ] Capacity planning

---

## Escalation Procedures

### Level 1: Auto-Repair

- **Duration:** 0-5 minutes
- **Action:** Automated repair attempt
- **Notification:** Info alert to Slack

### Level 2: On-Call Engineer

- **Duration:** 5-15 minutes
- **Action:** Manual investigation
- **Notification:** Warning alert to Slack + Email

### Level 3: Team Lead

- **Duration:** 15-30 minutes
- **Action:** Team coordination
- **Notification:** Error alert + PagerDuty

### Level 4: Emergency Response

- **Duration:** 30+ minutes
- **Action:** All hands on deck
- **Notification:** Critical alert + Phone calls

---

## Appendix

### Useful Commands

```bash
# System status
curl http://localhost:8000/api/health

# Self-healing stats
curl http://localhost:8000/api/admin/self-healing/stats

# Recent repairs
curl http://localhost:8000/api/admin/self-healing/repairs?limit=10

# Active alerts
curl http://localhost:8000/api/admin/alerts/active

# Version health
python scripts/version_health_report.py --version v2 --detailed

# Trigger manual repair
python scripts/manual_repair.py --action restart_service --service analysis

# Check quality gates
python scripts/check_quality_gates.py --version v1

# Migrate version
python scripts/migrate_version.py --from v1 --to v2 --module auth

# Rollback
./scripts/rollback.sh --to v2.0.5 --verify
```

### Log Locations

- **Application:** `/var/log/ai-code-review/app.log`
- **Self-Healing:** `/var/log/ai-code-review/self-healing.log`
- **Audit:** `/var/log/ai-code-review/audit.log`
- **Kubernetes:** `kubectl logs -n platform-v2 <pod-name>`

### Metrics Endpoints

- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3001
- **Health API:** http://localhost:8000/healthz
- **Metrics API:** http://localhost:8000/metrics

---

**This runbook should be the first reference for any operational issues. Keep it updated with new scenarios and resolutions!**
