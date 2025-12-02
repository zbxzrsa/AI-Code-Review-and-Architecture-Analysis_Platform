# Operations Runbook

## Daily Operations

### Health Checks

**Morning Health Check (Daily)**

```bash
# Check all services are running
kubectl get pods -n platform-v2-stable
kubectl get pods -n platform-v1-exp
kubectl get pods -n platform-v3-quarantine

# Check SLO compliance
curl http://localhost:8001/api/v1/health/slo

# Check database connectivity
kubectl -n platform-v2-stable exec -it <pod-name> -- \
  psql -h postgres -U platform_user -d production -c "SELECT 1"
```

### Monitoring Dashboard

1. Open Grafana: http://localhost:3000
2. Login: admin/admin
3. Check dashboards:
   - V2 Production SLO Compliance
   - V1 Experiment Progress
   - V3 Quarantine Statistics

### Alert Response

**SLO Violation Alert**

1. Check current metrics: `/metrics/slo-compliance`
2. Identify affected endpoint
3. Review recent deployments
4. If needed, rollback: `kubectl rollout undo deployment/platform-v2-api -n platform-v2-stable`

**High Error Rate Alert**

1. Check error logs: `kubectl logs -n platform-v2-stable <pod-name>`
2. Verify AI provider availability
3. Check database connectivity
4. Restart pod if needed: `kubectl delete pod -n platform-v2-stable <pod-name>`

## Experiment Management

### Creating a New Experiment

1. **Design the experiment**

   - Define hypothesis
   - Select AI model(s) to test
   - Create prompt template
   - Define success criteria

2. **Create via API**

```bash
curl -X POST http://localhost:8002/api/v1/experiments/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test new prompt v3",
    "description": "Improved prompt with examples",
    "primary_model": "gpt-4",
    "secondary_model": "claude-3-opus-20240229",
    "prompt_template": "Review this code...",
    "routing_strategy": "primary",
    "tags": ["prompt-v3", "gpt-4"]
  }'
```

3. **Run the experiment**

```bash
curl -X POST http://localhost:8002/api/v1/experiments/run/{experiment_id} \
  -H "Content-Type: application/json" \
  -d '{
    "code_samples": ["def test(): pass"],
    "language": "python"
  }'
```

4. **Monitor progress**

```bash
curl http://localhost:8002/api/v1/experiments/{experiment_id}
```

### Promoting to V2

**Automatic Promotion (if metrics pass)**

```bash
curl -X POST http://localhost:8002/api/v1/evaluation/promote/{experiment_id}
```

**Manual Promotion (force)**

```bash
curl -X POST http://localhost:8002/api/v1/evaluation/promote/{experiment_id}?force=true
```

**Verification after promotion**

1. Check V2 deployment: `kubectl get pods -n platform-v2-stable`
2. Run smoke tests: `curl http://localhost:8001/api/v1/health/status`
3. Monitor metrics for 1 hour
4. Verify SLO compliance: `curl http://localhost:8001/api/v1/health/slo`

### Quarantining Failed Experiments

```bash
curl -X POST http://localhost:8002/api/v1/evaluation/quarantine/{experiment_id} \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "Accuracy below threshold (0.92 < 0.95)",
    "impact_analysis": {
      "affected_models": ["gpt-4"],
      "recommendation": "Adjust prompt template"
    }
  }'
```

**Verify quarantine**

```bash
curl http://localhost:8003/api/v1/quarantine/records
```

## Scaling Operations

### Scale V2 Production

**Manual scaling**

```bash
kubectl scale deployment platform-v2-api -n platform-v2-stable --replicas=5
```

**Check HPA status**

```bash
kubectl get hpa -n platform-v2-stable
kubectl describe hpa platform-v2-hpa -n platform-v2-stable
```

**Adjust HPA limits**

```bash
kubectl patch hpa platform-v2-hpa -n platform-v2-stable -p '{"spec":{"maxReplicas":15}}'
```

### Scale V1 Experimentation

```bash
kubectl scale deployment platform-v1-api -n platform-v1-exp --replicas=3
```

## Database Operations

### Backup

**Full backup**

```bash
kubectl exec -it postgres-pod -- \
  pg_dump -U postgres platform > backup_$(date +%Y%m%d_%H%M%S).sql
```

**Schema-specific backup**

```bash
# V2 Production
kubectl exec -it postgres-pod -- \
  pg_dump -U postgres -n production platform > backup_v2_$(date +%Y%m%d_%H%M%S).sql

# V1 Experimentation
kubectl exec -it postgres-pod -- \
  pg_dump -U postgres -n experiments_v1 platform > backup_v1_$(date +%Y%m%d_%H%M%S).sql

# V3 Quarantine
kubectl exec -it postgres-pod -- \
  pg_dump -U postgres -n quarantine platform > backup_v3_$(date +%Y%m%d_%H%M%S).sql
```

### Restore

```bash
kubectl exec -it postgres-pod -- \
  psql -U postgres platform < backup.sql
```

### Database Maintenance

**Vacuum (cleanup)**

```bash
kubectl exec -it postgres-pod -- \
  psql -U postgres -d platform -c "VACUUM ANALYZE;"
```

**Check table sizes**

```bash
kubectl exec -it postgres-pod -- \
  psql -U postgres -d platform -c "
    SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
    FROM pg_tables
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

## Deployment Operations

### Rolling Update

```bash
# Update image
kubectl set image deployment/platform-v2-api \
  -n platform-v2-stable \
  api=your-registry/platform-v2:v1.1.0

# Monitor rollout
kubectl rollout status deployment/platform-v2-api -n platform-v2-stable

# Check history
kubectl rollout history deployment/platform-v2-api -n platform-v2-stable
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/platform-v2-api -n platform-v2-stable

# Rollback to specific revision
kubectl rollout undo deployment/platform-v2-api -n platform-v2-stable --to-revision=2
```

### Blue-Green Deployment

1. Deploy new version to separate namespace
2. Run smoke tests
3. Switch traffic via ingress
4. Keep old version running for quick rollback

## Troubleshooting

### Pod Crashes

```bash
# Check logs
kubectl logs -n platform-v2-stable <pod-name>

# Check previous logs (if crashed)
kubectl logs -n platform-v2-stable <pod-name> --previous

# Describe pod for events
kubectl describe pod -n platform-v2-stable <pod-name>
```

### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n platform-v2-stable

# Check resource limits
kubectl get pods -n platform-v2-stable -o json | \
  jq '.items[] | {name: .metadata.name, resources: .spec.containers[].resources}'
```

### Database Connection Issues

```bash
# Check connection pool status
kubectl exec -it postgres-pod -- \
  psql -U postgres -d platform -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
kubectl exec -it postgres-pod -- \
  psql -U postgres -d platform -c "
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE state = 'idle' AND query_start < now() - interval '1 hour';"
```

### Network Policy Issues

```bash
# Test connectivity between pods
kubectl -n platform-v2-stable exec -it <pod-name> -- \
  curl http://platform-v2-api:8000/health/live

# Check network policies
kubectl get networkpolicies -n platform-v2-stable

# Describe policy
kubectl describe networkpolicy v2-isolation -n platform-v2-stable
```

### API Latency Issues

1. Check database query performance
2. Check AI provider response times
3. Check network latency between services
4. Review Prometheus metrics for bottlenecks

```bash
# Check slow queries
kubectl exec -it postgres-pod -- \
  psql -U postgres -d production -c "
    SELECT query, calls, mean_time
    FROM pg_stat_statements
    ORDER BY mean_time DESC LIMIT 10;"
```

## Incident Response

### SLO Breach

1. **Immediate actions**

   - Page on-call engineer
   - Gather metrics and logs
   - Identify root cause

2. **Investigation**

   - Check recent deployments
   - Review error logs
   - Check external dependencies (AI providers)
   - Check database performance

3. **Mitigation**

   - Scale up if needed
   - Rollback if recent deployment
   - Switch to secondary AI provider if primary is down
   - Restart affected pods

4. **Post-incident**
   - Document root cause
   - Create action items
   - Update runbooks
   - Schedule postmortem

### Data Corruption

1. **Immediate actions**

   - Stop all writes to affected schema
   - Isolate affected pods
   - Notify team

2. **Investigation**

   - Check audit logs
   - Identify scope of corruption
   - Determine recovery point

3. **Recovery**
   - Restore from backup
   - Verify data integrity
   - Resume operations

### Security Incident

1. **Immediate actions**

   - Isolate affected systems
   - Revoke compromised credentials
   - Enable enhanced logging

2. **Investigation**

   - Review access logs
   - Check for unauthorized changes
   - Identify attack vector

3. **Recovery**
   - Patch vulnerabilities
   - Rotate credentials
   - Restore from clean backup if needed

## Maintenance Windows

### Planned Maintenance

1. **Schedule**

   - Announce 2 weeks in advance
   - Choose low-traffic window
   - Plan for 2-4 hours

2. **Pre-maintenance**

   - Backup all databases
   - Document current state
   - Prepare rollback plan

3. **During maintenance**

   - Monitor all systems
   - Have team on standby
   - Document all changes

4. **Post-maintenance**
   - Verify all systems operational
   - Run smoke tests
   - Monitor metrics for 24 hours

## Cost Optimization

### Monitor Costs

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n platform-v2-stable
kubectl top pods -n platform-v1-exp
kubectl top pods -n platform-v3-quarantine
```

### Optimize Resources

1. **Right-size requests/limits**

   - Monitor actual usage
   - Adjust based on data
   - Avoid over-provisioning

2. **Use spot instances for V1**

   - Experimentation can tolerate interruptions
   - Significant cost savings

3. **Consolidate workloads**
   - V3 can run on minimal resources
   - Consider shared infrastructure for V1

## Security Operations

### Access Control

```bash
# List RBAC roles
kubectl get roles -n platform-v2-stable
kubectl get rolebindings -n platform-v2-stable

# Audit access
kubectl get events -n platform-v2-stable
```

### Secret Rotation

```bash
# Rotate API keys
kubectl create secret generic platform-secrets \
  -n platform-v2-stable \
  --from-literal=primary_ai_api_key=new_key \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secrets
kubectl rollout restart deployment/platform-v2-api -n platform-v2-stable
```

### Network Security

```bash
# Verify network policies
kubectl get networkpolicies -n platform-v2-stable
kubectl get networkpolicies -n platform-v1-exp
kubectl get networkpolicies -n platform-v3-quarantine

# Test isolation
kubectl -n platform-v2-stable exec -it <v2-pod> -- \
  curl http://platform-v1-api:8000 # Should fail
```

## Compliance and Audit

### Audit Logging

```bash
# Check audit logs
kubectl exec -it postgres-pod -- \
  psql -U postgres -d platform -c "
    SELECT * FROM audit.event_log
    ORDER BY timestamp DESC LIMIT 100;"
```

### Compliance Reports

```bash
# SLO compliance report
curl http://localhost:8001/api/v1/metrics/slo-compliance

# Quarantine statistics
curl http://localhost:8003/api/v1/quarantine/statistics

# Model performance
curl http://localhost:8001/api/v1/metrics/models
```
