---
title: "ECS_FARGATE Troubleshooting Guide"
category: "ARCHITECTURE"
component: "ECS_FARGATE"
difficulty: "intermediate"
last_updated: "2024-02-20"
topics: ['troubleshooting', 'infrastructure', 'ecs_fargate']
---

# ECS_FARGATE Troubleshooting Guide

## Overview

ECS_FARGATE is a critical infrastructure component in the CTRM-SAP integration architecture. This guide covers common issues and resolutions.

## Common Issues

### 1. Container Exit Code 137

**Symptoms:** Container killed due to memory

**Resolution Steps:**

1. Check memory allocation in task definition
2. Increase memory limit
3. Review application memory usage
4. Enable container insights for monitoring

**Verification:**
```bash
# Check service status
curl -f http://localhost:8080/health || echo "Service unhealthy"
```

---

### 2. Health Check Failures

**Symptoms:** ALB marks targets unhealthy

**Resolution Steps:**

1. Verify health endpoint returns 200
2. Check security groups allow ALB access
3. Increase health check timeout
4. Review application startup time

**Verification:**
```bash
# Check service status
curl -f http://localhost:8080/health || echo "Service unhealthy"
```

---

### 3. Cannot Pull Image

**Symptoms:** Task stuck in PENDING state

**Resolution Steps:**

1. Verify NAT Gateway or VPC endpoints exist
2. Check IAM task execution role has ECR permissions
3. Confirm image exists in ECR
4. Review VPC configuration

**Verification:**
```bash
# Check service status
curl -f http://localhost:8080/health || echo "Service unhealthy"
```

---

## Monitoring

**Key Metrics to Track:**
- CPU utilization (target: <75%)
- Memory usage (target: <85%)
- Response time (target: <500ms)
- Error rate (target: <1%)

**CloudWatch Alarms:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name {component}-high-error-rate \
  --metric-name Errors \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold
```

## Best Practices

- Always implement health checks for proactive monitoring
- Use appropriate resource limits (CPU/memory)
- Enable container insights for detailed metrics
- Implement graceful shutdown handling (SIGTERM)
- Use service discovery for internal communication

## Related Documentation

- See also: [Performance Monitoring](../performance/monitoring_guide.md)
- See also: [Optimization Guide](../performance/optimization_checklist.md)
