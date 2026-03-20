---
title: "ALB Troubleshooting Guide"
category: "ARCHITECTURE"
component: "ALB"
difficulty: "intermediate"
last_updated: "2024-02-20"
topics: ['troubleshooting', 'infrastructure', 'alb']
---

# ALB Troubleshooting Guide

## Overview

ALB is a critical infrastructure component in the CTRM-SAP integration architecture. This guide covers common issues and resolutions.

## Common Issues

### 1. 502 Bad Gateway

**Symptoms:** ALB received invalid response from target

**Resolution Steps:**

1. Check target health status
2. Review ALB access logs for pattern
3. Verify application returns valid HTTP response
4. Adjust target timeouts if needed

**Verification:**
```bash
# Check service status
curl -f http://localhost:80/health || echo "Service unhealthy"
```

---

### 2. 504 Gateway Timeout

**Symptoms:** Target didn't respond in time

**Resolution Steps:**

1. Increase ALB timeout settings
2. Optimize application performance
3. Implement async processing for long tasks
4. Add caching layer

**Verification:**
```bash
# Check service status
curl -f http://localhost:80/health || echo "Service unhealthy"
```

---

### 3. Target Health Check Failures

**Symptoms:** Targets marked unhealthy

**Resolution Steps:**

1. Verify health check path and port
2. Ensure health endpoint returns 200
3. Check timeout and interval settings
4. Review security group rules

**Verification:**
```bash
# Check service status
curl -f http://localhost:80/health || echo "Service unhealthy"
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

- Enable access logs for troubleshooting
- Configure appropriate health checks
- Use SSL/TLS termination at ALB
- Implement connection draining for graceful shutdowns
- Monitor RequestCount and TargetResponseTime

## Related Documentation

- See also: [Performance Monitoring](../performance/monitoring_guide.md)
- See also: [Optimization Guide](../performance/optimization_checklist.md)
