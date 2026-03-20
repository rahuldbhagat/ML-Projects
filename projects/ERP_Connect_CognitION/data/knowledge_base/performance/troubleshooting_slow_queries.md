---
title: "Troubleshooting Slow Queries Guide"
category: "PERFORMANCE"
component: "MONITORING"
difficulty: "advanced"
last_updated: "2024-02-20"
topics: ['performance', 'monitoring', 'optimization']
---

# Troubleshooting Slow Queries Guide

## Overview

This guide covers troubleshooting slow queries for the CTRM-SAP integration platform. Follow these guidelines to maintain optimal performance.

## Key Metrics

### Query Execution Time

**Target:** < 100ms
**Warning Threshold:** > 500ms
**Critical Threshold:** > 2000ms

**Monitoring:**
```bash
SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC;
```

---

## Optimization Steps

### Step 1: Analyze Query Plans

**Problem:** Don't know why query is slow

**Solution:**
Use EXPLAIN ANALYZE to understand execution

**Implementation:**
```
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 123;
```

**Expected Improvement:** Identify missing indexes and inefficient joins

---

### Step 2: Optimize JOIN Operations

**Problem:** Query with multiple JOINs is slow

**Solution:**
Add indexes on join columns, reduce joined tables

**Implementation:**
```
CREATE INDEX idx_order_customer ON orders(customer_id);
CREATE INDEX idx_customer_id ON customers(id);
```

**Expected Improvement:** 5-10x improvement on complex queries

---

## Continuous Monitoring

**Set Up Alarms:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name performance-degradation \
  --metric-name ResponseTime \
  --threshold 1000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --period 300
```

**Dashboard Widgets:**
- Response time (P50, P95, P99)
- Error rate by endpoint
- Request throughput
- Resource utilization

## Best Practices

- Monitor continuously, not reactively
- Set appropriate alert thresholds
- Regular performance testing
- Capacity planning based on trends
- Document performance baselines

## Related Documentation

- See also: [Optimization Checklist](optimization_checklist.md)
- See also: [Monitoring Guide](monitoring_guide.md)
