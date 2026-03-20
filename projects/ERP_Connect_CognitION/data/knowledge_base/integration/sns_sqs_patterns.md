---
title: "SNS SQS Patterns Integration Guide"
category: "INTEGRATION"
component: "MIDDLEWARE"
difficulty: "advanced"
last_updated: "2024-02-20"
topics: ['integration', 'middleware', 'sns-sqs-patterns']
---

# SNS SQS Patterns Integration Guide

## Architecture Overview

The SNS SQS Patterns integration connects CTRM system with SAP using the following components:

- **SNS Topic**: Pub/sub message distribution
- **SQS Queue**: Reliable message queuing
- **Lambda Consumer**: Event-driven processing
- **DLQ**: Failed message handling

## Message Flow

The typical SNS SQS Patterns flow follows these steps:

1. Publisher sends message to SNS topic
2. SNS fans out to subscribed SQS queues
3. Messages stored durably in SQS
4. Lambda polls queue for messages
5. Message processed and deleted
6. Failed messages retry automatically
7. Persistent failures moved to DLQ

## Error Scenarios

### Message Visibility Timeout

**Root Cause:** Message reprocessed too soon

**Resolution:**
1. Increase visibility timeout
2. Match to processing time
3. Extend timeout if needed during processing
4. Monitor reprocessing rate

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/sns-sqs-patterns --follow
```

---

### Poison Pill Messages

**Root Cause:** Message always fails

**Resolution:**
1. Check DLQ for patterns
2. Fix data quality issues
3. Update validation logic
4. Manually process or discard

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/sns-sqs-patterns --follow
```

---

### Queue Depth Growing

**Root Cause:** Processing too slow

**Resolution:**
1. Monitor queue metrics
2. Increase Lambda concurrency
3. Optimize processing logic
4. Add more consumers

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/sns-sqs-patterns --follow
```

---

## Configuration

**Environment Variables:**
```bash
CTRM_ENDPOINT=https://ctrm.example.com/api
SAP_ENDPOINT=https://sap.example.com/api
RETRY_ATTEMPTS=3
TIMEOUT_SECONDS=30
```

**Message Format:**
```json
{
  "message_id": "MSG-12345",
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {
    "trade_id": "TRD-67890",
    "data": {}
  }
}
```

## Best Practices

- Implement idempotency for all operations
- Use message deduplication
- Enable dead letter queues
- Monitor queue depths and age
- Implement circuit breakers
- Log all integration events

## Related Documentation

- See also: [Message Transformation](message_transformation.md)
- See also: [Error Handling](../architecture/api_gateway_errors.md)
