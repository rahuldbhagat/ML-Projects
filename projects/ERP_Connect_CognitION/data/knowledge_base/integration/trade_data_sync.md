---
title: "Trade Data Sync Integration Guide"
category: "INTEGRATION"
component: "MIDDLEWARE"
difficulty: "advanced"
last_updated: "2024-02-20"
topics: ['integration', 'middleware', 'trade-data-sync']
---

# Trade Data Sync Integration Guide

## Architecture Overview

The Trade Data Sync integration connects CTRM system with SAP using the following components:

- **Trade Export**: CTRM exports trade data
- **Validation**: Middleware validates data
- **Transformation**: Convert to SAP format
- **SAP Post**: Post to SAP FI

## Message Flow

The typical Trade Data Sync flow follows these steps:

1. CTRM generates trade file (CSV/JSON)
2. File uploaded to S3 bucket
3. S3 event triggers Lambda
4. Lambda validates trade data
5. Data transformed to SAP BAPI format
6. BAPI_ACC_DOCUMENT_POST called
7. Response logged and stored

## Error Scenarios

### Missing Required Fields

**Root Cause:** Trade data incomplete

**Resolution:**
1. Check CTRM export configuration
2. Verify required field list
3. Review validation rules
4. Update export template

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/trade-data-sync --follow
```

---

### G/L Account Not Found

**Root Cause:** Account doesn't exist in SAP

**Resolution:**
1. Check interface mapping for G/L account
2. Verify account exists in SAP (FS00)
3. Create account if missing
4. Update mapping configuration

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/trade-data-sync --follow
```

---

### Balance Not Zero

**Root Cause:** Debit and credit don't match

**Resolution:**
1. Verify calculation logic
2. Check for rounding errors
3. Review offsetting entries
4. Add balancing line item if needed

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/trade-data-sync --follow
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
