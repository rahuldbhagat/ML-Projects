"""
Template Library for Synthetic Knowledge Base Generation
Compact templates that generate realistic technical documentation
"""

def create_frontmatter(title, category, component, difficulty, topics):
    """Generate YAML frontmatter for markdown files"""
    return f"""---
title: "{title}"
category: "{category}"
component: "{component}"
difficulty: "{difficulty}"
last_updated: "2024-02-20"
topics: {topics}
---

"""

def create_architecture_doc(component, port, common_errors, solutions):
    """Template for architecture components"""
    content = create_frontmatter(
        f"{component} Troubleshooting Guide",
        "ARCHITECTURE",
        component,
        "intermediate",
        ["troubleshooting", "infrastructure", component.lower()]
    )
    
    content += f"""# {component} Troubleshooting Guide

## Overview

{component} is a critical infrastructure component in the CTRM-SAP integration architecture. This guide covers common issues and resolutions.

## Common Issues

"""
    
    for i, (error, cause, fix) in enumerate(common_errors, 1):
        content += f"""### {i}. {error}

**Symptoms:** {cause}

**Resolution Steps:**

{fix}

**Verification:**
```bash
# Check service status
curl -f http://localhost:{port}/health || echo "Service unhealthy"
```

---

"""
    
    content += """## Monitoring

**Key Metrics to Track:**
- CPU utilization (target: <75%)
- Memory usage (target: <85%)
- Response time (target: <500ms)
- Error rate (target: <1%)

**CloudWatch Alarms:**
```bash
aws cloudwatch put-metric-alarm \\
  --alarm-name {component}-high-error-rate \\
  --metric-name Errors \\
  --threshold 10 \\
  --comparison-operator GreaterThanThreshold
```

## Best Practices

"""
    
    for practice in solutions:
        content += f"- {practice}\n"
    
    content += "\n## Related Documentation\n\n"
    content += "- See also: [Performance Monitoring](../performance/monitoring_guide.md)\n"
    content += "- See also: [Optimization Guide](../performance/optimization_checklist.md)\n"
    
    return content

def create_http_error_doc(error_codes, error_type):
    """Template for HTTP error documentation"""
    content = create_frontmatter(
        f"{error_type} HTTP Errors Guide",
        "HTTP_ERRORS",
        "HTTP",
        "beginner",
        ["http", "errors", error_type]
    )
    
    content += f"""# {error_type} HTTP Errors Guide

## Overview

{error_type} errors indicate client-side ({error_type[0]}XX) issues in the CTRM-SAP integration. This guide provides troubleshooting steps for each error code.

"""
    
    for code, name, cause, solution in error_codes:
        content += f"""## HTTP {code} - {name}

**Meaning:** {cause}

**Common Causes in CTRM-SAP:**
1. Malformed request payload
2. Authentication/authorization issues
3. Resource not found or unavailable
4. Rate limiting or quota exceeded

**Diagnostic Steps:**

**Step 1: Check Request Format**
```python
# Verify JSON payload
import json
try:
    data = json.loads(request_body)
    print("Valid JSON")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {{e}}")
```

**Step 2: Review Error Response**
```bash
curl -v https://api.example.com/endpoint
# Check response headers and body for details
```

**Resolution:**

{solution}

**Prevention:**
- Implement request validation
- Use proper error handling
- Monitor error rates
- Document API requirements

---

"""
    
    content += """## Monitoring HTTP Errors

**CloudWatch Metrics:**
```bash
aws cloudwatch get-metric-statistics \\
  --namespace AWS/ApiGateway \\
  --metric-name 4XXError \\
  --dimensions Name=ApiName,Value=CTRM-SAP-API \\
  --start-time 2024-01-01T00:00:00Z \\
  --end-time 2024-01-01T01:00:00Z \\
  --period 300 \\
  --statistics Sum
```

## Best Practices

- Return clear, actionable error messages
- Include error codes and request IDs
- Log errors with sufficient context
- Implement retry logic for transient errors
- Use exponential backoff for rate limits

## Related Documentation

- See also: [API Gateway Configuration](../architecture/api_gateway_errors.md)
- See also: [Integration Patterns](../integration/async_processing.md)
"""
    
    return content

def create_integration_doc(topic, components, flow_steps, error_scenarios):
    """Template for integration documentation"""
    content = create_frontmatter(
        f"{topic} Integration Guide",
        "INTEGRATION",
        "MIDDLEWARE",
        "advanced",
        ["integration", "middleware", topic.lower().replace(' ', '-')]
    )
    
    content += f"""# {topic} Integration Guide

## Architecture Overview

The {topic} integration connects CTRM system with SAP using the following components:

"""
    
    for comp in components:
        content += f"- **{comp[0]}**: {comp[1]}\n"
    
    content += f"""
## Message Flow

The typical {topic} flow follows these steps:

"""
    
    for i, step in enumerate(flow_steps, 1):
        content += f"{i}. {step}\n"
    
    content += """
## Error Scenarios

"""
    
    for scenario, cause, resolution in error_scenarios:
        content += f"""### {scenario}

**Root Cause:** {cause}

**Resolution:**
{resolution}

**Monitoring:**
```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/{topic.lower().replace(' ', '-')} --follow
```

---

"""
    
    content += """## Configuration

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
"""
    
    return content

def create_sap_doc(topic, details, examples):
    """Template for SAP documentation"""
    content = create_frontmatter(
        f"SAP {topic} Guide",
        "SAP",
        "SAP_BAPI",
        "intermediate",
        ["sap", "bapi", topic.lower().replace(' ', '-')]
    )
    
    content += f"""# SAP {topic} Guide

## Overview

This guide covers {topic} in the CTRM-SAP integration context. It provides configuration steps, troubleshooting procedures, and best practices.

## Configuration

{details['config']}

## Common Issues

"""
    
    for issue in details['issues']:
        content += f"""### {issue['name']}

**Symptoms:** {issue['symptom']}

**Cause:** {issue['cause']}

**Resolution:**

{issue['resolution']}

**SAP Transaction Codes:**
"""
        for tcode in issue.get('tcodes', []):
            content += f"- `{tcode[0]}` - {tcode[1]}\n"
        
        content += "\n---\n\n"
    
    content += """## Examples

"""
    
    for example in examples:
        content += f"""### {example['title']}

{example['description']}

```python
{example['code']}
```

**Expected Result:**
{example['result']}

---

"""
    
    content += """## Best Practices

- Always validate data before posting
- Use test system for validation
- Implement proper error handling
- Monitor BAPI return messages
- Document all custom mappings
- Regular authorization reviews

## Related Documentation

- See also: [BAPI Best Practices](bapi_best_practices.md)
- See also: [Interface Mapping](interface_mapping_guide.md)
"""
    
    return content

def create_performance_doc(topic, metrics, optimization_steps):
    """Template for performance documentation"""
    content = create_frontmatter(
        f"{topic} Guide",
        "PERFORMANCE",
        "MONITORING",
        "advanced",
        ["performance", "monitoring", "optimization"]
    )
    
    content += f"""# {topic} Guide

## Overview

This guide covers {topic.lower()} for the CTRM-SAP integration platform. Follow these guidelines to maintain optimal performance.

## Key Metrics

"""
    
    for metric in metrics:
        content += f"""### {metric['name']}

**Target:** {metric['target']}
**Warning Threshold:** {metric['warning']}
**Critical Threshold:** {metric['critical']}

**Monitoring:**
```bash
{metric['command']}
```

---

"""
    
    content += """## Optimization Steps

"""
    
    for i, step in enumerate(optimization_steps, 1):
        content += f"""### Step {i}: {step['title']}

**Problem:** {step['problem']}

**Solution:**
{step['solution']}

**Implementation:**
```
{step['code']}
```

**Expected Improvement:** {step['improvement']}

---

"""
    
    content += """## Continuous Monitoring

**Set Up Alarms:**
```bash
aws cloudwatch put-metric-alarm \\
  --alarm-name performance-degradation \\
  --metric-name ResponseTime \\
  --threshold 1000 \\
  --comparison-operator GreaterThanThreshold \\
  --evaluation-periods 2 \\
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
"""
    
    return content

def create_bapi_errors_json():
    """Generate BAPI errors catalog in JSON format"""
    errors = [
        {
            "error_code": "PERIOD_NOT_OPEN",
            "message": "Posting period 013 2024 is not open",
            "category": "POSTING",
            "severity": "ERROR",
            "causes": [
                "Fiscal period not opened in SAP",
                "Year-end closing in progress",
                "Authorization to open period missing"
            ],
            "resolution_steps": [
                "Check period status using transaction OB52",
                "Open posting period for company code",
                "Verify user has authorization to post",
                "Confirm fiscal year variant configuration"
            ],
            "prevention": "Implement automated period opening process",
            "related_tcodes": ["OB52", "FAGLGVTR"],
            "sap_note": "SAP Note 0000101"
        },
        {
            "error_code": "GLACCOUNT_NOT_FOUND",
            "message": "G/L account 400000 does not exist in company code 1000",
            "category": "MASTER_DATA",
            "severity": "ERROR",
            "causes": [
                "G/L account not created in company code",
                "Incorrect account number in mapping",
                "Chart of accounts mismatch"
            ],
            "resolution_steps": [
                "Verify G/L account exists using FS00",
                "Create G/L account in company code if missing",
                "Update interface mapping with correct account",
                "Check chart of accounts assignment"
            ],
            "prevention": "Validate G/L accounts before interface mapping",
            "related_tcodes": ["FS00", "FSP0"],
            "sap_note": "SAP Note 0000102"
        },
        {
            "error_code": "COSTCENTER_NOT_VALID",
            "message": "Cost center ENERGY572 is not valid on 2024-01-15",
            "category": "INTERFACE_MAPPING",
            "severity": "ERROR",
            "causes": [
                "Cost center not active for posting date",
                "Cost center expired or not created",
                "Validity dates not configured correctly"
            ],
            "resolution_steps": [
                "Check cost center validity in KS03",
                "Extend validity period if needed using KS02",
                "Verify cost center is assigned to correct company code",
                "Update interface mapping with active cost center"
            ],
            "prevention": "Regular review of cost center validity periods",
            "related_tcodes": ["KS03", "KS02", "KSH1"],
            "sap_note": "SAP Note 0000103"
        },
        {
            "error_code": "AUTHORIZATION_MISSING",
            "message": "No authorization for transaction FB01 in company code 1000",
            "category": "AUTHORIZATION",
            "severity": "ERROR",
            "causes": [
                "User lacks posting authorization",
                "Company code not in authorization profile",
                "Authorization object F_BKPF_BUK not maintained"
            ],
            "resolution_steps": [
                "Check user authorizations using SU53",
                "Add company code to user's authorization profile",
                "Assign role with F_BKPF_BUK authorization",
                "Test authorization using SU53 after changes"
            ],
            "prevention": "Regular authorization audits and reviews",
            "related_tcodes": ["SU53", "PFCG", "SU01"],
            "sap_note": "SAP Note 0000104"
        },
        {
            "error_code": "DOCUMENT_LOCKED",
            "message": "Document 1400000000 2024 is locked by user SMITHJ",
            "category": "CONCURRENCY",
            "severity": "WARNING",
            "causes": [
                "Another user editing the document",
                "Previous session not closed properly",
                "System lock not released"
            ],
            "resolution_steps": [
                "Wait for other user to finish editing",
                "Check lock entries using SM12",
                "Contact user to release lock",
                "Delete lock entry if user confirms (SM12)"
            ],
            "prevention": "Implement proper session management",
            "related_tcodes": ["SM12", "FB03"],
            "sap_note": "SAP Note 0000105"
        },
        {
            "error_code": "BALANCE_ZERO_ERROR",
            "message": "Document is not balanced: debit 1000.00 credit 900.00",
            "category": "POSTING",
            "severity": "ERROR",
            "causes": [
                "Incorrect line item amounts",
                "Rounding errors in calculation",
                "Missing offsetting entry"
            ],
            "resolution_steps": [
                "Verify debit and credit amounts match",
                "Check for rounding errors in source system",
                "Add missing line items to balance document",
                "Review transformation logic"
            ],
            "prevention": "Implement balance validation before posting",
            "related_tcodes": ["FB01", "FB03"],
            "sap_note": "SAP Note 0000106"
        },
        {
            "error_code": "TAX_CODE_INVALID",
            "message": "Tax code V0 is not defined for country US",
            "category": "CONFIGURATION",
            "severity": "ERROR",
            "causes": [
                "Tax code not configured for country",
                "Wrong tax code in mapping",
                "Tax configuration incomplete"
            ],
            "resolution_steps": [
                "Check tax code configuration using FTXP",
                "Configure tax code for country if missing",
                "Update interface mapping with correct tax code",
                "Verify tax calculation procedure"
            ],
            "prevention": "Validate tax codes during interface setup",
            "related_tcodes": ["FTXP", "OB40"],
            "sap_note": "SAP Note 0000107"
        },
        {
            "error_code": "VENDOR_BLOCKED",
            "message": "Vendor 100234 is blocked for posting",
            "category": "MASTER_DATA",
            "severity": "ERROR",
            "causes": [
                "Vendor marked for deletion",
                "Payment block set",
                "Posting block activated"
            ],
            "resolution_steps": [
                "Check vendor status using XK03",
                "Remove posting block using XK02",
                "Verify reason for block before removing",
                "Contact procurement team if needed"
            ],
            "prevention": "Regular vendor master data reviews",
            "related_tcodes": ["XK03", "XK02", "FK03"],
            "sap_note": "SAP Note 0000108"
        },
        {
            "error_code": "PROFIT_CENTER_MISSING",
            "message": "Enter profit center for G/L account 500000",
            "category": "INTERFACE_MAPPING",
            "severity": "ERROR",
            "causes": [
                "Profit center required but not provided",
                "G/L account requires profit center",
                "Interface mapping incomplete"
            ],
            "resolution_steps": [
                "Check G/L account master for profit center requirement (FS00)",
                "Add profit center to interface mapping",
                "Verify profit center is active (KE53)",
                "Update transformation logic to include profit center"
            ],
            "prevention": "Validate required fields in interface mapping",
            "related_tcodes": ["FS00", "KE53", "KE52"],
            "sap_note": "SAP Note 0000109"
        },
        {
            "error_code": "NUMBER_RANGE_EXHAUSTED",
            "message": "Number range 01 for document type SA in 2024 is exhausted",
            "category": "CONFIGURATION",
            "severity": "CRITICAL",
            "causes": [
                "Number range limit reached",
                "Number range not extended",
                "High volume of postings"
            ],
            "resolution_steps": [
                "Check number range using FBN1",
                "Extend number range using FBN1",
                "Create new number range interval if needed",
                "Plan for larger intervals in future"
            ],
            "prevention": "Monitor number range usage regularly",
            "related_tcodes": ["FBN1", "SNUM"],
            "sap_note": "SAP Note 0000110"
        },
        {
            "error_code": "CURRENCY_MISMATCH",
            "message": "Currency USD does not match company code currency EUR",
            "category": "POSTING",
            "severity": "ERROR",
            "causes": [
                "Transaction currency different from local currency",
                "Exchange rate not maintained",
                "Multi-currency posting configuration issue"
            ],
            "resolution_steps": [
                "Verify exchange rate exists for date (OB08)",
                "Maintain exchange rate if missing",
                "Use document currency field for foreign currency",
                "Check company code currency configuration"
            ],
            "prevention": "Automated exchange rate updates",
            "related_tcodes": ["OB08", "OB22", "S_BCE_68000174"],
            "sap_note": "SAP Note 0000111"
        },
        {
            "error_code": "PAYMENT_TERMS_INVALID",
            "message": "Payment terms Z030 is not defined",
            "category": "MASTER_DATA",
            "severity": "ERROR",
            "causes": [
                "Payment terms not configured",
                "Incorrect payment terms code",
                "Payment terms deleted"
            ],
            "resolution_steps": [
                "Check payment terms configuration using OBB8",
                "Create payment terms if missing",
                "Update interface mapping with valid payment terms",
                "Verify payment terms in vendor master"
            ],
            "prevention": "Standardize payment terms across systems",
            "related_tcodes": ["OBB8", "XK03"],
            "sap_note": "SAP Note 0000112"
        },
        {
            "error_code": "BUSINESS_AREA_REQUIRED",
            "message": "Enter business area for G/L account 600000",
            "category": "INTERFACE_MAPPING",
            "severity": "ERROR",
            "causes": [
                "Business area mandatory for account",
                "Interface mapping missing business area",
                "Configuration requires business area"
            ],
            "resolution_steps": [
                "Check G/L account settings (FS00)",
                "Add business area to interface mapping",
                "Verify business area exists and is active",
                "Update transformation to include business area"
            ],
            "prevention": "Complete field requirement analysis",
            "related_tcodes": ["FS00", "SPRO"],
            "sap_note": "SAP Note 0000113"
        },
        {
            "error_code": "COMMITMENT_ITEM_MISSING",
            "message": "Enter commitment item for cost center ENERGY572",
            "category": "INTERFACE_MAPPING",
            "severity": "ERROR",
            "causes": [
                "Commitment item required for cost center",
                "Funds management active",
                "Interface mapping incomplete"
            ],
            "resolution_steps": [
                "Check cost center for commitment item requirement (KS03)",
                "Add commitment item to interface mapping",
                "Verify funds management configuration",
                "Consult with finance team on correct commitment item"
            ],
            "prevention": "Document funds management requirements",
            "related_tcodes": ["KS03", "FM5S"],
            "sap_note": "SAP Note 0000114"
        },
        {
            "error_code": "ASSIGNMENT_FIELD_TOO_LONG",
            "message": "Assignment field exceeds maximum length of 18 characters",
            "category": "VALIDATION",
            "severity": "ERROR",
            "causes": [
                "Source data too long",
                "Field mapping incorrect",
                "No truncation logic implemented"
            ],
            "resolution_steps": [
                "Review source data format",
                "Implement truncation in transformation logic",
                "Use abbreviations or codes instead of full text",
                "Document field length limitations"
            ],
            "prevention": "Validate field lengths in transformation layer",
            "related_tcodes": ["FB01"],
            "sap_note": "SAP Note 0000115"
        }
    ]
    
    return {"bapi_errors": errors, "last_updated": "2024-02-20", "version": "1.0"}
