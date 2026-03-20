---
title: "SAP Master Data Setup Guide"
category: "SAP"
component: "SAP_BAPI"
difficulty: "intermediate"
last_updated: "2024-02-20"
topics: ['sap', 'bapi', 'master-data-setup']
---

# SAP Master Data Setup Guide

## Overview

This guide covers Master Data Setup in the CTRM-SAP integration context. It provides configuration steps, troubleshooting procedures, and best practices.

## Configuration

Master data must be configured in SAP before integration can post transactions. Critical master data includes G/L accounts, cost centers, profit centers, vendors, and materials.

**Prerequisites:**
- Chart of accounts assigned to company code
- Fiscal year variant configured
- Posting periods open
- Number ranges maintained

## Common Issues

### Creating G/L Account

**Symptoms:** G/L account not found during posting

**Cause:** Account not created in SAP for company code

**Resolution:**

1. Use transaction FS00
2. Enter G/L account number (e.g., 400000)
3. Enter company code (e.g., 1000)
4. Click 'Create'
5. Enter account description
6. Select account group
7. Set control data (currency, tax category)
8. Save

**SAP Transaction Codes:**
- `FS00` - G/L Account Master
- `FSP0` - Display Chart of Accounts

---

### Cost Center Validity

**Symptoms:** Cost center not valid on posting date

**Cause:** Validity period expired or not set

**Resolution:**

1. Use transaction KS02
2. Enter cost center code
3. Check validity dates
4. Extend 'Valid to' date if needed
5. Verify controlling area assignment
6. Save changes

**SAP Transaction Codes:**
- `KS02` - Change Cost Center
- `KS03` - Display Cost Center

---

## Examples

### Validate Master Data

Check if master data exists before posting

```python
def validate_master_data(glaccount, costcenter, company_code):
    # Check G/L account
    result = sap.call('BAPI_GL_ACC_GETDETAIL', {
        'GL_ACCOUNT': glaccount,
        'COMPANYCODE': company_code
    })
    if not result['success']:
        raise ValueError(f"G/L Account {glaccount} not found")
    
    # Check cost center
    result = sap.call('BAPI_COSTCENTER_GETDETAIL', {
        'COSTCENTER': costcenter
    })
    if not result['success']:
        raise ValueError(f"Cost Center {costcenter} not found")
    
    return True
```

**Expected Result:**
Returns True if all master data valid, raises exception otherwise

---

## Best Practices

- Always validate data before posting
- Use test system for validation
- Implement proper error handling
- Monitor BAPI return messages
- Document all custom mappings
- Regular authorization reviews

## Related Documentation

- See also: [BAPI Best Practices](bapi_best_practices.md)
- See also: [Interface Mapping](interface_mapping_guide.md)
