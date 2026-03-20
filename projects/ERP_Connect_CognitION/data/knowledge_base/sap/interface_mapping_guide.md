---
title: "SAP Interface Mapping Guide"
category: "SAP"
component: "SAP_BAPI"
difficulty: "intermediate"
last_updated: "2024-02-20"
topics: ['sap', 'bapi', 'interface-mapping']
---

# SAP Interface Mapping Guide

## Overview

This guide covers Interface Mapping in the CTRM-SAP integration context. It provides configuration steps, troubleshooting procedures, and best practices.

## Configuration

Interface mapping connects CTRM data elements to SAP master data. Configuration is managed in the Web UI under the Interface Mapping widget.

**Mapping Types:**
- G/L Account: Maps CTRM accounts to SAP G/L accounts
- Cost Center: Maps CTRM cost centers to SAP cost centers
- Profit Center: Maps trading books to SAP profit centers
- Business Unit: Maps legal entities to SAP company codes
- Material: Maps CTRM products to SAP material masters

**Configuration Location:** Web UI > Configuration > Interface Mapping

## Common Issues

### Adding New Cost Center Mapping

**Symptoms:** Cost center not found error in SAP posting

**Cause:** CTRM cost center ENERGY572 not mapped to SAP

**Resolution:**

1. Login to Web UI
2. Navigate to Interface Mapping widget
3. Select 'Cost Center' mapping type
4. Click 'Add New Mapping'
5. Enter CTRM Cost Center: ENERGY572
6. Enter SAP Cost Center: 4711ENERGY
7. Select Business Unit: AMEX001
8. Click 'Save'
9. Test mapping with sample trade

**SAP Transaction Codes:**
- `KS03` - Display Cost Center
- `KS02` - Change Cost Center

---

### G/L Account Mapping Missing

**Symptoms:** G/L account does not exist error

**Cause:** CTRM G/L account not mapped to SAP chart of accounts

**Resolution:**

1. Identify CTRM G/L account from error log
2. Determine appropriate SAP G/L account
3. Open Interface Mapping widget
4. Select 'G/L Account' type
5. Add mapping with correct company code
6. Verify SAP account exists (FS00)
7. Save and test mapping

**SAP Transaction Codes:**
- `FS00` - G/L Account Master
- `FSP0` - Change G/L Account

---

## Examples

### Query Interface Mapping

Retrieve SAP cost center for CTRM code

```python
from interface_mapping import InterfaceMapping

mapper = InterfaceMapping()
sap_costcenter = mapper.get_sap_value(
    mapping_type='COSTCENTER',
    ctrm_value='ENERGY572',
    business_unit='AMEX001'
)
print(f"SAP Cost Center: {sap_costcenter}")
```

**Expected Result:**
Returns SAP cost center 4711ENERGY if mapping exists, None otherwise

---

### Add New Mapping Programmatically

Create interface mapping via API

```python
mapper.add_mapping(
    mapping_type='PROFITCENTER',
    ctrm_value='BOOK_POWER_EAST',
    sap_value='PC_PWR_EAST',
    business_unit='AMEX001',
    valid_from='2024-01-01'
)
```

**Expected Result:**
Mapping created and available immediately for use

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
