# Tamper-Proof Audit Logging

## Overview

Comprehensive audit logging system with cryptographic signatures, chain verification, and compliance support.

---

## Architecture

```
Application Event
    ↓
Audit Logger
    ├─ Construct log entry
    ├─ Compute SHA256 hash
    ├─ Sign with RSA-PSS
    ├─ Chain with previous entry
    └─ Store in database
    ↓
PostgreSQL (Immutable)
    ├─ Signature verification
    ├─ Chain validation
    └─ Compliance export
```

---

## Features

### Cryptographic Signatures

- **Algorithm**: RSA-PSS with SHA256
- **Key Size**: 4096-bit
- **Padding**: PSS with MGF1
- **Salt Length**: Maximum

### Chain Verification

- **Previous Hash**: Links to previous entry
- **Immutability**: Detects tampering
- **Integrity**: Validates entire chain

### Audit Trail

- **Entity Tracking**: What was changed
- **Action Tracking**: How it was changed
- **Actor Tracking**: Who made the change
- **Timestamp**: When it happened
- **Status**: Success/failure

---

## Setup

### Generate RSA Key Pair

```bash
# Generate private key (4096-bit)
openssl genrsa -out audit_private_key.pem 4096

# Extract public key
openssl rsa -in audit_private_key.pem -pubout -out audit_public_key.pem

# Encrypt private key with password
openssl genrsa -aes256 -out audit_private_key_encrypted.pem 4096
```

### Store Securely

**Production Options**:

- AWS KMS (Key Management Service)
- HashiCorp Vault
- Hardware Security Module (HSM)
- Azure Key Vault

**Example with Vault**:

```python
import hvac

client = hvac.Client(url='http://vault:8200')
private_key_pem = client.secrets.kv.read_secret_version(
    path='audit/private_key'
)['data']['data']['key']
```

### Initialize Logger

```python
from backend.shared.security.audit_logger import TamperProofAuditLogger

# Initialize with key file
audit_logger = TamperProofAuditLogger(
    db_client=db,
    private_key_path="/secure/audit_private_key.pem",
    private_key_password=b"your-password"
)

# Or with key from Vault
audit_logger = TamperProofAuditLogger(
    db_client=db,
    private_key_path=private_key_pem
)
```

---

## Usage

### Logging Events

```python
from backend.shared.security.audit_logger import AuditEntity, AuditAction

# Log version promotion
await audit_logger.log_event(
    entity=AuditEntity.VERSION.value,
    action=AuditAction.PROMOTE.value,
    actor_id=admin["id"],
    resource_id="v1-exp-123",
    payload={
        "from_tag": "v1",
        "to_tag": "v2",
        "metrics": {
            "accuracy": 0.87,
            "error_rate": 0.03,
            "cost_increase": 0.15
        }
    },
    status="success"
)

# Log failed access attempt
await audit_logger.log_event(
    entity=AuditEntity.USER.value,
    action=AuditAction.PERMISSION_DENIED.value,
    actor_id=user["id"],
    resource_id="admin_settings",
    payload={
        "reason": "insufficient_permissions",
        "requested_action": "delete_experiment"
    },
    status="failure"
)

# Log quota exceeded
await audit_logger.log_event(
    entity=AuditEntity.ANALYSIS.value,
    action=AuditAction.QUOTA_EXCEEDED.value,
    actor_id=user["id"],
    payload={
        "daily_requests": 1000,
        "daily_limit": 1000,
        "daily_cost": 100,
        "daily_cost_limit": 100
    },
    status="failure"
)
```

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from backend.shared.security.auth import CurrentUser

app = FastAPI()

@app.post("/versions/{version_id}/promote")
async def promote_version(
    version_id: str,
    admin = Depends(RoleBasedAccess.require_admin)
):
    try:
        # Perform promotion
        result = await version_control_service.promote(version_id, admin["id"])

        # Log success
        await audit_logger.log_event(
            entity="version",
            action="promote",
            actor_id=admin["id"],
            resource_id=version_id,
            payload=result,
            status="success"
        )

        return result
    except Exception as e:
        # Log failure
        await audit_logger.log_event(
            entity="version",
            action="promote",
            actor_id=admin["id"],
            resource_id=version_id,
            payload={"error": str(e)},
            status="failure"
        )
        raise

@app.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    admin = Depends(RoleBasedAccess.require_admin)
):
    try:
        # Perform deletion
        await experiment_service.delete(experiment_id)

        # Log deletion
        await audit_logger.log_event(
            entity="experiment",
            action="delete",
            actor_id=admin["id"],
            resource_id=experiment_id,
            payload={"deleted_at": datetime.utcnow().isoformat()},
            status="success"
        )

        return {"deleted": experiment_id}
    except Exception as e:
        await audit_logger.log_event(
            entity="experiment",
            action="delete",
            actor_id=admin["id"],
            resource_id=experiment_id,
            payload={"error": str(e)},
            status="failure"
        )
        raise
```

---

## Verification

### Verify Integrity

```python
from datetime import datetime, timedelta

# Verify all logs from last 24 hours
result = await audit_logger.verify_integrity(
    from_ts=datetime.utcnow() - timedelta(days=1),
    to_ts=datetime.utcnow()
)

if result["valid"]:
    print(f"✓ All {result['verified_count']} entries verified")
else:
    print(f"✗ Integrity issues found:")
    for tampered in result["tampered_logs"]:
        print(f"  - Tampered log {tampered['id']}: {tampered['error']}")
    for broken in result["broken_chains"]:
        print(f"  - Broken chain at {broken['id']}")

# Verify specific entity
result = await audit_logger.verify_integrity(
    entity="version"
)

# Verify specific time range
result = await audit_logger.verify_integrity(
    from_ts=datetime(2024, 12, 1),
    to_ts=datetime(2024, 12, 2)
)
```

### Get Audit Trail

```python
# Get all events for user
trail = await audit_logger.get_audit_trail(
    actor_id="user123",
    limit=100
)

# Get all promotions
trail = await audit_logger.get_audit_trail(
    entity="version",
    action="promote",
    limit=50
)

# Get events in time range
trail = await audit_logger.get_audit_trail(
    from_ts=datetime(2024, 12, 1),
    to_ts=datetime(2024, 12, 2),
    limit=1000
)

# Print trail
for entry in trail:
    print(f"{entry['timestamp']} - {entry['actor_id']} "
          f"{entry['action']} {entry['entity']} "
          f"({entry['status']})")
```

### Export for Compliance

```python
# Export as JSON
json_export = await audit_logger.export_audit_log(
    from_ts=datetime(2024, 12, 1),
    to_ts=datetime(2024, 12, 31),
    format="json"
)

with open("audit_log_2024_12.json", "w") as f:
    f.write(json_export)

# Export as CSV
csv_export = await audit_logger.export_audit_log(
    from_ts=datetime(2024, 12, 1),
    to_ts=datetime(2024, 12, 31),
    format="csv"
)

with open("audit_log_2024_12.csv", "w") as f:
    f.write(csv_export)
```

### Get Statistics

```python
# Get statistics for period
stats = await audit_logger.get_statistics(
    from_ts=datetime(2024, 12, 1),
    to_ts=datetime(2024, 12, 31)
)

print(f"Total entries: {stats['total_entries']}")
print(f"By entity: {stats['by_entity']}")
print(f"By action: {stats['by_action']}")
print(f"By actor: {stats['by_actor']}")
print(f"By status: {stats['by_status']}")
```

---

## Key Management

### Export Public Key

```python
# Export public key for verification
audit_logger.export_public_key("/secure/audit_public_key.pem")

# Share with compliance team
# They can verify signatures independently
```

### Backup Private Key

```python
# Backup with password protection
audit_logger.export_private_key(
    output_path="/backup/audit_private_key_backup.pem",
    password=b"backup-password"
)

# Store in secure location
```

### Key Rotation

```python
# Generate new key pair
new_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
    backend=default_backend()
)

# Create new logger with new key
new_audit_logger = TamperProofAuditLogger(
    db_client=db,
    private_key=new_private_key
)

# Log key rotation event
await audit_logger.log_event(
    entity="audit",
    action="key_rotation",
    actor_id="system",
    payload={
        "old_key_id": "key_2024_11",
        "new_key_id": "key_2024_12"
    }
)

# Switch to new logger
audit_logger = new_audit_logger
```

---

## Compliance

### GDPR Compliance

```python
# Right to be forgotten (pseudonymization)
# Instead of deleting, create new entry
await audit_logger.log_event(
    entity="user",
    action="pseudonymize",
    actor_id="compliance",
    resource_id=user_id,
    payload={
        "reason": "GDPR_right_to_be_forgotten",
        "pseudonym": "user_" + hashlib.sha256(user_id.encode()).hexdigest()[:8]
    }
)
```

### SOC 2 Compliance

```python
# Generate compliance report
stats = await audit_logger.get_statistics(
    from_ts=datetime(2024, 1, 1),
    to_ts=datetime(2024, 12, 31)
)

# Verify integrity
verification = await audit_logger.verify_integrity(
    from_ts=datetime(2024, 1, 1),
    to_ts=datetime(2024, 12, 31)
)

# Export audit log
export = await audit_logger.export_audit_log(
    from_ts=datetime(2024, 1, 1),
    to_ts=datetime(2024, 12, 31),
    format="json"
)

# Generate report
report = {
    "period": "2024",
    "total_entries": stats["total_entries"],
    "integrity_verified": verification["valid"],
    "export_available": True,
    "generated_at": datetime.utcnow().isoformat()
}
```

---

## Security Best Practices

1. **Key Management**

   - Store private key in HSM or KMS
   - Protect with strong password
   - Rotate keys regularly
   - Back up securely

2. **Access Control**

   - Restrict audit log access
   - Require authentication
   - Log all audit log access
   - Monitor for suspicious activity

3. **Integrity**

   - Verify chain regularly
   - Monitor for tampering
   - Alert on verification failures
   - Archive verified logs

4. **Retention**

   - Follow compliance requirements
   - Archive old logs
   - Maintain backups
   - Document retention policy

5. **Monitoring**
   - Monitor log volume
   - Alert on anomalies
   - Track key metrics
   - Review statistics regularly

---

## Troubleshooting

### Signature Verification Failed

```python
# Check if key is correct
public_key_pem = audit_logger.public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Compare with expected key
```

### Chain Broken

```python
# Investigate broken chain
result = await audit_logger.verify_integrity()

for broken in result["broken_chains"]:
    print(f"Broken at log {broken['id']}")
    print(f"Expected: {broken['expected_prev_hash']}")
    print(f"Actual: {broken['actual_prev_hash']}")

    # This indicates tampering
```

### Performance Issues

```python
# Verify in batches
batch_size = 1000
for i in range(0, total_logs, batch_size):
    from_ts = start_ts + timedelta(seconds=i * batch_duration)
    to_ts = from_ts + timedelta(seconds=batch_duration)

    result = await audit_logger.verify_integrity(
        from_ts=from_ts,
        to_ts=to_ts
    )
```

---

## Future Enhancements

- [ ] Blockchain integration
- [ ] Distributed verification
- [ ] Real-time monitoring
- [ ] Advanced analytics
- [ ] Machine learning anomaly detection
- [ ] Multi-signature support
