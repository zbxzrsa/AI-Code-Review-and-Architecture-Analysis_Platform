# Upgrade & Migration Guide

- Pre-checks: Ensure databases and caches are healthy; back up critical data
- Compatibility: Validate against `docs/versions/COMPATIBILITY.md` before rollout
- Rollout: Use gray-scale promotion (1%→5%→25%→50%→100%) with SLO monitoring
- Rollback: Trigger automatic abort on SLO violation; restore previous baseline
- Data migrations: Apply additive schemas; provide fallback transforms for consumers
- Client updates: Communicate breaking changes at least one release ahead with deprecation notices

