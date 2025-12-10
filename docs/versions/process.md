# Version Communication Protocols

- Synchronization meetings: Weekly (Mon 09:30 UTC), participants: Tech Leads, QA, DevOps
- Update notifications: Slack channel `#version-updates` via CI workflow `version-notify.yml`
- Change impact assessments: Required in PR description; include performance, API, and risk analysis
- Escalation: Hotfix branches `hotfix/*` with immediate review, CI checks must pass
- Release cadence: Minor every 2 weeks, patch as needed; release notes generated from changelog

