# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@ai-code-review.dev**

You should receive a response within 72 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Preferred Languages

We prefer all communications to be in English or Chinese.

## Security Update Process

1. **Report received** - We acknowledge receipt within 72 hours
2. **Triage** - We assess severity and impact within 1 week
3. **Fix development** - We develop and test a fix
4. **Disclosure** - We coordinate disclosure with reporter
5. **Release** - We release patched version
6. **Announcement** - We publish security advisory

## Security Best Practices

When using this platform:

- Keep your installation up to date
- Use strong passwords and enable 2FA
- Restrict API access with proper authentication
- Monitor logs for suspicious activity
- Follow the principle of least privilege
- Keep dependencies updated
- Use HTTPS in production
- Implement rate limiting
- Regular security audits

## Known Security Considerations

### API Keys

- Never commit API keys to version control
- Use environment variables or secret management systems
- Rotate keys regularly

### Database

- Use parameterized queries (we do)
- Enable SSL/TLS for database connections
- Regular backups with encryption

### Dependencies

- We use Dependabot for automated dependency updates
- Regular security audits with `safety` and `bandit`
- Minimal dependency footprint

## Security Features

- JWT-based authentication
- Role-based access control (RBAC)
- OPA policy engine integration
- Audit logging with cryptographic signatures
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection
- Rate limiting
- Circuit breakers for external services

## Compliance

This project aims to comply with:

- OWASP Top 10
- CWE Top 25
- GDPR (for EU users)
- SOC 2 Type II (in progress)

## Security Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities:

- [Your name could be here]

## Contact

- Security issues: security@ai-code-review.dev
- General inquiries: team@ai-code-review.dev
- PGP Key: [Available on request]
