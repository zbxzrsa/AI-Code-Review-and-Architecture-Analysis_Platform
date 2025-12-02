# Contributing Guide

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Development Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- PostgreSQL 13+ (or use Docker)

### Local Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd AI-Code-Review-and-Architecture-Analysis_Platform
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
# For V2 Production
cd backend/v2-production
pip install -r requirements.txt

# For V1 Experimentation
cd ../v1-experimentation
pip install -r requirements.txt

# For V3 Quarantine
cd ../v3-quarantine
pip install -r requirements.txt
```

4. **Start services**

```bash
docker-compose up -d
```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:

- `feature/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation updates
- `refactor/`: Code refactoring
- `test/`: Test additions

### Code Style

- **Python**: Follow PEP 8
- **Formatting**: Use `black` for Python code
- **Linting**: Use `pylint` or `flake8`
- **Type hints**: Use type hints for all functions

```bash
# Format code
black backend/

# Lint code
pylint backend/

# Check types
mypy backend/
```

### Testing

Write tests for all new features:

```bash
# Run tests
pytest backend/v2-production/tests/
pytest backend/v1-experimentation/tests/
pytest backend/v3-quarantine/tests/

# Run with coverage
pytest --cov=backend backend/
```

### Commit Messages

Follow conventional commits:

```
type(scope): subject

body

footer
```

Examples:

- `feat(v2): add code review caching`
- `fix(v1): correct evaluation threshold calculation`
- `docs(deployment): update Kubernetes guide`
- `refactor(shared): extract AI client logic`

### Pull Request Process

1. **Create PR with descriptive title**

   - Reference related issues: `Closes #123`
   - Describe changes clearly
   - Include testing notes

2. **Code review checklist**

   - [ ] Code follows style guide
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   - [ ] SLO impact assessed (for V2)

3. **Approval and merge**
   - Requires 2 approvals for V2 changes
   - Requires 1 approval for V1 changes
   - Requires 1 approval for V3 changes

## Architecture Guidelines

### Version-Specific Development

#### V2 Production Changes

- **Impact**: Affects all end users
- **Testing**: Comprehensive testing required
- **Deployment**: Blue-green or canary deployment
- **Rollback**: Must be reversible
- **SLO**: Must not violate SLOs

#### V1 Experimentation Changes

- **Impact**: Only affects experiments
- **Testing**: Unit tests required
- **Deployment**: Rolling update acceptable
- **Rollback**: Can be manual

#### V3 Quarantine Changes

- **Impact**: Read-only archive
- **Testing**: Basic testing required
- **Deployment**: Can be immediate
- **Rollback**: Not critical

### Adding New AI Providers

1. **Create provider class** in `backend/shared/utils/ai_client.py`
2. **Implement AIProvider interface**
3. **Add configuration** to `backend/shared/config/settings.py`
4. **Add tests** for provider
5. **Update documentation**

Example:

```python
class CustomProvider(AIProvider):
    async def analyze_code(self, code: str, language: str, prompt_template: str) -> AIResponse:
        # Implementation
        pass

    async def health_check(self) -> bool:
        # Implementation
        pass
```

### Adding New Metrics

1. **Define metric** in Prometheus format
2. **Add to monitoring** middleware
3. **Add to Grafana** dashboard
4. **Document** in operations guide

## Database Changes

### Schema Migrations

Use Alembic for migrations:

```bash
# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Backward Compatibility

- Never remove columns (mark as deprecated)
- Always provide default values
- Test migrations with production data

## Documentation

### Update Documentation When

- Adding new API endpoints
- Changing deployment process
- Adding new features
- Fixing bugs that affect operations

### Documentation Files

- `README.md`: Project overview
- `docs/architecture.md`: System design
- `docs/deployment.md`: Deployment guide
- `docs/api-reference.md`: API documentation
- `docs/operations.md`: Operations runbook

## Performance Considerations

### V2 Production

- Target: < 3s p95 response time
- Monitor: Query performance, AI provider latency
- Optimize: Database indexes, caching, connection pooling

### V1 Experimentation

- Target: Flexible, for testing
- Monitor: Experiment success rate, metrics accuracy
- Optimize: Experiment execution time

### V3 Quarantine

- Target: Minimal resource usage
- Monitor: Archive integrity
- Optimize: Read-only queries

## Security Guidelines

### Code Security

- Never hardcode secrets
- Use environment variables for sensitive data
- Validate all inputs
- Use parameterized queries
- Implement rate limiting

### Dependency Security

```bash
# Check for vulnerabilities
pip audit

# Update dependencies
pip install --upgrade pip
pip install -U -r requirements.txt
```

### API Security

- Implement authentication (JWT, OAuth)
- Use HTTPS in production
- Implement CORS properly
- Add request validation
- Implement rate limiting

## Release Process

### Version Numbering

Use semantic versioning: `MAJOR.MINOR.PATCH`

- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes

### Release Steps

1. **Update version** in `__init__.py` and `setup.py`
2. **Update CHANGELOG.md**
3. **Create release branch**: `release/v1.2.0`
4. **Tag release**: `git tag v1.2.0`
5. **Build and push** Docker images
6. **Deploy** to staging, then production

## Reporting Issues

### Bug Report Template

```markdown
## Description

Brief description of the bug

## Steps to Reproduce

1. Step 1
2. Step 2
3. Step 3

## Expected Behavior

What should happen

## Actual Behavior

What actually happens

## Environment

- OS:
- Python version:
- Docker version:
- Kubernetes version:

## Logs

Relevant logs or error messages
```

### Feature Request Template

```markdown
## Description

Brief description of the feature

## Use Case

Why is this feature needed?

## Proposed Solution

How should it work?

## Alternatives

Other possible approaches

## Impact

Which version(s) affected?
```

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing issues
- **Discussions**: Start a discussion for questions
- **Slack**: Join team Slack channel

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guide
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No console.log or debug statements
- [ ] No hardcoded secrets
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts

## Continuous Integration

All PRs must pass:

- Linting (pylint, flake8)
- Type checking (mypy)
- Unit tests (pytest)
- Integration tests
- Security scanning

## Performance Benchmarking

For performance-sensitive changes:

```bash
# Run benchmarks
pytest --benchmark backend/

# Compare with baseline
pytest --benchmark --benchmark-compare backend/
```

## Accessibility

- Write clear, descriptive error messages
- Use consistent terminology
- Document complex logic
- Provide examples in documentation

## Questions?

- Check existing documentation
- Search closed issues
- Ask in discussions
- Contact maintainers

Thank you for contributing!
