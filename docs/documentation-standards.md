# Documentation Standardization Guide

| **Document Information** |                |
| ------------------------ | -------------- |
| **Version**              | 1.0.0          |
| **Status**               | Approved       |
| **Last Updated**         | 2024-12-06     |
| **Language**             | English / 中文 |

---

## Change History

| Version | Date       | Author             | Description                |
| ------- | ---------- | ------------------ | -------------------------- |
| 1.0.0   | 2024-12-06 | Documentation Team | Initial standards document |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Document Templates](#2-document-templates)
3. [Bilingual Documentation](#3-bilingual-documentation)
4. [API Documentation](#4-api-documentation)
5. [Architecture Decision Records](#5-architecture-decision-records)
6. [Document Workflow](#6-document-workflow)

---

## 1. Overview

This guide defines the documentation standards for the AI Code Review Platform. All project documentation must follow these guidelines to ensure consistency, maintainability, and accessibility.

### 1.1 Goals

- **Consistency**: All documents follow the same structure and format
- **Bilingual Support**: Core documents available in English and Chinese
- **Automation**: API documentation generated automatically
- **Traceability**: All architecture decisions are recorded

### 1.2 Document Types

| Type                   | Template                        | Priority                           |
| ---------------------- | ------------------------------- | ---------------------------------- |
| General Documents      | `DOCUMENT_TEMPLATE.md`          | Required                           |
| Technical Design       | `TECHNICAL_DESIGN_TEMPLATE.md`  | Required for new features          |
| API Documentation      | `API_DOCUMENTATION_TEMPLATE.md` | Auto-generated                     |
| User Manuals           | `USER_MANUAL_TEMPLATE.md`       | Required for user-facing features  |
| Architecture Decisions | `ADR_TEMPLATE.md`               | Required for significant decisions |

---

## 2. Document Templates

### 2.1 Template Location

All templates are located in `docs/templates/`:

```
docs/templates/
├── DOCUMENT_TEMPLATE.md           # General document template
├── DOCUMENT_TEMPLATE_ZH.md        # General template (Chinese)
├── TECHNICAL_DESIGN_TEMPLATE.md   # Technical design template
├── API_DOCUMENTATION_TEMPLATE.md  # API documentation template
├── USER_MANUAL_TEMPLATE.md        # User manual template
├── ADR_TEMPLATE.md                # Architecture Decision Record
└── ADR_TEMPLATE_ZH.md             # ADR template (Chinese)
```

### 2.2 Required Elements

Every document MUST include:

| Element            | Description              | Example                                   |
| ------------------ | ------------------------ | ----------------------------------------- |
| **Title**          | Clear, descriptive title | `# Technical Design: User Authentication` |
| **Version**        | Semantic version number  | `1.0.0`                                   |
| **Status**         | Current document status  | `Draft`, `In Review`, `Approved`          |
| **Author**         | Document creator         | `John Smith`                              |
| **Created Date**   | Creation date            | `2024-01-15`                              |
| **Last Updated**   | Last modification date   | `2024-02-20`                              |
| **Change History** | Table of changes         | See template                              |

### 2.3 Status Definitions

| Status         | Description                           |
| -------------- | ------------------------------------- |
| **Draft**      | Initial writing, not ready for review |
| **In Review**  | Under review by stakeholders          |
| **Approved**   | Reviewed and approved for use         |
| **Deprecated** | No longer current, kept for reference |
| **Superseded** | Replaced by a newer document          |

### 2.4 Creating New Documents

```bash
# 1. Copy appropriate template
cp docs/templates/TECHNICAL_DESIGN_TEMPLATE.md docs/designs/my-feature-design.md

# 2. Fill in document information
# 3. Write content following template structure
# 4. Submit for review
```

---

## 3. Bilingual Documentation

### 3.1 Language Requirements

| Document Type    | English     | Chinese     | Notes                   |
| ---------------- | ----------- | ----------- | ----------------------- |
| README           | ✅ Required | ✅ Required | Main project README     |
| Quick Start      | ✅ Required | ✅ Required | Getting started guide   |
| User Manual      | ✅ Required | ✅ Required | End-user documentation  |
| API Reference    | ✅ Required | Optional    | Auto-generated          |
| Technical Design | ✅ Required | Optional    | Developer documentation |
| ADR              | ✅ Required | Optional    | Architecture decisions  |

### 3.2 File Organization

```
docs/
├── README.md                 # English (primary)
├── QUICKSTART.md             # English
├── zh-CN/                    # Chinese translations
│   ├── README.md             # Chinese README
│   ├── QUICKSTART.md         # Chinese Quick Start
│   └── user-manual.md        # Chinese User Manual
├── api/                      # API documentation
│   ├── openapi.yaml          # OpenAPI specification
│   ├── api-reference.md      # Generated Markdown
│   └── index.html            # Generated HTML
├── adr/                      # Architecture Decision Records
│   ├── README.md             # ADR index
│   └── ADR-0001-*.md         # Individual ADRs
└── templates/                # Document templates
```

### 3.3 Translation Guidelines

#### Technical Terms

Maintain consistency in translating technical terms:

| English        | Chinese                 | Notes             |
| -------------- | ----------------------- | ----------------- |
| API            | API                     | Keep as-is        |
| Repository     | 仓库                    |                   |
| Pull Request   | Pull Request / 拉取请求 | Either acceptable |
| Deploy         | 部署                    |                   |
| Container      | 容器                    |                   |
| Microservice   | 微服务                  |                   |
| Cache          | 缓存                    |                   |
| Authentication | 认证                    |                   |
| Authorization  | 授权                    |                   |

#### Translation Workflow

1. **Create English Version First**: Always start with English
2. **Mark for Translation**: Add label or tag for translation
3. **Translate**: Create Chinese version in `zh-CN/` directory
4. **Review**: Have native speaker review translation
5. **Sync Updates**: Keep versions synchronized

### 3.4 Language Switching

Documents should include a language switcher:

```markdown
[English](../README.md) | [中文](./README.md)
```

---

## 4. API Documentation

### 4.1 Generation Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│   OpenAPI    │────▶│   HTML/MD    │
│   Endpoints  │     │   Spec       │     │   Docs       │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 4.2 Running the Generator

```bash
# Generate documentation
python scripts/generate_api_docs.py \
  --source app \
  --output docs/api \
  --formats html,markdown

# Generate with PDF
python scripts/generate_api_docs.py \
  --formats html,markdown,pdf \
  --output docs/api
```

### 4.3 CI/CD Integration

API documentation is automatically generated on:

- Push to `main` or `develop` branches
- Changes to `backend/**` or `docs/**`
- Manual workflow trigger

### 4.4 Code Comments

All API endpoints MUST have proper docstrings:

```python
@app.post("/api/v1/analyze")
async def analyze_code(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze code for issues.

    Performs AI-powered analysis on the provided code to detect:
    - Security vulnerabilities
    - Performance issues
    - Code quality problems

    Args:
        request: Analysis request containing code and options

    Returns:
        AnalyzeResponse: Analysis results with issues and metrics

    Raises:
        HTTPException: 400 if invalid request
        HTTPException: 429 if rate limited
    """
    ...
```

### 4.5 Output Formats

| Format           | Use Case            | Location                     |
| ---------------- | ------------------- | ---------------------------- |
| **OpenAPI YAML** | Programmatic access | `docs/api/openapi.yaml`      |
| **HTML**         | Web viewing         | `docs/api/index.html`        |
| **Markdown**     | Git-friendly        | `docs/api/api-reference.md`  |
| **PDF**          | Offline/print       | `docs/api/api-reference.pdf` |

---

## 5. Architecture Decision Records

### 5.1 When to Create an ADR

Create an ADR for:

- ✅ Major architectural changes
- ✅ Technology choices (frameworks, databases, etc.)
- ✅ Design patterns adoption
- ✅ API versioning decisions
- ✅ Security-related decisions
- ✅ Performance optimization strategies

Do NOT create an ADR for:

- ❌ Bug fixes
- ❌ Minor refactoring
- ❌ Routine maintenance
- ❌ UI changes

### 5.2 ADR Workflow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Proposed │───▶│ Reviewed │───▶│ Accepted │───▶│ Implement│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │
                     ▼
               ┌──────────┐
               │ Rejected │
               └──────────┘
```

### 5.3 Creating an ADR

```bash
# 1. Get next ADR number
NEXT_NUM=$(ls docs/adr/ADR-*.md 2>/dev/null | wc -l)
NEXT_NUM=$(printf "%04d" $((NEXT_NUM + 1)))

# 2. Create ADR from template
cp docs/templates/ADR_TEMPLATE.md "docs/adr/ADR-${NEXT_NUM}-your-decision.md"

# 3. Fill in the template
# 4. Submit PR for review
# 5. Get approval from ADR Review Committee
```

### 5.4 ADR Review Committee

For significant decisions, require approval from:

| Role          | Required    | Notes                       |
| ------------- | ----------- | --------------------------- |
| Tech Lead     | ✅ Yes      | Always required             |
| Architect     | ✅ Yes      | For architectural decisions |
| Security Lead | Conditional | For security decisions      |
| Product Owner | Conditional | For user-facing changes     |

### 5.5 ADR Lifecycle

| Status     | Next States            | Description                   |
| ---------- | ---------------------- | ----------------------------- |
| Proposed   | Accepted, Rejected     | Initial proposal              |
| Accepted   | Deprecated, Superseded | Approved for implementation   |
| Deprecated | -                      | No longer recommended         |
| Superseded | -                      | Replaced by newer ADR         |
| Rejected   | Proposed               | Not approved (can be revised) |

---

## 6. Document Workflow

### 6.1 Creation Process

```
1. Identify need for documentation
       ↓
2. Select appropriate template
       ↓
3. Create draft
       ↓
4. Self-review
       ↓
5. Submit for peer review
       ↓
6. Address feedback
       ↓
7. Get approval
       ↓
8. Publish / Merge
```

### 6.2 Review Checklist

Before submitting for review:

- [ ] Used correct template
- [ ] Filled in all required metadata
- [ ] Spell-checked content
- [ ] Verified all links work
- [ ] Included code examples (if applicable)
- [ ] Created Chinese version (if required)
- [ ] Updated related documents (if needed)

### 6.3 Maintenance

| Task                 | Frequency      | Responsible        |
| -------------------- | -------------- | ------------------ |
| Review outdated docs | Monthly        | Documentation Team |
| Update API docs      | On code change | CI/CD Pipeline     |
| Sync translations    | Quarterly      | Documentation Team |
| ADR review           | On decision    | ADR Committee      |

---

## Quick Reference

### Commands

```bash
# Generate API docs
python scripts/generate_api_docs.py --output docs/api

# Validate markdown
markdownlint 'docs/**/*.md'

# Check links
markdown-link-check docs/README.md
```

### File Naming

| Type         | Pattern                     | Example                         |
| ------------ | --------------------------- | ------------------------------- |
| General docs | `lowercase-with-hyphens.md` | `getting-started.md`            |
| ADR          | `ADR-XXXX-description.md`   | `ADR-0001-three-version.md`     |
| Templates    | `UPPERCASE_TEMPLATE.md`     | `ADR_TEMPLATE.md`               |
| Chinese docs | Same as English             | `docs/zh-CN/getting-started.md` |

---

## Support

- **Questions**: Create issue with `documentation` label
- **Template Updates**: Submit PR to `docs/templates/`
- **Translation Help**: Contact documentation team

---

**Version:** 1.0.0 | **Last Updated:** 2024-12-06
