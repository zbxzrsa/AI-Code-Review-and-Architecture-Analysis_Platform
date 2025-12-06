#!/usr/bin/env python3
"""
API Documentation Generator (Document Standardization #3)

Automatically generates API documentation from OpenAPI/Swagger specifications.

Features:
- Extract OpenAPI specs from FastAPI applications
- Generate HTML and PDF documentation
- Include test case examples
- Support multiple output formats
- Integrate with CI/CD pipeline

Usage:
    python scripts/generate_api_docs.py --output docs/api
    python scripts/generate_api_docs.py --format html,pdf --output docs/api
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """Represents an API endpoint."""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict] = field(default_factory=list)
    request_body: Optional[Dict] = None
    responses: Dict[str, Dict] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    security: List[Dict] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)


@dataclass
class APIDocConfig:
    """Configuration for API documentation generation."""
    title: str = "AI Code Review Platform API"
    version: str = "1.0.0"
    description: str = "API documentation for the AI Code Review Platform"
    base_url: str = "http://localhost:8000"
    output_dir: str = "docs/api"
    formats: List[str] = field(default_factory=lambda: ["html", "markdown"])
    include_examples: bool = True
    include_test_cases: bool = True
    language: str = "en"  # en or zh


class OpenAPIExtractor:
    """
    Extracts OpenAPI specification from FastAPI application.
    """
    
    def __init__(self, app_module: str = "backend.app.main:app"):
        self.app_module = app_module
        self.spec: Dict[str, Any] = {}
    
    def extract_from_app(self) -> Dict[str, Any]:
        """Extract OpenAPI spec from running FastAPI app."""
        try:
            # Import the FastAPI app
            module_path, app_name = self.app_module.rsplit(":", 1)
            
            # Try to import and get openapi schema
            import importlib
            module = importlib.import_module(module_path.replace("/", "."))
            app = getattr(module, app_name)
            
            self.spec = app.openapi()
            return self.spec
            
        except Exception as e:
            logger.warning(f"Could not extract from app: {e}")
            return self._get_default_spec()
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract OpenAPI spec from a file."""
        path = Path(file_path)
        
        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                self.spec = yaml.safe_load(f)
            else:
                self.spec = json.load(f)
        
        return self.spec
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """Extract OpenAPI spec from a URL."""
        import urllib.request
        
        with urllib.request.urlopen(url) as response:
            content = response.read().decode()
            if url.endswith((".yaml", ".yml")):
                self.spec = yaml.safe_load(content)
            else:
                self.spec = json.loads(content)
        
        return self.spec
    
    def _get_default_spec(self) -> Dict[str, Any]:
        """Return a default OpenAPI spec structure."""
        return {
            "openapi": "3.0.3",
            "info": {
                "title": "AI Code Review Platform API",
                "version": "1.0.0",
                "description": "API for code analysis and review",
            },
            "servers": [{"url": "http://localhost:8000"}],
            "paths": {},
            "components": {"schemas": {}, "securitySchemes": {}},
        }
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """Extract endpoint information from spec."""
        endpoints = []
        
        for path, methods in self.spec.get("paths", {}).items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "patch", "delete"]:
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=details.get("summary", ""),
                        description=details.get("description", ""),
                        parameters=details.get("parameters", []),
                        request_body=details.get("requestBody"),
                        responses=details.get("responses", {}),
                        tags=details.get("tags", []),
                        deprecated=details.get("deprecated", False),
                        security=details.get("security", []),
                    )
                    endpoints.append(endpoint)
        
        return endpoints


class MarkdownGenerator:
    """
    Generates Markdown documentation from OpenAPI spec.
    """
    
    def __init__(self, spec: Dict[str, Any], config: APIDocConfig):
        self.spec = spec
        self.config = config
    
    def generate(self) -> str:
        """Generate complete Markdown documentation."""
        sections = [
            self._generate_header(),
            self._generate_overview(),
            self._generate_authentication(),
            self._generate_endpoints_toc(),
            self._generate_endpoints(),
            self._generate_schemas(),
            self._generate_examples(),
            self._generate_footer(),
        ]
        
        return "\n\n".join(sections)
    
    def _generate_header(self) -> str:
        """Generate document header."""
        info = self.spec.get("info", {})
        
        return f"""# {info.get('title', 'API Documentation')}

| **Document Information** |                                    |
|--------------------------|-------------------------------------|
| **Version**              | {info.get('version', '1.0.0')}     |
| **Last Updated**         | {datetime.now().strftime('%Y-%m-%d')} |
| **Base URL**             | {self.spec.get('servers', [{}])[0].get('url', '')} |

---

## Change History

| Version | Date       | Description                   |
|---------|------------|-------------------------------|
| {info.get('version', '1.0.0')} | {datetime.now().strftime('%Y-%m-%d')} | Auto-generated documentation |"""
    
    def _generate_overview(self) -> str:
        """Generate overview section."""
        info = self.spec.get("info", {})
        
        return f"""---

## Overview

{info.get('description', 'API documentation for this service.')}

### Base URL

```
{self.spec.get('servers', [{}])[0].get('url', 'http://localhost:8000')}
```

### Response Format

All responses are in JSON format."""
    
    def _generate_authentication(self) -> str:
        """Generate authentication section."""
        security_schemes = self.spec.get("components", {}).get("securitySchemes", {})
        
        auth_docs = ["---\n\n## Authentication\n"]
        
        if not security_schemes:
            auth_docs.append("This API does not require authentication.")
        else:
            for name, scheme in security_schemes.items():
                scheme_type = scheme.get("type", "")
                
                if scheme_type == "http" and scheme.get("scheme") == "bearer":
                    auth_docs.append(f"""### {name}

**Type:** Bearer Token

Include the JWT token in the Authorization header:

```http
Authorization: Bearer YOUR_TOKEN
```""")
                elif scheme_type == "apiKey":
                    location = scheme.get("in", "header")
                    key_name = scheme.get("name", "X-API-Key")
                    auth_docs.append(f"""### {name}

**Type:** API Key

Include the API key in the {location}:

```http
{key_name}: YOUR_API_KEY
```""")
        
        return "\n\n".join(auth_docs)
    
    def _generate_endpoints_toc(self) -> str:
        """Generate table of contents for endpoints."""
        toc = ["---\n\n## API Endpoints\n\n### Table of Contents\n"]
        
        # Group by tags
        endpoints_by_tag: Dict[str, List] = {}
        for path, methods in self.spec.get("paths", {}).items():
            for method, details in methods.items():
                if method not in ["get", "post", "put", "patch", "delete"]:
                    continue
                
                tags = details.get("tags", ["General"])
                for tag in tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append({
                        "path": path,
                        "method": method.upper(),
                        "summary": details.get("summary", "")
                    })
        
        for tag, endpoints in sorted(endpoints_by_tag.items()):
            toc.append(f"\n**{tag}**\n")
            for ep in endpoints:
                anchor = f"{ep['method'].lower()}-{ep['path'].replace('/', '-').replace('{', '').replace('}', '')}"
                toc.append(f"- [{ep['method']} {ep['path']}](#{anchor}) - {ep['summary']}")
        
        return "\n".join(toc)
    
    def _generate_endpoints(self) -> str:
        """Generate detailed endpoint documentation."""
        sections = []
        
        for path, methods in self.spec.get("paths", {}).items():
            for method, details in methods.items():
                if method not in ["get", "post", "put", "patch", "delete"]:
                    continue
                
                section = self._generate_endpoint_section(path, method.upper(), details)
                sections.append(section)
        
        return "\n\n".join(sections)
    
    def _generate_endpoint_section(self, path: str, method: str, details: Dict) -> str:
        """Generate documentation for a single endpoint."""
        anchor = f"{method.lower()}-{path.replace('/', '-').replace('{', '').replace('}', '')}"
        deprecated_badge = " ⚠️ DEPRECATED" if details.get("deprecated") else ""
        
        doc = [f"""---

### {method} {path}{deprecated_badge} {{#{anchor}}}

{details.get('summary', '')}

{details.get('description', '')}

```http
{method} {path}
```"""]
        
        # Parameters
        params = details.get("parameters", [])
        if params:
            doc.append("\n#### Parameters\n")
            doc.append("| Name | In | Type | Required | Description |")
            doc.append("|------|-----|------|----------|-------------|")
            
            for param in params:
                required = "Yes" if param.get("required") else "No"
                param_type = param.get("schema", {}).get("type", "string")
                doc.append(f"| `{param.get('name')}` | {param.get('in')} | {param_type} | {required} | {param.get('description', '')} |")
        
        # Request Body
        request_body = details.get("requestBody")
        if request_body:
            doc.append("\n#### Request Body\n")
            content = request_body.get("content", {})
            
            if "application/json" in content:
                schema = content["application/json"].get("schema", {})
                example = content["application/json"].get("example")
                
                if example:
                    doc.append("```json")
                    doc.append(json.dumps(example, indent=2))
                    doc.append("```")
                elif schema:
                    doc.append(f"Schema: `{schema.get('$ref', 'Object')}`")
        
        # Responses
        responses = details.get("responses", {})
        if responses:
            doc.append("\n#### Responses\n")
            
            for status_code, response in responses.items():
                doc.append(f"\n**{status_code}** - {response.get('description', '')}\n")
                
                content = response.get("content", {})
                if "application/json" in content:
                    example = content["application/json"].get("example")
                    if example:
                        doc.append("```json")
                        doc.append(json.dumps(example, indent=2))
                        doc.append("```")
        
        return "\n".join(doc)
    
    def _generate_schemas(self) -> str:
        """Generate schema documentation."""
        schemas = self.spec.get("components", {}).get("schemas", {})
        
        if not schemas:
            return ""
        
        doc = ["---\n\n## Data Models\n"]
        
        for name, schema in schemas.items():
            doc.append(f"\n### {name}\n")
            
            if schema.get("description"):
                doc.append(f"{schema['description']}\n")
            
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            if properties:
                doc.append("| Property | Type | Required | Description |")
                doc.append("|----------|------|----------|-------------|")
                
                for prop_name, prop_schema in properties.items():
                    prop_type = prop_schema.get("type", "object")
                    is_required = "Yes" if prop_name in required else "No"
                    prop_desc = prop_schema.get("description", "")
                    doc.append(f"| `{prop_name}` | {prop_type} | {is_required} | {prop_desc} |")
        
        return "\n".join(doc)
    
    def _generate_examples(self) -> str:
        """Generate example section."""
        if not self.config.include_examples:
            return ""
        
        return """---

## Examples

### cURL

```bash
# Authentication
curl -X POST http://localhost:8000/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email": "user@example.com", "password": "password"}'

# Analyze Code
curl -X POST http://localhost:8000/api/v1/analyze \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"code": "def hello(): pass", "language": "python"}'
```

### Python

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "your-jwt-token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Analyze code
response = requests.post(
    f"{BASE_URL}/api/v1/analyze",
    headers=headers,
    json={"code": "def hello(): pass", "language": "python"}
)
print(response.json())
```

### JavaScript

```javascript
const BASE_URL = 'http://localhost:8000';

async function analyzeCode(code, language) {
  const response = await fetch(`${BASE_URL}/api/v1/analyze`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ code, language })
  });
  return response.json();
}
```"""
    
    def _generate_footer(self) -> str:
        """Generate document footer."""
        return f"""---

## Support

- **Documentation:** https://docs.example.com
- **API Status:** https://status.example.com
- **Support Email:** api-support@example.com

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by API Documentation Generator*
"""


class HTMLGenerator:
    """
    Generates HTML documentation from OpenAPI spec.
    """
    
    def __init__(self, spec: Dict[str, Any], config: APIDocConfig):
        self.spec = spec
        self.config = config
    
    def generate(self) -> str:
        """Generate HTML documentation."""
        info = self.spec.get("info", {})
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{info.get('title', 'API Documentation')}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        body {{ margin: 0; padding: 0; }}
        .swagger-ui .topbar {{ display: none; }}
        .custom-header {{
            background: #1a1a2e;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .custom-header h1 {{ margin: 0; }}
        .custom-header p {{ margin: 10px 0 0; opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="custom-header">
        <h1>{info.get('title', 'API Documentation')}</h1>
        <p>Version {info.get('version', '1.0.0')} | Generated {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {{
            SwaggerUIBundle({{
                spec: {json.dumps(self.spec)},
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout"
            }});
        }};
    </script>
</body>
</html>"""


class APIDocGenerator:
    """
    Main API documentation generator.
    """
    
    def __init__(self, config: Optional[APIDocConfig] = None):
        self.config = config or APIDocConfig()
        self.extractor = OpenAPIExtractor()
    
    def generate(
        self,
        source: str = "app",
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate API documentation.
        
        Args:
            source: Source for OpenAPI spec ("app", file path, or URL)
            output_dir: Output directory for generated docs
            
        Returns:
            Dict mapping format to output file path
        """
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract OpenAPI spec
        if source == "app":
            spec = self.extractor.extract_from_app()
        elif source.startswith("http"):
            spec = self.extractor.extract_from_url(source)
        else:
            spec = self.extractor.extract_from_file(source)
        
        # Save OpenAPI spec
        spec_path = Path(output_dir) / "openapi.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved OpenAPI spec: {spec_path}")
        
        outputs = {"openapi": str(spec_path)}
        
        # Generate Markdown
        if "markdown" in self.config.formats or "md" in self.config.formats:
            md_generator = MarkdownGenerator(spec, self.config)
            md_content = md_generator.generate()
            
            md_path = Path(output_dir) / "api-reference.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Generated Markdown: {md_path}")
            outputs["markdown"] = str(md_path)
        
        # Generate HTML
        if "html" in self.config.formats:
            html_generator = HTMLGenerator(spec, self.config)
            html_content = html_generator.generate()
            
            html_path = Path(output_dir) / "index.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Generated HTML: {html_path}")
            outputs["html"] = str(html_path)
        
        # Generate PDF (requires wkhtmltopdf or similar)
        if "pdf" in self.config.formats:
            try:
                html_path = Path(output_dir) / "index.html"
                pdf_path = Path(output_dir) / "api-reference.pdf"
                
                # Try using wkhtmltopdf
                result = subprocess.run(
                    ["wkhtmltopdf", str(html_path), str(pdf_path)],
                    capture_output=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Generated PDF: {pdf_path}")
                    outputs["pdf"] = str(pdf_path)
                else:
                    logger.warning("PDF generation failed (wkhtmltopdf not available)")
            except FileNotFoundError:
                logger.warning("PDF generation skipped (wkhtmltopdf not installed)")
        
        return outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate API documentation from OpenAPI specifications"
    )
    parser.add_argument(
        "--source",
        default="app",
        help="Source for OpenAPI spec: 'app', file path, or URL"
    )
    parser.add_argument(
        "--output",
        default="docs/api",
        help="Output directory for generated documentation"
    )
    parser.add_argument(
        "--formats",
        default="html,markdown",
        help="Comma-separated list of output formats (html, markdown, pdf)"
    )
    parser.add_argument(
        "--title",
        default="AI Code Review Platform API",
        help="API documentation title"
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="API version"
    )
    
    args = parser.parse_args()
    
    config = APIDocConfig(
        title=args.title,
        version=args.version,
        output_dir=args.output,
        formats=args.formats.split(","),
    )
    
    generator = APIDocGenerator(config)
    outputs = generator.generate(source=args.source, output_dir=args.output)
    
    print("\n✅ API documentation generated successfully!")
    print("\nGenerated files:")
    for format_name, path in outputs.items():
        print(f"  - {format_name}: {path}")


if __name__ == "__main__":
    main()
