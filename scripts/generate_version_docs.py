import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return res.stdout.strip()


def collect_endpoints(base: Path) -> List[str]:
    endpoints: List[str] = []
    for py in base.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"@router\.(get|post|put|delete|patch)\([\s\S]*?\)\n\s*async\s+def\s+(\w+)", text):
            method = m.group(1).upper()
            # rough path extraction
            path_m = re.search(r"@router\.(?:get|post|put|delete|patch)\(\s*['\"]([^'\"]+)['\"]", m.group(0))
            path = path_m.group(1) if path_m else "<unknown>"
            endpoints.append(f"{method} {path}")
    return sorted(set(endpoints))


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate_changelog() -> None:
    try:
        log = run(["git", "log", "--pretty=format:%h %s", "-n", "200"]).splitlines()
    except Exception:
        log = []
    groups: Dict[str, List[str]] = {"feat": [], "fix": [], "docs": [], "perf": [], "refactor": [], "test": [], "chore": []}
    for line in log:
        msg = line.split(" ", 1)[1] if " " in line else line
        m = re.match(r"(feat|fix|docs|perf|refactor|test|chore)(\([^)]*\))?:\s*(.*)", msg)
        if m:
            groups[m.group(1)].append(msg)
        else:
            groups.setdefault("misc", []).append(msg)
    out = ["# Changelog (auto-generated)", "", "This file is generated from Conventional Commits.", ""]
    for k, items in groups.items():
        if items:
            out.append(f"## {k}")
            out.extend([f"- {i}" for i in items])
            out.append("")
    write_markdown(ROOT / "docs" / "versions" / "CHANGELOG.md", "\n".join(out))


def generate_endpoint_diffs() -> None:
    v1 = collect_endpoints(ROOT / "backend" / "v1-experimentation" / "src")
    v2 = collect_endpoints(ROOT / "backend" / "v2-production" / "src")
    v3 = collect_endpoints(ROOT / "backend" / "v3-quarantine" / "src")
    def section(name: str, eps: List[str]) -> str:
        lines = [f"### {name}", ""] + [f"- {e}" for e in eps] + [""]
        return "\n".join(lines)
    common = sorted(set(v1) & set(v2))
    added_v1 = sorted(set(v1) - set(v2))
    removed_v1 = sorted(set(v2) - set(v1))
    added_v3 = sorted(set(v3) - set(v2))
    content = ["# Version Endpoint Differences", "", "## Common (V1 & V2)", ""]
    content += [f"- {e}" for e in common] + [""]
    content.append(section("Added in V1 (vs V2)", added_v1))
    content.append(section("Removed from V1 (present in V2)", removed_v1))
    content.append(section("V3-only (vs V2)", added_v3))
    write_markdown(ROOT / "docs" / "versions" / "DIFFS.md", "\n".join(content))


def generate_dependencies() -> None:
    paths = [ROOT / "backend" / "requirements.txt", ROOT / "backend" / "requirements-dev.txt", ROOT / "backend" / "requirements-test.txt"]
    deps: List[Tuple[str, str]] = []
    for p in paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    deps.append((p.name, line.strip()))
    out = ["# Version Dependencies", "", "This lists Python dependencies used by backend.", ""]
    for src, pkg in deps:
        out.append(f"- `{src}`: {pkg}")
    write_markdown(ROOT / "docs" / "versions" / "DEPENDENCIES.md", "\n".join(out))


def generate_compat_guidelines() -> None:
    content = "\n".join([
        "# Backward Compatibility Guidelines",
        "",
        "- Avoid breaking API changes; use additive endpoints and versioned paths.",
        "- Maintain data schema migrations with backward-compatible transformations.",
        "- Deprecate features with clear timelines and provide shims in V3.",
        "- Preserve request/response contracts; document any optional fields.",
        "- Ensure monitoring covers latency, error rate, and accuracy deltas per version.",
    ])
    write_markdown(ROOT / "docs" / "versions" / "COMPATIBILITY.md", content)


def main() -> None:
    generate_changelog()
    generate_endpoint_diffs()
    generate_dependencies()
    generate_compat_guidelines()


if __name__ == "__main__":
    main()
