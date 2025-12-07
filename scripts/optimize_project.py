#!/usr/bin/env python3
"""
Project Optimization Script

Reduces redundancy by:
1. Identifying duplicate code and unused functions
2. Finding unused imports and resources
3. Consolidating documentation
4. Cleaning up dependencies
5. Measuring project metrics before/after

Run: python scripts/optimize_project.py [--dry-run] [--report]
"""

import os
import re
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class OptimizationReport:
    """Report of optimization findings and actions."""
    duplicate_files: List[Tuple[str, str]] = field(default_factory=list)
    unused_imports: Dict[str, List[str]] = field(default_factory=dict)
    duplicate_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    redundant_docs: List[str] = field(default_factory=list)
    unused_resources: List[str] = field(default_factory=list)
    size_before: int = 0
    size_after: int = 0
    files_removed: int = 0
    lines_removed: int = 0


class ProjectOptimizer:
    """Main optimization class."""

    # Directories to skip
    SKIP_DIRS = {
        'node_modules', '.git', '__pycache__', '.venv', 'venv',
        'dist', 'build', '.next', '.cache', 'coverage', '.pytest_cache'
    }

    # File patterns to analyze
    PYTHON_PATTERN = re.compile(r'\.py$')
    TS_PATTERN = re.compile(r'\.(ts|tsx)$')
    MD_PATTERN = re.compile(r'\.md$')
    REQUIREMENTS_PATTERN = re.compile(r'requirements.*\.txt$')

    # Import patterns
    PYTHON_IMPORT = re.compile(r'^(?:from\s+(\S+)\s+)?import\s+(.+)$', re.MULTILINE)
    TS_IMPORT = re.compile(r"^import\s+.*?from\s+['\"](.+?)['\"]", re.MULTILINE)

    def __init__(self, root_path: str, dry_run: bool = True):
        self.root = Path(root_path)
        self.dry_run = dry_run
        self.report = OptimizationReport()

    def run(self) -> OptimizationReport:
        """Run all optimizations."""
        print("=" * 60)
        print("Project Optimization Report")
        print("=" * 60)

        # Calculate initial size
        self.report.size_before = self._calculate_project_size()
        print(f"\nInitial project size: {self._format_size(self.report.size_before)}")

        # Run analysis
        print("\n1. Analyzing duplicate files...")
        self._find_duplicate_files()

        print("\n2. Analyzing Python imports...")
        self._analyze_python_imports()

        print("\n3. Analyzing duplicate dependencies...")
        self._analyze_dependencies()

        print("\n4. Analyzing documentation redundancy...")
        self._analyze_documentation()

        print("\n5. Finding unused resources...")
        self._find_unused_resources()

        # Calculate final size
        self.report.size_after = self._calculate_project_size()

        return self.report

    def _calculate_project_size(self) -> int:
        """Calculate total project size in bytes."""
        total = 0
        for path in self.root.rglob('*'):
            if path.is_file() and not self._should_skip(path):
                try:
                    total += path.stat().st_size
                except (OSError, PermissionError):
                    pass
        return total

    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        parts = path.parts
        return any(skip in parts for skip in self.SKIP_DIRS)

    def _format_size(self, size: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"

    def _get_file_hash(self, path: Path) -> str:
        """Get MD5 hash of file content."""
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except (OSError, PermissionError):
            return ""

    def _find_duplicate_files(self):
        """Find duplicate files by content hash."""
        hashes: Dict[str, List[Path]] = defaultdict(list)

        for path in self.root.rglob('*'):
            if path.is_file() and not self._should_skip(path):
                # Only check Python and TS files
                if self.PYTHON_PATTERN.search(str(path)) or self.TS_PATTERN.search(str(path)):
                    file_hash = self._get_file_hash(path)
                    if file_hash:
                        hashes[file_hash].append(path)

        # Find duplicates
        for file_hash, paths in hashes.items():
            if len(paths) > 1:
                self.report.duplicate_files.append(
                    (str(paths[0].relative_to(self.root)),
                     [str(p.relative_to(self.root)) for p in paths[1:]])
                )

        print(f"   Found {len(self.report.duplicate_files)} sets of duplicate files")

    def _analyze_python_imports(self):
        """Find potentially unused imports in Python files."""
        for path in self.root.rglob('*.py'):
            if self._should_skip(path):
                continue

            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                imports = self.PYTHON_IMPORT.findall(content)

                unused = []
                for module, names in imports:
                    # Check if imported names are used
                    for name in names.split(','):
                        name = name.strip().split(' as ')[-1].strip()
                        if name and name != '*':
                            # Simple check: is the name used elsewhere in the file?
                            pattern = re.compile(rf'\b{re.escape(name)}\b')
                            uses = len(pattern.findall(content))
                            if uses <= 1:  # Only the import itself
                                unused.append(name)

                if unused:
                    rel_path = str(path.relative_to(self.root))
                    self.report.unused_imports[rel_path] = unused[:5]  # Limit to 5
            except Exception:
                pass

        print(f"   Found {len(self.report.unused_imports)} files with potentially unused imports")

    def _analyze_dependencies(self):
        """Find duplicate dependencies across requirements files."""
        all_deps: Dict[str, List[str]] = defaultdict(list)

        for path in self.root.rglob('requirements*.txt'):
            if self._should_skip(path):
                continue

            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
                rel_path = str(path.relative_to(self.root))

                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-r'):
                        # Extract package name
                        match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                        if match:
                            pkg = match.group(1).lower()
                            all_deps[pkg].append(rel_path)
            except Exception:
                pass

        # Find packages in multiple files
        for pkg, files in all_deps.items():
            if len(files) > 3:  # Package appears in more than 3 files
                self.report.duplicate_dependencies[pkg] = files

        print(f"   Found {len(self.report.duplicate_dependencies)} packages duplicated across files")

    def _analyze_documentation(self):
        """Find redundant documentation files."""
        md_files: Dict[str, List[Path]] = defaultdict(list)

        for path in self.root.rglob('*.md'):
            if self._should_skip(path):
                continue

            try:
                # Group by similar names
                name = path.stem.lower().replace('_', '-').replace(' ', '-')
                md_files[name].append(path)
            except Exception:
                pass

        # Find similar named docs
        seen_content = {}
        for name, paths in md_files.items():
            if len(paths) > 1:
                for path in paths:
                    self.report.redundant_docs.append(str(path.relative_to(self.root)))

            # Also check for similar content
            for path in paths:
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')[:500]
                    content_hash = hashlib.md5(content.encode()).hexdigest()

                    if content_hash in seen_content:
                        self.report.redundant_docs.append(
                            f"{path.relative_to(self.root)} (similar to {seen_content[content_hash]})"
                        )
                    else:
                        seen_content[content_hash] = str(path.relative_to(self.root))
                except Exception:
                    pass

        print(f"   Found {len(self.report.redundant_docs)} potentially redundant doc files")

    def _find_unused_resources(self):
        """Find potentially unused static resources."""
        # Find all images and CSS files
        resources: Set[str] = set()
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.ico', '*.css']:
            for path in self.root.rglob(ext):
                if self._should_skip(path):
                    continue
                resources.add(path)

        # Check if they're referenced in code
        all_code = ""
        for ext in ['*.py', '*.ts', '*.tsx', '*.js', '*.jsx', '*.html']:
            for path in self.root.rglob(ext):
                if self._should_skip(path):
                    continue
                try:
                    all_code += path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    pass

        # Check each resource
        for resource in resources:
            name = resource.name
            if name not in all_code:
                self.report.unused_resources.append(str(resource.relative_to(self.root)))

        print(f"   Found {len(self.report.unused_resources)} potentially unused resources")

    def generate_report(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# Project Optimization Report",
            "",
            "## Summary",
            "",
            f"- **Initial Size**: {self._format_size(self.report.size_before)}",
            f"- **Final Size**: {self._format_size(self.report.size_after)}",
            f"- **Reduction**: {self._format_size(self.report.size_before - self.report.size_after)}",
            "",
            "## Findings",
            "",
        ]

        # Duplicate files
        lines.append("### Duplicate Files")
        if self.report.duplicate_files:
            for original, duplicates in self.report.duplicate_files[:10]:
                lines.append(f"- `{original}` duplicated in:")
                for dup in duplicates:
                    lines.append(f"  - `{dup}`")
        else:
            lines.append("No duplicate files found.")
        lines.append("")

        # Duplicate dependencies
        lines.append("### Duplicate Dependencies")
        lines.append("These packages appear in multiple requirements files:")
        if self.report.duplicate_dependencies:
            for pkg, files in list(self.report.duplicate_dependencies.items())[:15]:
                lines.append(f"- **{pkg}**: {len(files)} files")
        else:
            lines.append("No major duplications found.")
        lines.append("")

        # Unused imports
        lines.append("### Potentially Unused Imports")
        if self.report.unused_imports:
            for file, imports in list(self.report.unused_imports.items())[:10]:
                lines.append(f"- `{file}`: {', '.join(imports)}")
        else:
            lines.append("No unused imports found.")
        lines.append("")

        # Redundant docs
        lines.append("### Redundant Documentation")
        if self.report.redundant_docs:
            for doc in self.report.redundant_docs[:15]:
                lines.append(f"- `{doc}`")
        else:
            lines.append("No redundant documentation found.")
        lines.append("")

        # Recommendations
        lines.extend([
            "## Recommendations",
            "",
            "### 1. Dependency Consolidation",
            "- Create a shared `requirements-base.txt` (DONE)",
            "- Have service-specific files reference base: `-r ../requirements-base.txt`",
            "",
            "### 2. Documentation Consolidation",
            "- Merge similar docs into single comprehensive files",
            "- Create a docs index (MASTER_INDEX.md) linking all docs",
            "",
            "### 3. Code Optimization",
            "- Review and remove unused imports",
            "- Extract common utilities to shared modules",
            "",
            "### 4. Build Optimization",
            "- Use lazy loading for frontend components",
            "- Enable tree-shaking in production builds",
            "",
        ])

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Optimize project structure')
    parser.add_argument('--dry-run', action='store_true', help='Only analyze, don\'t modify')
    parser.add_argument('--report', action='store_true', help='Generate markdown report')
    parser.add_argument('--path', default='.', help='Project root path')
    args = parser.parse_args()

    # Find project root
    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Error: Path {root} does not exist")
        sys.exit(1)

    # Run optimization
    optimizer = ProjectOptimizer(str(root), dry_run=args.dry_run)
    report = optimizer.run()

    # Generate report
    if args.report:
        report_content = optimizer.generate_report()
        report_path = root / 'OPTIMIZATION_REPORT.md'
        report_path.write_text(report_content, encoding='utf-8')
        print(f"\nReport saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Duplicate files found: {len(report.duplicate_files)}")
    print(f"Files with unused imports: {len(report.unused_imports)}")
    print(f"Duplicated dependencies: {len(report.duplicate_dependencies)}")
    print(f"Redundant docs: {len(report.redundant_docs)}")
    print(f"Unused resources: {len(report.unused_resources)}")


if __name__ == '__main__':
    main()
