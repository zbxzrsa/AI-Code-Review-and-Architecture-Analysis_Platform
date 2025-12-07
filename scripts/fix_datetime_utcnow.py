#!/usr/bin/env python3
"""
修复 datetime.utcnow() 废弃警告脚本

脚本功能描述:
    在整个代码库中将 datetime.utcnow() 替换为 datetime.now(timezone.utc)。
    解决 Python 3.12+ 中的废弃警告，因为 datetime.utcnow() 返回的是
    无时区信息的日期时间，可能导致时区感知比较问题。

主要功能:
    - 扫描指定目录中的 Python 文件
    - 检测并替换 datetime.utcnow() 调用
    - 自动添加必要的 timezone 导入
    - 支持预览模式（dry-run）

使用方法:
    python scripts/fix_datetime_utcnow.py --dry-run  # 预览更改
    python scripts/fix_datetime_utcnow.py            # 应用更改

最后修改日期: 2024-12-07
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

# Directories to scan
SCAN_DIRS = [
    "backend",
    "ai_core",
    "services",
    "tests",
]

# File extensions to process
EXTENSIONS = [".py"]

# Files to skip
SKIP_FILES = [
    "fix_datetime_utcnow.py",  # This script
]

# Replacement patterns configuration
REPLACEMENT_PATTERNS = [
    # (pattern_for_counting, is_literal, replacement)
    (r"datetime\.utcnow\(\)", True, "datetime.utcnow()", "datetime.now(timezone.utc)"),
    (r"datetime\.datetime\.utcnow\(\)", True, "datetime.datetime.utcnow()", "datetime.datetime.now(timezone.utc)"),
    (r"default=datetime\.utcnow(?!\()", False, None, "default=lambda: datetime.now(timezone.utc)"),
    (r"onupdate=datetime\.utcnow(?!\()", False, None, "onupdate=lambda: datetime.now(timezone.utc)"),
    (r"default_factory=datetime\.utcnow(?!\()", False, None, "default_factory=lambda: datetime.now(timezone.utc)"),
]


def _count_and_apply_replacements(content: str) -> Tuple[str, int]:
    """Count occurrences and apply all replacements."""
    total_fixes = 0
    for pattern, is_literal, literal_str, replacement in REPLACEMENT_PATTERNS:
        count = len(re.findall(pattern, content))
        total_fixes += count
        if count > 0:
            if is_literal and literal_str:
                content = content.replace(literal_str, replacement)
            else:
                content = re.sub(pattern, replacement, content)
    return content, total_fixes


def _add_timezone_import(content: str) -> Tuple[str, bool]:
    """Add timezone import if needed. Returns (content, was_modified)."""
    has_timezone_import = (
        "from datetime import" in content and "timezone" in content
    ) or "import datetime" in content
    
    if has_timezone_import:
        return content, False
    
    import_pattern = r"from datetime import ([^;\n]+)"
    match = re.search(import_pattern, content)
    if match and "timezone" not in match.group(1):
        new_imports = match.group(1).rstrip() + ", timezone"
        content = re.sub(import_pattern, f"from datetime import {new_imports}", content, count=1)
        return content, True
    return content, False


def find_python_files(base_path: Path) -> List[Path]:
    """Find all Python files in the given directories."""
    files = []
    for scan_dir in SCAN_DIRS:
        dir_path = base_path / scan_dir
        if dir_path.exists():
            for ext in EXTENSIONS:
                files.extend(dir_path.rglob(f"*{ext}"))
    return files


def fix_file(file_path: Path, dry_run: bool = True) -> Tuple[int, List[str]]:
    """
    Fix datetime.utcnow() in a single file.
    
    Returns:
        Tuple of (number of fixes, list of changes)
    """
    if file_path.name in SKIP_FILES:
        return 0, []
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"  Error reading {file_path}: {e}")
        return 0, []
    
    original_content = content
    changes = []
    
    # Apply all replacements
    content, total_fixes = _count_and_apply_replacements(content)
    
    if total_fixes == 0:
        return 0, []
    
    # Add timezone import if needed
    content, import_added = _add_timezone_import(content)
    if import_added:
        changes.append("  Added 'timezone' to datetime imports")
    
    changes.append(f"  Replaced {total_fixes} occurrences of datetime.utcnow()")
    
    # Write changes if not dry run
    if not dry_run and content != original_content:
        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as e:
            print(f"  Error writing {file_path}: {e}")
            return 0, []
    
    return total_fixes, changes


def main():
    parser = argparse.ArgumentParser(
        description="Fix datetime.utcnow() → datetime.now(timezone.utc)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    args = parser.parse_args()
    
    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    print(f"Project root: {project_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLYING CHANGES'}")
    print("-" * 60)
    
    # Find all Python files
    files = find_python_files(project_root)
    print(f"Found {len(files)} Python files to scan")
    print("-" * 60)
    
    total_fixes = 0
    files_fixed = 0
    
    for file_path in sorted(files):
        fixes, changes = fix_file(file_path, dry_run=args.dry_run)
        if fixes > 0:
            rel_path = file_path.relative_to(project_root)
            print(f"\n{rel_path}:")
            for change in changes:
                print(change)
            total_fixes += fixes
            files_fixed += 1
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Files {'would be ' if args.dry_run else ''}modified: {files_fixed}")
    print(f"  Total fixes {'would be ' if args.dry_run else ''}applied: {total_fixes}")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
