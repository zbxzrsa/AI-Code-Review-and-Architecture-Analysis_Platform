#!/usr/bin/env python3
"""
Script to fix datetime.utcnow() → datetime.now(timezone.utc) across the codebase.

This addresses the deprecation warning in Python 3.12+ where datetime.utcnow() 
returns naive datetimes which can cause issues with timezone-aware comparisons.

Usage:
    python scripts/fix_datetime_utcnow.py --dry-run  # Preview changes
    python scripts/fix_datetime_utcnow.py            # Apply changes
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
    changes = []
    
    if file_path.name in SKIP_FILES:
        return 0, []
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return 0, []
    
    original_content = content
    
    # Pattern 1: datetime.utcnow() → datetime.now(timezone.utc)
    pattern1 = r"datetime\.utcnow\(\)"
    replacement1 = "datetime.now(timezone.utc)"
    
    # Pattern 2: datetime.datetime.utcnow() → datetime.datetime.now(timezone.utc)
    pattern2 = r"datetime\.datetime\.utcnow\(\)"
    replacement2 = "datetime.datetime.now(timezone.utc)"
    
    # Pattern 3: default=datetime.utcnow (SQLAlchemy factory) → lambda
    pattern3 = r"default=datetime\.utcnow(?!\()"
    replacement3 = "default=lambda: datetime.now(timezone.utc)"
    
    # Pattern 4: onupdate=datetime.utcnow (SQLAlchemy factory) → lambda
    pattern4 = r"onupdate=datetime\.utcnow(?!\()"
    replacement4 = "onupdate=lambda: datetime.now(timezone.utc)"
    
    # Pattern 5: default_factory=datetime.utcnow (Pydantic) → lambda
    pattern5 = r"default_factory=datetime\.utcnow(?!\()"
    replacement5 = "default_factory=lambda: datetime.now(timezone.utc)"
    
    # Count matches
    count1 = len(re.findall(pattern1, content))
    count2 = len(re.findall(pattern2, content))
    count3 = len(re.findall(pattern3, content))
    count4 = len(re.findall(pattern4, content))
    count5 = len(re.findall(pattern5, content))
    total_fixes = count1 + count2 + count3 + count4 + count5
    
    if total_fixes == 0:
        return 0, []
    
    # Apply replacements
    content = re.sub(pattern1, replacement1, content)
    content = re.sub(pattern2, replacement2, content)
    content = re.sub(pattern3, replacement3, content)
    content = re.sub(pattern4, replacement4, content)
    content = re.sub(pattern5, replacement5, content)
    
    # Check if we need to add timezone import
    needs_import = False
    if total_fixes > 0:
        # Check various import patterns
        has_timezone_import = (
            "from datetime import" in content and "timezone" in content
        ) or "import datetime" in content
        
        if not has_timezone_import:
            # Need to add timezone to imports
            needs_import = True
            
            # Try to add to existing datetime import
            import_pattern = r"from datetime import ([^;\n]+)"
            match = re.search(import_pattern, content)
            if match:
                existing_imports = match.group(1)
                if "timezone" not in existing_imports:
                    new_imports = existing_imports.rstrip() + ", timezone"
                    content = re.sub(
                        import_pattern,
                        f"from datetime import {new_imports}",
                        content,
                        count=1
                    )
                    changes.append(f"  Added 'timezone' to datetime imports")
    
    if total_fixes > 0:
        changes.append(f"  Replaced {total_fixes} occurrences of datetime.utcnow()")
    
    if not dry_run and content != original_content:
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
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
    print(f"Summary:")
    print(f"  Files {'would be ' if args.dry_run else ''}modified: {files_fixed}")
    print(f"  Total fixes {'would be ' if args.dry_run else ''}applied: {total_fixes}")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
