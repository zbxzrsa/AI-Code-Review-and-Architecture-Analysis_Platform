#!/usr/bin/env python3
"""
Self-Healing Fix Application Script

This script applies all stored fixes from the error knowledge base
to resolve known issues automatically.

Usage:
    python scripts/apply_self_healing_fixes.py [--dry-run] [--error-id ERROR_ID]
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Apply self-healing fixes')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--error-id', type=str, help='Apply fix for specific error ID only')
    parser.add_argument('--list', action='store_true', help='List all known errors')
    parser.add_argument('--stats', action='store_true', help='Show knowledge base statistics')
    args = parser.parse_args()

    # Import after path setup
    from shared.self_healing import get_knowledge_base, RepairStatus

    kb = get_knowledge_base()

    if args.stats:
        stats = kb.get_statistics()
        print("\n" + "=" * 60)
        print("ERROR KNOWLEDGE BASE STATISTICS")
        print("=" * 60)
        print(f"Total errors stored: {stats['total_errors_stored']}")
        print(f"Total repairs executed: {stats['total_repairs_executed']}")
        print(f"Successful repairs: {stats['successful_repairs']}")
        print(f"Failed repairs: {stats['failed_repairs']}")
        print(f"Patterns matched: {stats['patterns_matched']}")
        print(f"Auto-repair enabled: {stats['auto_repair_enabled_count']}")
        print("\nErrors by severity:")
        for severity, count in stats['errors_by_severity'].items():
            if count > 0:
                print(f"  {severity}: {count}")
        print("\nErrors by category:")
        for category, count in stats['errors_by_category'].items():
            if count > 0:
                print(f"  {category}: {count}")
        return

    if args.list:
        errors = kb.get_all_errors()
        print("\n" + "=" * 60)
        print("KNOWN ERROR PATTERNS")
        print("=" * 60)
        for error in errors:
            print(f"\n[{error['severity'].upper()}] {error['error_id']}")
            print(f"  Title: {error['title']}")
            print(f"  File: {error['file_path']}")
            print(f"  Category: {error['category']}")
            print(f"  Auto-repair: {'Yes' if error['auto_repair_enabled'] else 'No'}")
            print(f"  Fixes: {len(error['fixes'])}")
        return

    print("\n" + "=" * 60)
    print("SELF-HEALING FIX APPLICATION")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")

    # Get errors to fix
    if args.error_id:
        error = kb.get_error(args.error_id)
        if not error:
            print(f"Error {args.error_id} not found")
            return
        errors_to_fix = [error]
    else:
        errors_to_fix = list(kb.errors.values())

    # Sort by priority
    errors_to_fix.sort(key=lambda e: e.priority)

    print(f"Found {len(errors_to_fix)} errors to process\n")

    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0
    }

    for error in errors_to_fix:
        print(f"\n{'─' * 50}")
        print(f"Processing: {error.error_id}")
        print(f"  Title: {error.title}")
        print(f"  Severity: {error.severity.value}")
        print(f"  Priority: {error.priority}")
        print(f"  Fixes to apply: {len(error.fixes)}")

        if not error.auto_repair_enabled:
            print("  Status: SKIPPED (auto-repair disabled)")
            results["skipped"] += 1
            continue

        log = await kb.auto_repair(error.error_id, dry_run=args.dry_run)

        if log.status in [RepairStatus.SUCCESS, RepairStatus.VERIFIED]:
            print(f"  Status: {log.status.value.upper()}")
            if log.files_modified:
                print(f"  Files modified: {', '.join(log.files_modified)}")
            if log.verification_results:
                print(f"  Verification: {log.verification_results}")
            results["success"] += 1
        elif log.status == RepairStatus.SKIPPED:
            print(f"  Status: SKIPPED")
            print(f"  Reason: {log.error_message}")
            results["skipped"] += 1
        else:
            print(f"  Status: FAILED")
            print(f"  Error: {log.error_message}")
            results["failed"] += 1

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successful: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Total: {sum(results.values())}")


if __name__ == "__main__":
    asyncio.run(main())
