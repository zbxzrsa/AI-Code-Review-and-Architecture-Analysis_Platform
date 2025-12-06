#!/usr/bin/env python3
"""
Batch fix Kubernetes security issues:
1. Add automountServiceAccountToken: false
2. Add ephemeral-storage limits to all containers
"""

import re
from pathlib import Path

def fix_deployment_file(file_path: Path) -> bool:
    """Fix a single deployment file."""
    print(f"Processing {file_path}...")
    
    content = file_path.read_text(encoding='utf-8')
    original_content = content
    
    # Fix 1: Add automountServiceAccountToken: false after serviceAccountName
    # Pattern: spec:\n      serviceAccountName: (something)
    # Add: automountServiceAccountToken: false after serviceAccountName
    pattern1 = r'(\n {4}spec:\n(?: {6}serviceAccountName: [^\n]+\n)?)( {6}securityContext:)'
    replacement1 = r'\1      automountServiceAccountToken: false\n\2'
    content = re.sub(pattern1, replacement1, content)
    
    # Fix 2: Add automountServiceAccountToken to pods that don't have serviceAccountName
    # Pattern: spec: followed by securityContext (no serviceAccountName)
    pattern2 = r'(\n {4}spec:\n)( {6}securityContext:)'
    def replace_func(match):
        # Only add if not already present and no serviceAccountName before
        if 'automountServiceAccountToken' not in match.group(0):
            return f'{match.group(1)}      automountServiceAccountToken: false\n{match.group(2)}'
        return match.group(0)
    content = re.sub(pattern2, replace_func, content)
    
    # Fix 3: Add ephemeral-storage limits to resources sections
    # Find all resources blocks without ephemeral-storage
    def add_ephemeral_storage(match):
        resources_block = match.group(0)
        
        # Check if ephemeral-storage already exists
        if 'ephemeral-storage' in resources_block:
            return resources_block
        
        # Add to limits section
        limits_pattern = r'(limits:\n(?: +[a-z-]+: [^\n]+\n)+)'
        def add_to_limits(limits_match):
            limits_content = limits_match.group(0)
            # Insert before the last line (which should be memory or cpu)
            lines = limits_content.rstrip('\n').split('\n')
            indent = '              '  # Match typical indentation
            lines.append(f'{indent}ephemeral-storage: "2Gi"')
            return '\n'.join(lines) + '\n'
        
        resources_block = re.sub(limits_pattern, add_to_limits, resources_block)
        
        # Add to requests section
        requests_pattern = r'(requests:\n(?: +[a-z-]+: [^\n]+\n)+)'
        def add_to_requests(requests_match):
            requests_content = requests_match.group(0)
            lines = requests_content.rstrip('\n').split('\n')
            indent = '              '
            lines.append(f'{indent}ephemeral-storage: "1Gi"')
            return '\n'.join(lines) + '\n'
        
        resources_block = re.sub(requests_pattern, add_to_requests, resources_block)
        
        return resources_block
    
    # Match resources blocks
    resources_pattern = r'( {10}resources:\n(?: +[a-z-]+:\n(?: +[a-z-]+: [^\n]+\n)+)+)'
    content = re.sub(resources_pattern, add_ephemeral_storage, content)
    
    # Write back if changed
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"  ✅ Fixed {file_path}")
        return True
    else:
        print(f"  ℹ️  No changes needed for {file_path}")
        return False


def main():
    """Fix all deployment files."""
    base_path = Path(__file__).parent.parent
    
    # List of files to fix (from the security report)
    files_to_fix = [
        "kubernetes/deployments/code-review-ai.yaml",
        "kubernetes/deployments/three-version-service.yaml",
        "kubernetes/deployments/v1-deployment.yaml",
        "kubernetes/deployments/v2-deployment.yaml",
        "kubernetes/deployments/v3-deployment.yaml",
        "kubernetes/deployments/v3-services.yaml",
        "kubernetes/deployments/version-control-ai.yaml",
        "kubernetes/overlays/offline/local-models.yaml",
        "kubernetes/services/all-services.yaml",
        "kubernetes/services/auth-service.yaml",
        "kubernetes/workloads/spot-instances.yaml",
        "monitoring/observability/otel-collector-config.yaml",
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        full_path = base_path / file_path
        if full_path.exists():
            if fix_deployment_file(full_path):
                fixed_count += 1
        else:
            print(f"  ⚠️  File not found: {full_path}")
    
    print(f"\n✅ Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
