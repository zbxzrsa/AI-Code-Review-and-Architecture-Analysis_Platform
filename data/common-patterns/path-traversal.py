# Common Path Traversal Pattern - For Cache Warming
# This pattern should be detected as a security vulnerability

import os
from pathlib import Path

# VULNERABLE: Direct path concatenation with user input
def read_file_unsafe(filename):
    """DANGEROUS: Path traversal vulnerability"""
    # User can pass "../../../etc/passwd"
    filepath = "/var/uploads/" + filename
    with open(filepath, 'r') as f:
        return f.read()


# VULNERABLE: Using os.path.join incorrectly
def get_document_unsafe(doc_name):
    """DANGEROUS: os.path.join doesn't prevent traversal"""
    # If doc_name is "/etc/passwd", it becomes the full path
    base_dir = "/app/documents"
    doc_path = os.path.join(base_dir, doc_name)
    return open(doc_path).read()


# VULNERABLE: No validation of file path
def download_file_unsafe(user_path):
    """DANGEROUS: No path validation"""
    download_dir = Path("/downloads")
    file_path = download_dir / user_path
    # User can pass "../../secrets/api_keys.json"
    return file_path.read_bytes()


# SAFE: Validate path is within allowed directory
def read_file_safe(filename):
    """SAFE: Validates path stays within base directory"""
    base_dir = Path("/var/uploads").resolve()
    
    # Remove any path traversal attempts
    safe_name = os.path.basename(filename)
    
    filepath = base_dir / safe_name
    
    # Verify the resolved path is still within base_dir
    if not str(filepath.resolve()).startswith(str(base_dir)):
        raise ValueError("Invalid file path")
    
    with open(filepath, 'r') as f:
        return f.read()


# SAFE: Using realpath validation
def get_document_safe(doc_name):
    """SAFE: Uses realpath to prevent traversal"""
    base_dir = os.path.realpath("/app/documents")
    
    # Construct and resolve the full path
    requested_path = os.path.realpath(
        os.path.join(base_dir, doc_name)
    )
    
    # Verify it's still within base directory
    if not requested_path.startswith(base_dir + os.sep):
        raise PermissionError("Access denied: path traversal attempt")
    
    with open(requested_path, 'r') as f:
        return f.read()


# SAFE: Using Path.resolve() with validation
def download_file_safe(user_path):
    """SAFE: Properly validates path"""
    download_dir = Path("/downloads").resolve()
    
    # Construct the path
    file_path = (download_dir / user_path).resolve()
    
    # Check if it's within the allowed directory
    try:
        file_path.relative_to(download_dir)
    except ValueError:
        raise PermissionError("Path traversal detected")
    
    # Also check file exists
    if not file_path.is_file():
        raise FileNotFoundError("File not found")
    
    return file_path.read_bytes()
