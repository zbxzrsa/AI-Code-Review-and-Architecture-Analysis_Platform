"""
Test Fixtures: Sample Code Snippets

Provides sample code snippets for testing the code review AI.
Each snippet is categorized by vulnerability type and expected result.
"""

# =============================================================================
# Security Vulnerabilities
# =============================================================================

SQL_INJECTION_VULNERABLE = """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
"""

SQL_INJECTION_SAFE = """
def get_user(username):
    cursor.execute(
        "SELECT * FROM users WHERE username = %s",
        (username,)
    )
    return cursor.fetchone()
"""

XSS_VULNERABLE = """
function displayUserInput(input) {
    document.getElementById('output').innerHTML = input;
}
"""

XSS_SAFE = """
function displayUserInput(input) {
    document.getElementById('output').textContent = input;
}
"""

COMMAND_INJECTION_VULNERABLE = """
import os

def run_command(user_input):
    os.system(f"echo {user_input}")
"""

COMMAND_INJECTION_SAFE = """
import subprocess

def run_command(user_input):
    subprocess.run(["echo", user_input], check=True)
"""

HARDCODED_SECRET_VULNERABLE = """
API_KEY = "sk-1234567890abcdef"
DATABASE_URL = "postgresql://admin:password123@localhost/db"
"""

HARDCODED_SECRET_SAFE = """
import os

API_KEY = os.environ.get("API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
"""

PATH_TRAVERSAL_VULNERABLE = """
def read_file(filename):
    path = f"/uploads/{filename}"
    with open(path, 'r') as f:
        return f.read()
"""

PATH_TRAVERSAL_SAFE = """
import os
from pathlib import Path

def read_file(filename):
    base_dir = Path("/uploads").resolve()
    safe_path = (base_dir / os.path.basename(filename)).resolve()
    
    if not str(safe_path).startswith(str(base_dir)):
        raise ValueError("Invalid path")
    
    with open(safe_path, 'r') as f:
        return f.read()
"""

# =============================================================================
# Code Quality Issues
# =============================================================================

UNUSED_VARIABLE = """
def calculate(x, y, z):
    result = x + y
    unused = z * 2
    return result
"""

EMPTY_EXCEPT = """
try:
    risky_operation()
except:
    pass
"""

DUPLICATE_CODE = """
def process_user(user):
    if user.age > 18:
        print("Adult")
        user.category = "adult"
        log_user(user)
    else:
        print("Minor")
        user.category = "minor"
        log_user(user)
"""

COMPLEX_FUNCTION = """
def do_everything(data, config, options, flags, mode, level):
    if mode == 'a':
        if level > 5:
            if flags.get('x'):
                if options.get('y'):
                    result = process_a(data)
                else:
                    result = process_b(data)
            else:
                if config.get('z'):
                    result = process_c(data)
                else:
                    result = process_d(data)
        else:
            result = process_e(data)
    else:
        result = process_f(data)
    return result
"""

# =============================================================================
# Performance Issues
# =============================================================================

N_PLUS_ONE_QUERY = """
def get_orders_with_items():
    orders = Order.query.all()
    for order in orders:
        items = OrderItem.query.filter_by(order_id=order.id).all()
        order.items = items
    return orders
"""

N_PLUS_ONE_FIXED = """
def get_orders_with_items():
    return Order.query.options(
        joinedload(Order.items)
    ).all()
"""

INEFFICIENT_STRING_CONCAT = """
def build_message(items):
    result = ""
    for item in items:
        result = result + str(item) + ", "
    return result
"""

INEFFICIENT_STRING_FIXED = """
def build_message(items):
    return ", ".join(str(item) for item in items)
"""

# =============================================================================
# Clean Code (No Issues Expected)
# =============================================================================

CLEAN_CODE_PYTHON = """
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

def get_active_users(users: List[User]) -> List[User]:
    \"\"\"Filter and return only active users.\"\"\"
    return [user for user in users if is_active(user)]

def is_active(user: User) -> bool:
    \"\"\"Check if a user is active.\"\"\"
    return user.id > 0 and user.email is not None
"""

CLEAN_CODE_JAVASCRIPT = """
/**
 * Calculate the total price of items in cart
 * @param {Array} items - Cart items
 * @returns {number} Total price
 */
function calculateTotal(items) {
    return items.reduce((total, item) => {
        const price = item.price * item.quantity;
        const discount = item.discount || 0;
        return total + (price * (1 - discount));
    }, 0);
}

export { calculateTotal };
"""

CLEAN_CODE_TYPESCRIPT = """
interface User {
    id: number;
    name: string;
    email: string;
}

class UserService {
    private users: Map<number, User> = new Map();

    async getUser(id: number): Promise<User | undefined> {
        return this.users.get(id);
    }

    async createUser(user: Omit<User, 'id'>): Promise<User> {
        const id = this.generateId();
        const newUser = { id, ...user };
        this.users.set(id, newUser);
        return newUser;
    }

    private generateId(): number {
        return Date.now();
    }
}

export { User, UserService };
"""

# =============================================================================
# Multi-language Samples
# =============================================================================

SAMPLES_BY_LANGUAGE = {
    "python": {
        "vulnerable": [
            SQL_INJECTION_VULNERABLE,
            COMMAND_INJECTION_VULNERABLE,
            HARDCODED_SECRET_VULNERABLE,
            PATH_TRAVERSAL_VULNERABLE,
        ],
        "safe": [
            SQL_INJECTION_SAFE,
            COMMAND_INJECTION_SAFE,
            HARDCODED_SECRET_SAFE,
            PATH_TRAVERSAL_SAFE,
            CLEAN_CODE_PYTHON,
        ],
        "quality": [
            UNUSED_VARIABLE,
            EMPTY_EXCEPT,
            DUPLICATE_CODE,
            COMPLEX_FUNCTION,
        ],
        "performance": [
            N_PLUS_ONE_QUERY,
            INEFFICIENT_STRING_CONCAT,
        ],
    },
    "javascript": {
        "vulnerable": [
            XSS_VULNERABLE,
        ],
        "safe": [
            XSS_SAFE,
            CLEAN_CODE_JAVASCRIPT,
        ],
    },
    "typescript": {
        "safe": [
            CLEAN_CODE_TYPESCRIPT,
        ],
    },
}


def get_sample(language: str, category: str, index: int = 0) -> str:
    """Get a sample code snippet."""
    samples = SAMPLES_BY_LANGUAGE.get(language, {}).get(category, [])
    if index < len(samples):
        return samples[index]
    return ""


def get_all_vulnerable_samples() -> list:
    """Get all vulnerable code samples."""
    samples = []
    for lang, categories in SAMPLES_BY_LANGUAGE.items():
        for code in categories.get("vulnerable", []):
            samples.append({"language": lang, "code": code})
    return samples


def get_all_safe_samples() -> list:
    """Get all safe code samples."""
    samples = []
    for lang, categories in SAMPLES_BY_LANGUAGE.items():
        for code in categories.get("safe", []):
            samples.append({"language": lang, "code": code})
    return samples
