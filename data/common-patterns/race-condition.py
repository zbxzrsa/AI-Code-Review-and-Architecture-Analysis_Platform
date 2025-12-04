# Common Race Condition Pattern - For Cache Warming
# This pattern should be detected as a security/reliability vulnerability

import threading
import time

# Global shared state (vulnerable)
balance = 1000


# VULNERABLE: Race condition in balance update
def withdraw_unsafe(amount):
    """DANGEROUS: Non-atomic read-modify-write"""
    global balance
    
    # Time-of-check
    if balance >= amount:
        # Race window - another thread could modify balance here
        time.sleep(0.001)  # Simulates processing delay
        
        # Time-of-use
        balance = balance - amount  # Non-atomic operation
        return True
    return False


# VULNERABLE: Check-then-act pattern
def transfer_unsafe(from_account, to_account, amount):
    """DANGEROUS: Check and action not atomic"""
    # Check
    if from_account.balance >= amount:
        # Gap where race can occur
        
        # Act
        from_account.balance -= amount
        to_account.balance += amount
        return True
    return False


# VULNERABLE: Singleton without proper locking
class SingletonUnsafe:
    """DANGEROUS: Double-checked locking without proper synchronization"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            # Race condition - multiple threads could create instances
            cls._instance = super().__new__(cls)
        return cls._instance


# VULNERABLE: File access race condition (TOCTOU)
import os

def process_file_unsafe(filepath):
    """DANGEROUS: Time-of-check to time-of-use vulnerability"""
    # Check if file exists
    if os.path.exists(filepath):
        # Race window - file could be modified/deleted here
        time.sleep(0.001)
        
        # Use the file
        with open(filepath, 'r') as f:
            return f.read()
    return None


# ============================================================
# SAFE ALTERNATIVES
# ============================================================

import threading
from contextlib import contextmanager


# SAFE: Using a lock for atomic operations
class BankAccount:
    """SAFE: Thread-safe balance operations"""
    
    def __init__(self, initial_balance=0):
        self._balance = initial_balance
        self._lock = threading.Lock()
    
    @property
    def balance(self):
        with self._lock:
            return self._balance
    
    def withdraw(self, amount):
        """SAFE: Atomic withdraw with lock"""
        with self._lock:
            if self._balance >= amount:
                self._balance -= amount
                return True
            return False
    
    def deposit(self, amount):
        """SAFE: Atomic deposit with lock"""
        with self._lock:
            self._balance += amount


# SAFE: Atomic transfer with lock ordering to prevent deadlock
def transfer_safe(from_account, to_account, amount):
    """SAFE: Atomic transfer with proper lock ordering"""
    # Always acquire locks in consistent order to prevent deadlock
    first, second = sorted([from_account, to_account], key=id)
    
    with first._lock:
        with second._lock:
            if from_account._balance >= amount:
                from_account._balance -= amount
                to_account._balance += amount
                return True
            return False


# SAFE: Thread-safe singleton
class SingletonSafe:
    """SAFE: Thread-safe singleton with lock"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance


# SAFE: Using atomic file operations
import tempfile
import shutil

def process_file_safe(filepath):
    """SAFE: Atomic file operations"""
    try:
        # Open file directly - let the OS handle race conditions
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None


def write_file_atomic(filepath, content):
    """SAFE: Atomic file write using temp file + rename"""
    dir_name = os.path.dirname(filepath)
    
    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=dir_name,
        delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # Atomic rename (on POSIX systems)
    shutil.move(tmp_path, filepath)


# SAFE: Using database transactions
from contextlib import contextmanager

@contextmanager
def database_transaction(connection):
    """SAFE: Database transaction for atomicity"""
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def transfer_with_db(connection, from_id, to_id, amount):
    """SAFE: Database transaction ensures atomicity"""
    with database_transaction(connection) as cursor:
        # SELECT FOR UPDATE locks the rows
        cursor.execute(
            "SELECT balance FROM accounts WHERE id = %s FOR UPDATE",
            (from_id,)
        )
        from_balance = cursor.fetchone()[0]
        
        if from_balance >= amount:
            cursor.execute(
                "UPDATE accounts SET balance = balance - %s WHERE id = %s",
                (amount, from_id)
            )
            cursor.execute(
                "UPDATE accounts SET balance = balance + %s WHERE id = %s",
                (amount, to_id)
            )
            return True
        return False
