# Common SQL Injection Pattern - For Cache Warming
# This pattern should be detected as a security vulnerability

import sqlite3

def get_user_unsafe(username):
    """VULNERABLE: SQL Injection via string concatenation"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # DANGEROUS: Direct string interpolation
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    
    return cursor.fetchone()


def get_user_safe(username):
    """SAFE: Using parameterized queries"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # SAFE: Parameterized query
    query = "SELECT * FROM users WHERE username = ?"
    cursor.execute(query, (username,))
    
    return cursor.fetchone()
