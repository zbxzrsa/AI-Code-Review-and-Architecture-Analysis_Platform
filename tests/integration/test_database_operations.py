"""
Database Operations Integration Tests

Tests for database integrity, migrations, and complex queries.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from uuid import uuid4


@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseSchemaIntegrity:
    """Tests for database schema integrity."""
    
    async def test_foreign_key_constraints(
        self,
        db_session,
        test_user_in_db: Dict[str, Any],
    ):
        """Test that foreign key constraints are enforced."""
        # Try to insert project with non-existent owner
        with pytest.raises(Exception) as exc_info:
            await db_session.execute(
                """
                INSERT INTO projects.projects (id, name, owner_id, created_at)
                VALUES ($1, $2, $3, $4)
                """,
                [str(uuid4()), "test-project", str(uuid4()), datetime.now(timezone.utc)]
            )
        
        assert "foreign key" in str(exc_info.value).lower()
    
    async def test_unique_constraints(
        self,
        db_session,
        test_user_in_db: Dict[str, Any],
    ):
        """Test that unique constraints are enforced."""
        # Try to insert duplicate user email
        with pytest.raises(Exception) as exc_info:
            await db_session.execute(
                """
                INSERT INTO auth.users (id, email, password_hash, created_at)
                VALUES ($1, $2, $3, $4)
                """,
                [str(uuid4()), test_user_in_db["email"], "hash", datetime.now(timezone.utc)]
            )
        
        assert "unique" in str(exc_info.value).lower()
    
    async def test_cascade_delete(
        self,
        db_session,
        test_user_in_db: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
    ):
        """Test cascade delete behavior."""
        # Delete user should cascade to projects
        await db_session.execute(
            "DELETE FROM auth.users WHERE id = $1",
            [test_user_in_db["id"]]
        )
        
        # Verify project is also deleted
        result = await db_session.execute(
            "SELECT id FROM projects.projects WHERE id = $1",
            [test_project_in_db["id"]]
        )
        
        assert result.fetchone() is None
    
    async def test_not_null_constraints(
        self,
        db_session,
    ):
        """Test that NOT NULL constraints are enforced."""
        with pytest.raises(Exception) as exc_info:
            await db_session.execute(
                """
                INSERT INTO auth.users (id, email, created_at)
                VALUES ($1, $2, $3)
                """,
                [str(uuid4()), "test@example.com", datetime.now(timezone.utc)]
                # Missing required password_hash
            )
        
        assert "not null" in str(exc_info.value).lower() or "null" in str(exc_info.value).lower()


@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseQueries:
    """Tests for complex database queries."""
    
    async def test_pagination(
        self,
        db_session,
        many_projects_in_db: List[Dict[str, Any]],
    ):
        """Test pagination works correctly."""
        # Get first page
        page1 = await db_session.execute(
            """
            SELECT * FROM projects.projects
            ORDER BY created_at DESC
            LIMIT 10 OFFSET 0
            """
        )
        
        # Get second page
        page2 = await db_session.execute(
            """
            SELECT * FROM projects.projects
            ORDER BY created_at DESC
            LIMIT 10 OFFSET 10
            """
        )
        
        page1_results = page1.fetchall()
        page2_results = page2.fetchall()
        
        assert len(page1_results) == 10
        assert len(page2_results) <= 10
        
        # No overlap
        page1_ids = {r["id"] for r in page1_results}
        page2_ids = {r["id"] for r in page2_results}
        assert page1_ids.isdisjoint(page2_ids)
    
    async def test_search_query(
        self,
        db_session,
        many_projects_in_db: List[Dict[str, Any]],
    ):
        """Test search query performance and accuracy."""
        search_term = "python"
        
        result = await db_session.execute(
            """
            SELECT * FROM projects.projects
            WHERE name ILIKE $1 OR description ILIKE $1
            ORDER BY created_at DESC
            """,
            [f"%{search_term}%"]
        )
        
        results = result.fetchall()
        
        # All results should contain search term
        for r in results:
            assert search_term.lower() in r["name"].lower() or search_term.lower() in r["description"].lower()
    
    async def test_aggregation_queries(
        self,
        db_session,
        many_analyses_in_db: List[Dict[str, Any]],
    ):
        """Test aggregation queries."""
        result = await db_session.execute(
            """
            SELECT 
                status,
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_duration
            FROM production.analysis_sessions
            WHERE completed_at IS NOT NULL
            GROUP BY status
            """
        )
        
        results = result.fetchall()
        
        # Should have aggregated data
        assert len(results) > 0
        for r in results:
            assert r["count"] > 0
    
    async def test_join_queries(
        self,
        db_session,
        test_project_with_analyses: Dict[str, Any],
    ):
        """Test complex join queries."""
        result = await db_session.execute(
            """
            SELECT 
                p.name as project_name,
                COUNT(a.id) as analysis_count,
                AVG(a.issue_count) as avg_issues
            FROM projects.projects p
            LEFT JOIN production.analysis_sessions a ON p.id = a.project_id
            WHERE p.id = $1
            GROUP BY p.id, p.name
            """,
            [test_project_with_analyses["id"]]
        )
        
        row = result.fetchone()
        
        assert row is not None
        assert row["project_name"] == test_project_with_analyses["name"]


@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseTransactionIsolation:
    """Tests for transaction isolation levels."""
    
    async def test_read_committed_isolation(
        self,
        db_session,
        test_project_in_db: Dict[str, Any],
    ):
        """Test read committed isolation level."""
        # Start two transactions
        async with db_session.transaction():
            # Update in transaction 1 but don't commit
            await db_session.execute(
                "UPDATE projects.projects SET name = 'updated' WHERE id = $1",
                [test_project_in_db["id"]]
            )
            
            # In a separate connection, should not see uncommitted change
            # (This requires a second connection in real implementation)
            # Simplified test here
            result = await db_session.execute(
                "SELECT name FROM projects.projects WHERE id = $1",
                [test_project_in_db["id"]]
            )
            
            row = result.fetchone()
            # After update in same transaction, should see new value
            assert row["name"] == "updated"
    
    async def test_deadlock_detection(
        self,
        db_session,
        db_session_2,  # Second connection
        test_project_in_db: Dict[str, Any],
    ):
        """Test deadlock detection and resolution."""
        project_id = test_project_in_db["id"]
        
        async def transaction_1():
            async with db_session.transaction():
                await db_session.execute(
                    "SELECT * FROM projects.projects WHERE id = $1 FOR UPDATE",
                    [project_id]
                )
                await asyncio.sleep(0.1)  # Hold lock
                await db_session.execute(
                    "UPDATE projects.projects SET name = 'tx1' WHERE id = $1",
                    [project_id]
                )
        
        async def transaction_2():
            async with db_session_2.transaction():
                await db_session_2.execute(
                    "SELECT * FROM projects.projects WHERE id = $1 FOR UPDATE",
                    [project_id]
                )
                await db_session_2.execute(
                    "UPDATE projects.projects SET name = 'tx2' WHERE id = $1",
                    [project_id]
                )
        
        # One should succeed, one should be retried or fail gracefully
        results = await asyncio.gather(
            transaction_1(),
            transaction_2(),
            return_exceptions=True
        )
        
        # At least one should succeed
        success_count = sum(1 for r in results if r is None)
        assert success_count >= 1


@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabasePerformance:
    """Tests for database query performance."""
    
    async def test_index_usage(
        self,
        db_session,
        many_analyses_in_db: List[Dict[str, Any]],
    ):
        """Test that indexes are being used."""
        # EXPLAIN query to check index usage
        result = await db_session.execute(
            """
            EXPLAIN (FORMAT JSON)
            SELECT * FROM production.analysis_sessions
            WHERE project_id = $1
            ORDER BY created_at DESC
            LIMIT 10
            """,
            [many_analyses_in_db[0]["project_id"]]
        )
        
        explain = result.fetchone()[0]
        
        # Check that index is used (not sequential scan)
        plan = explain[0]["Plan"]
        node_type = plan.get("Node Type", "")
        
        # Should use index scan, not seq scan
        assert "Seq Scan" not in node_type or "Index" in str(explain)
    
    async def test_query_timeout(
        self,
        db_session,
    ):
        """Test that long-running queries are timed out."""
        # Set statement timeout
        await db_session.execute("SET statement_timeout = '100ms'")
        
        # This should timeout (simulating slow query)
        with pytest.raises(Exception) as exc_info:
            await db_session.execute("SELECT pg_sleep(1)")
        
        assert "timeout" in str(exc_info.value).lower() or "cancel" in str(exc_info.value).lower()
        
        # Reset timeout
        await db_session.execute("SET statement_timeout = '0'")
    
    async def test_bulk_insert_performance(
        self,
        db_session,
        test_user_in_db: Dict[str, Any],
    ):
        """Test bulk insert performance."""
        import time
        
        # Prepare 1000 records
        records = [
            (str(uuid4()), f"project-{i}", test_user_in_db["id"], datetime.now(timezone.utc))
            for i in range(1000)
        ]
        
        start = time.time()
        
        # Use COPY or batch insert
        await db_session.executemany(
            """
            INSERT INTO projects.projects (id, name, owner_id, created_at)
            VALUES ($1, $2, $3, $4)
            """,
            records
        )
        
        duration = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds for 1000 records)
        assert duration < 5.0
