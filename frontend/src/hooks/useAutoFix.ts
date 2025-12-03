/**
 * Auto-Fix Hook
 * 
 * Hook for interacting with the auto-fix cycle API:
 * - Get cycle status
 * - List vulnerabilities
 * - Approve/reject fixes
 * - Start/stop cycle
 */

import { useState, useCallback, useEffect } from 'react';
import { message } from 'antd';

export interface Vulnerability {
  vuln_id: string;
  pattern_id: string;
  file_path: string;
  line_number: number;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category: string;
  description: string;
  confidence: number;
  detected_at: string;
}

export interface Fix {
  fix_id: string;
  vuln_id: string;
  file_path: string;
  original_code: string;
  fixed_code: string;
  status: 'pending' | 'approved' | 'applied' | 'verified' | 'rejected' | 'rolled_back';
  confidence: number;
  applied_at?: string;
  verified_at?: string;
}

export interface CycleStatus {
  running: boolean;
  phase: string;
  pending_fixes: number;
  metrics: {
    cycles_completed: number;
    vulnerabilities_detected: number;
    fixes_applied: number;
    fixes_verified: number;
    fixes_failed: number;
    fixes_rolled_back: number;
  };
  last_cycle_at?: string;
}

interface UseAutoFixReturn {
  status: CycleStatus | null;
  vulnerabilities: Vulnerability[];
  fixes: Fix[];
  pendingFixes: Fix[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  startCycle: () => Promise<void>;
  stopCycle: () => Promise<void>;
  approveFix: (fixId: string) => Promise<void>;
  rejectFix: (fixId: string) => Promise<void>;
  applyFix: (fixId: string) => Promise<void>;
  rollbackFix: (fixId: string) => Promise<void>;
}

export function useAutoFix(): UseAutoFixReturn {
  const [status, setStatus] = useState<CycleStatus | null>(null);
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([]);
  const [fixes, setFixes] = useState<Fix[]>([]);
  const [pendingFixes, setPendingFixes] = useState<Fix[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch from mock API or real API
      const [statusRes, vulnsRes, fixesRes, pendingRes] = await Promise.all([
        fetch('/api/auto-fix/status').then((r) => r.json()),
        fetch('/api/auto-fix/vulnerabilities').then((r) => r.json()),
        fetch('/api/auto-fix/fixes').then((r) => r.json()),
        fetch('/api/auto-fix/fixes/pending').then((r) => r.json()),
      ]);

      setStatus(statusRes);
      setVulnerabilities(Array.isArray(vulnsRes) ? vulnsRes : []);
      setFixes(Array.isArray(fixesRes) ? fixesRes : []);
      setPendingFixes(Array.isArray(pendingRes) ? pendingRes : []);
    } catch (err) {
      console.error('Failed to fetch auto-fix data:', err);
      setError('Failed to load auto-fix data');
      
      // Set mock data as fallback
      setStatus({
        running: true,
        phase: 'idle',
        pending_fixes: 2,
        metrics: {
          cycles_completed: 15,
          vulnerabilities_detected: 25,
          fixes_applied: 15,
          fixes_verified: 12,
          fixes_failed: 3,
          fixes_rolled_back: 2,
        },
        last_cycle_at: new Date().toISOString(),
      });
      setVulnerabilities([
        {
          vuln_id: 'vuln-001',
          pattern_id: 'SEC-001',
          file_path: 'backend/shared/security/auth.py',
          line_number: 19,
          severity: 'critical',
          category: 'security',
          description: 'Hardcoded secret key',
          confidence: 0.95,
          detected_at: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    
    // Poll for updates every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const startCycle = useCallback(async () => {
    try {
      const res = await fetch('/api/auto-fix/start', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        message.success('Auto-fix cycle started');
        await fetchData();
      } else {
        message.warning(data.message || 'Failed to start cycle');
      }
    } catch (err) {
      message.error('Failed to start cycle');
    }
  }, [fetchData]);

  const stopCycle = useCallback(async () => {
    try {
      const res = await fetch('/api/auto-fix/stop', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        message.success('Auto-fix cycle stopped');
        await fetchData();
      } else {
        message.warning(data.message || 'Failed to stop cycle');
      }
    } catch (err) {
      message.error('Failed to stop cycle');
    }
  }, [fetchData]);

  const approveFix = useCallback(async (fixId: string) => {
    try {
      const res = await fetch(`/api/auto-fix/fixes/${fixId}/approve`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        message.success('Fix approved');
        await fetchData();
      }
    } catch (err) {
      message.error('Failed to approve fix');
    }
  }, [fetchData]);

  const rejectFix = useCallback(async (fixId: string) => {
    try {
      const res = await fetch(`/api/auto-fix/fixes/${fixId}/reject`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        message.success('Fix rejected');
        await fetchData();
      }
    } catch (err) {
      message.error('Failed to reject fix');
    }
  }, [fetchData]);

  const applyFix = useCallback(async (fixId: string) => {
    try {
      const res = await fetch(`/api/auto-fix/fixes/${fixId}/apply`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        message.success('Fix applied');
        await fetchData();
      }
    } catch (err) {
      message.error('Failed to apply fix');
    }
  }, [fetchData]);

  const rollbackFix = useCallback(async (fixId: string) => {
    try {
      const res = await fetch(`/api/auto-fix/fixes/${fixId}/rollback`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        message.success('Fix rolled back');
        await fetchData();
      }
    } catch (err) {
      message.error('Failed to rollback fix');
    }
  }, [fetchData]);

  return {
    status,
    vulnerabilities,
    fixes,
    pendingFixes,
    loading,
    error,
    refresh: fetchData,
    startCycle,
    stopCycle,
    approveFix,
    rejectFix,
    applyFix,
    rollbackFix,
  };
}

export default useAutoFix;
