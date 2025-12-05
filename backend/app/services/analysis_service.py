"""
Analysis Service / 分析服务

Business logic for code analysis operations.
代码分析操作的业务逻辑。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib
import random


class AnalysisService:
    """Service for code analysis operations / 代码分析操作服务"""
    
    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode
        self._analysis_cache: Dict[str, Any] = {}
    
    def generate_session_id(self, project_id: str) -> str:
        """Generate unique session ID / 生成唯一会话 ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"session_{project_id}_{timestamp}"
    
    def calculate_code_hash(self, code: str) -> str:
        """Calculate hash of code for caching / 计算代码哈希用于缓存"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    
    async def analyze_code(
        self,
        code: str,
        language: str = "typescript",
        model: str = "gpt-4-turbo",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze code and return issues.
        分析代码并返回问题。
        """
        code_hash = self.calculate_code_hash(code)
        
        # Check cache first
        if code_hash in self._analysis_cache:
            cached = self._analysis_cache[code_hash]
            cached["from_cache"] = True
            return cached
        
        if self.mock_mode:
            result = self._mock_analysis(code, language, model)
        else:
            result = await self._real_analysis(code, language, model, context)
        
        # Cache result
        self._analysis_cache[code_hash] = result
        return result
    
    def _mock_analysis(
        self,
        code: str,
        language: str,
        model: str
    ) -> Dict[str, Any]:
        """Generate mock analysis results / 生成模拟分析结果"""
        lines = code.split('\n')
        num_lines = len(lines)
        
        # Generate realistic mock issues based on code patterns
        issues = []
        
        # Check for common issues
        for i, line in enumerate(lines, 1):
            if 'var ' in line:
                issues.append({
                    "id": f"issue_{len(issues)+1}",
                    "type": "warning",
                    "severity": "medium",
                    "message": "Consider using 'let' or 'const' instead of 'var'",
                    "line": i,
                    "column": line.find('var') + 1,
                    "suggestion": "Replace 'var' with 'const' or 'let'"
                })
            
            if 'console.log' in line:
                issues.append({
                    "id": f"issue_{len(issues)+1}",
                    "type": "info",
                    "severity": "low",
                    "message": "Console statement should be removed in production",
                    "line": i,
                    "column": line.find('console.log') + 1,
                    "suggestion": "Remove or replace with proper logging"
                })
            
            if len(line) > 120:
                issues.append({
                    "id": f"issue_{len(issues)+1}",
                    "type": "suggestion",
                    "severity": "low",
                    "message": "Line exceeds recommended length of 120 characters",
                    "line": i,
                    "column": 121,
                    "suggestion": "Break line into multiple lines"
                })
        
        # Calculate metrics
        complexity = min(50, num_lines // 5 + random.randint(5, 15))
        maintainability = max(50, 100 - complexity - len(issues) * 2)
        
        return {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "completed",
            "model": model,
            "language": language,
            "issues": issues,
            "metrics": {
                "complexity": complexity,
                "maintainability": maintainability,
                "test_coverage": 0,
                "lines_analyzed": num_lines,
                "issues_found": len(issues)
            },
            "summary": f"Analysis completed. Found {len(issues)} issues in {num_lines} lines.",
            "processing_time_ms": random.randint(500, 2000),
            "from_cache": False
        }
    
    async def _real_analysis(
        self,
        code: str,
        language: str,
        model: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Perform real AI analysis.
        执行真实的 AI 分析。
        
        TODO: Integrate with OpenAI/Anthropic APIs
        """
        raise NotImplementedError("Real AI analysis requires API integration")
    
    def get_cached_analysis(self, code_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result / 获取缓存的分析结果"""
        return self._analysis_cache.get(code_hash)
    
    def clear_cache(self):
        """Clear analysis cache / 清除分析缓存"""
        self._analysis_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics / 获取缓存统计"""
        return {
            "cached_analyses": len(self._analysis_cache),
            "total_issues_cached": sum(
                len(a.get("issues", []))
                for a in self._analysis_cache.values()
            )
        }
