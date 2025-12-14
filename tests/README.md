# 测试说明

## 测试覆盖率目标：98%

## 运行测试

### Python测试
```bash
# 运行所有测试
pytest

# 运行单元测试
pytest -m unit

# 运行集成测试（需要启动docker-compose.test.yml）
pytest -m integration

# 查看覆盖率
pytest --cov=ai_core --cov=backend --cov-report=html
```

### TypeScript测试
```bash
cd services
npm test
```

## 集成测试基础设施

启动测试基础设施：
```bash
docker-compose -f docker-compose.test.yml up -d
```

停止：
```bash
docker-compose -f docker-compose.test.yml down
```

## 测试分类

- `@pytest.mark.unit`: 单元测试（快速，无外部依赖）
- `@pytest.mark.integration`: 集成测试（需要服务）
- `@pytest.mark.e2e`: 端到端测试
- `@pytest.mark.critical`: 关键路径测试

