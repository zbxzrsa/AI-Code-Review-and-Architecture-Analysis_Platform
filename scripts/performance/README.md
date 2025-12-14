# 性能测试说明

## 目标
验证优化后的性能提升≥20%

## 测试工具

### 1. k6 负载测试
用于测试API服务的性能。

**安装**:
```bash
# macOS
brew install k6

# Linux
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**运行**:
```bash
k6 run scripts/performance/k6-load-test.js
```

**自定义配置**:
```bash
BASE_URL=http://localhost:3000 k6 run scripts/performance/k6-load-test.js
```

### 2. Python基准测试
用于测试项目扫描和补丁生成的性能。

**运行**:
```bash
# 基本扫描性能
python scripts/performance/benchmark_project_scan.py

# 指定项目路径
python scripts/performance/benchmark_project_scan.py /path/to/project
```

**输出**:
- 平均耗时
- 中位数耗时
- 吞吐量（文件/秒）
- 并发效率

### 3. 性能对比
对比优化前后的性能数据。

**使用**:
```bash
# 1. 运行优化前的基准测试，保存结果
python scripts/performance/benchmark_project_scan.py > baseline.json

# 2. 运行优化后的测试，保存结果
python scripts/performance/benchmark_project_scan.py > current.json

# 3. 对比结果
python scripts/performance/compare_performance.py baseline.json current.json
```

## 性能指标

### API服务
- **响应时间**: P95 < 500ms, P99 < 1000ms
- **错误率**: < 1%
- **API网关**: P95 < 200ms
- **发布闸门**: P95 < 100ms

### 项目扫描
- **扫描速度**: 提升≥20%
- **吞吐量**: 提升≥20%
- **并发效率**: 提升≥20%

## 测试场景

### 1. 基本扫描性能
- 单次扫描耗时
- 文件处理速度
- 问题识别速度

### 2. 并发扫描性能
- 并发扫描效率
- 资源利用率
- 吞吐量提升

### 3. 补丁生成性能
- 补丁生成速度
- AI模型调用性能
- 规则改进性能

### 4. API服务性能
- 路由决策性能
- 发布闸门决策性能
- 沙盒编排性能
- 技术监测性能

## 性能优化验证

### 优化前基准
（需要运行基准测试获取）

### 优化后结果
（运行当前测试获取）

### 提升验证
- ✅ 扫描性能提升: XX%
- ✅ 吞吐量提升: XX%
- ✅ 并发效率提升: XX%
- ✅ API响应时间: 符合要求

## 持续监控

建议在生产环境中持续监控：
- 平均响应时间
- P95/P99响应时间
- 错误率
- 吞吐量
- 资源利用率

