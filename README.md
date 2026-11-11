# Private LLM Platform

基于 vLLM、Qwen3-4B、RAG、评测与监控的私有化大模型平台项目。

当前进度：Day 1 已完成——环境固定、Qwen3-4B 服务和生成请求均已验证。

完整计划见
[`vllm_serving_lab_project_plan.md`](vllm_serving_lab_project_plan.md)。

## Day 1

### 1. 采集环境信息

```bash
./scripts/capture_environment.sh
```

结果保存到 `artifacts/day1/environment.txt`。

### 2. 启动 vLLM

```bash
./scripts/start_vllm_o0.sh
```

服务默认监听 `127.0.0.1:8000`，使用 Qwen3-4B、FP16、4096 Token
上下文和 O0 eager 模式。

### 3. 运行烟雾测试

在另一个终端执行：

```bash
/home/aida/vllm/.venv/bin/python scripts/smoke_test.py
```

成功标准：

- `/health` 返回 HTTP 200；
- `/v1/models` 包含 `qwen3-4b`；
- `/v1/chat/completions` 返回非空文本；
- 服务日志中没有 OOM、dtype 或模型架构错误。

Day 1 实测结果见 [`artifacts/day1/README.md`](artifacts/day1/README.md)。
