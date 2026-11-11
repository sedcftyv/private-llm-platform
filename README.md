# Private LLM Platform

基于 vLLM、Qwen3-4B、RAG、评测与监控的私有化大模型平台项目。

当前进度：Day 2 已完成——FastAPI 网关、健康检查以及普通/流式 Chat
Completions 转发均已验证。

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

## Day 2

安装网关开发环境：

```bash
uv sync --dev
```

先启动 Day 1 的 vLLM 服务，再启动网关：

```bash
./scripts/start_vllm_o0.sh
./scripts/start_gateway.sh
```

网关监听 `127.0.0.1:8080`，上游 vLLM 地址为 `127.0.0.1:8000`。

运行测试：

```bash
.venv/bin/python -m pytest -v
.venv/bin/python scripts/gateway_smoke_test.py
```

Day 2 实测结果见 [`artifacts/day2/README.md`](artifacts/day2/README.md)。
