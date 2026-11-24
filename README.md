# Private LLM Platform

基于 vLLM、Qwen3-4B、RAG、评测与监控的私有化大模型平台项目。

当前进度：Day 3 已完成——FastAPI 网关支持逻辑模型注册、fast/reasoning
能力路由，以及普通/流式 Chat Completions 转发。

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

## Day 3

逻辑模型配置位于
[`models.toml`](src/private_llm_platform/gateway/models.toml)：

| 客户端模型名 | vLLM 模型名 | Thinking | 超时 |
| --- | --- | --- | --- |
| `qwen3-fast` | `qwen3-4b` | 关闭 | 60 秒 |
| `qwen3-reasoning` | `qwen3-4b` | 开启 | 120 秒 |

网关会强制应用路由配置，因此客户端不能通过
`chat_template_kwargs.enable_thinking` 改变逻辑模型的语义。可通过以下接口查看逻辑模型：

```bash
curl http://127.0.0.1:8080/v1/models
```

发送普通问答请求：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-fast",
    "messages": [{"role": "user", "content": "解释模型网关"}]
  }'
```

将 `model` 改为 `qwen3-reasoning` 即可启用思考模式。响应头包含
`x-model-route`、`x-upstream-model` 和 `x-thinking-enabled`，用于确认实际路由。

运行 Day 3 真实模型验收：

```bash
.venv/bin/python scripts/gateway_smoke_test.py
```

也可以通过 `LLM_MODEL_REGISTRY_PATH` 指向自定义 TOML 注册表。Day 3 实测结果见
[`artifacts/day3/README.md`](artifacts/day3/README.md)。
