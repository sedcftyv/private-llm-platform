# Day 2 验收报告

日期：2026-11-12（Asia/Shanghai）

## 结论

FastAPI 模型网关已完成最小实现，并通过 mock 单元测试和真实 Qwen3-4B 端到端
验证。网关能够区分自身存活与上游就绪状态，并透明转发普通 JSON 和 SSE 流式
Chat Completions 请求。

## 实现范围

| 能力 | 状态 |
| --- | --- |
| `GET /health` 网关存活检查 | 通过 |
| `GET /ready` vLLM 就绪检查 | 通过 |
| `POST /v1/chat/completions` 普通转发 | 通过 |
| `POST /v1/chat/completions` SSE 流式转发 | 通过 |
| 客户端 `x-request-id` 透传与回传 | 通过 |
| 自动生成请求 ID | 通过 |
| 上游连接失败统一返回 503 | 通过 |
| 上游超时统一返回 504 | 已实现并由异常分支覆盖 |
| 非法 JSON 返回 400 | 通过 |
| 上游错误响应保留状态码和正文 | 已实现 |

## 服务拓扑

```text
Client
  │
  ▼ 127.0.0.1:8080
FastAPI Gateway
  │
  ▼ 127.0.0.1:8000
vLLM Qwen3-4B FP16/O0
```

网关自身 `/health` 不依赖模型后端；`/ready` 只有在 vLLM `/health` 返回 200 时
才返回 200。关闭 vLLM 后，网关保持存活，但 `/ready` 返回 503 和结构化错误。

## 单元测试

测试覆盖：

1. 网关健康检查不依赖上游；
2. 上游健康时 readiness 通过；
3. 上游连接失败时 readiness 返回 503；
4. 普通 Chat Completion 请求与请求 ID 正确转发；
5. Chat Completion 上游不可用时返回统一错误；
6. SSE 数据按多个异步字节块完整转发并正确关闭响应；
7. 非法 JSON 不调用上游并返回 400。

执行命令：

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -v
```

## 真实端到端测试

烟雾测试依次验证：

- `GET /health`；
- `GET /ready`；
- 非流式 Qwen3 生成；
- SSE 流式 Qwen3 生成；
- 流式响应出现 `[DONE]`；
- 流式和非流式请求均返回非空正文。

测试结果见 [`smoke-test.txt`](smoke-test.txt)。普通和流式请求得到一致结果：

```text
模型网关的作用是统一管理、调度和安全控制AI模型的服务调用，实现模型的高效、
可靠和可扩展部署。
```

## 上游不可用行为

关闭 vLLM 后，网关 `/ready` 返回 HTTP 503：

```json
{
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "Generation backend is not ready",
    "request_id": "动态请求 ID",
    "retryable": true
  }
}
```

实际响应见 [`upstream-unavailable.json`](upstream-unavailable.json)。

## 依赖与可复现性

- 项目使用独立 `.venv`；
- `pyproject.toml` 声明运行和开发依赖；
- `uv.lock` 锁定完整版本；
- 通过 `uv sync --dev` 可以重建环境；
- vLLM 仍在独立的 `/home/aida/vllm/.venv` 中运行，网关不依赖导入 vLLM。

## 已知边界

- Day 2 只转发真实后端模型名 `qwen3-4b`；逻辑模型路由将在 Day 3 实现；
- API Key、限流、指标和日志治理将在后续阶段实现；
- 当前不会重试生成请求，避免重复生成带来的非幂等问题；
- FastAPI TestClient 产生一条上游库迁移提示，不影响功能；
- vLLM 在测试结束时使用中止模式退出，日志中的 EngineDeadError 出现在关闭阶段，
  不是请求处理失败。

## Day 2 验收状态

- [x] FastAPI 网关可独立启动；
- [x] 网关 health/readiness 语义分离；
- [x] 普通 Chat Completions 转发通过；
- [x] SSE 流式转发通过；
- [x] 请求 ID 可追踪；
- [x] 上游不可用和非法请求具有结构化错误；
- [x] 单元测试和 Ruff 检查通过；
- [x] 真实 Qwen3 端到端测试通过；
- [x] 临时服务关闭，GPU 已释放。

Day 3 将实现模型注册，以及 `qwen3-fast`/`qwen3-reasoning` 到真实
`qwen3-4b` 后端的能力路由。
