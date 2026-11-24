# Day 3 验收报告

日期：2026-11-22（Asia/Shanghai）

## 结论

模型网关已完成配置驱动的逻辑模型注册，并通过单元测试和真实 Qwen3-4B
端到端验证。客户端只需选择 `qwen3-fast` 或 `qwen3-reasoning`，网关会将请求
映射到同一个 `qwen3-4b` vLLM 后端，并强制应用对应的 Thinking 策略和超时。

## 路由配置

| 逻辑模型 | 后端 | 上游模型 | Thinking | 请求超时 |
| --- | --- | --- | --- | --- |
| `qwen3-fast` | `qwen3` | `qwen3-4b` | `false` | 60 秒 |
| `qwen3-reasoning` | `qwen3` | `qwen3-4b` | `true` | 120 秒 |

配置保存在包内 `models.toml`，也可通过 `LLM_MODEL_REGISTRY_PATH` 加载外部
TOML 文件。注册表启动时完成字段和数值校验，错误配置会阻止网关启动。

## 实现行为

- `GET /v1/models` 只暴露客户端可使用的两个逻辑模型；
- 网关将逻辑模型名改写为上游 `qwen3-4b`；
- 路由配置强制覆盖客户端传入的 `enable_thinking`；
- 其他 `chat_template_kwargs` 保持不变；
- fast/reasoning 分别使用 60 秒和 120 秒读取超时；
- 未注册模型在访问 vLLM 前返回 HTTP 404；
- 普通 JSON 和 SSE 流式响应都返回路由元数据响应头；
- 上游 OpenAI-compatible 响应正文保持不变。

路由元数据响应头包括：

```text
x-model-route
x-model-backend
x-upstream-model
x-thinking-enabled
x-upstream-status
```

未知模型行为与 OpenAI/vLLM 的 404 语义保持一致。实际响应见
[`unknown-model-response.json`](unknown-model-response.json)。

## 自动化测试

执行命令：

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -v
```

结果：Ruff 通过，11 个测试全部通过。新增覆盖包括：

1. `/v1/models` 仅列出逻辑模型；
2. fast 路由关闭 Thinking 并使用 60 秒超时；
3. reasoning 路由开启 Thinking 并使用 120 秒超时；
4. 客户端不能覆盖路由规定的 Thinking 模式；
5. 未知模型返回 404 且不调用上游；
6. 缺失模型字段返回 400 且不调用上游；
7. SSE 流式响应保留正文及路由响应头。

测试仍有一条 FastAPI TestClient 的上游库迁移警告，不影响功能。

## 真实端到端测试

测试拓扑：

```text
gateway_smoke_test.py
  → FastAPI Gateway :8080
  → vLLM Qwen3-4B FP16/O0 :8000
```

真实验收依次检查：

- 网关 `/health` 和 `/ready`；
- 逻辑模型注册表；
- `qwen3-fast` 非流式生成；
- `qwen3-fast` SSE 流式生成及 `[DONE]`；
- `qwen3-reasoning` 强制开启 Thinking；
- reasoning 请求正常结束并得到 `17 × 19 = 323`；
- fast/reasoning 路由响应头与配置一致。

最终输出见 [`smoke-test.txt`](smoke-test.txt)。

## 已知边界

- 当前两个逻辑模型共享同一个物理模型和 vLLM 后端；
- API Key、限流、指标和结构化日志将在后续阶段实现；
- vLLM 未配置 reasoning parser，因此 Qwen3 思考内容仍位于生成文本中；
- 为保持 OpenAI-compatible 正文不变，逻辑路由信息通过响应头返回；
- 生成请求不会自动重试，避免非幂等请求产生重复结果。

## Day 3 验收状态

- [x] 配置驱动的模型注册表；
- [x] `qwen3-fast` 路由；
- [x] `qwen3-reasoning` 路由；
- [x] Thinking 参数强制覆盖；
- [x] 每路由独立超时；
- [x] 逻辑模型列表接口；
- [x] 未知模型兼容性错误；
- [x] 普通与流式路由元数据；
- [x] Ruff 和 11 项测试通过；
- [x] 真实 Qwen3-4B 端到端测试通过。
