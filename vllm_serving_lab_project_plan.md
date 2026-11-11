# 私有化大模型网关与 RAG 评测平台项目计划

## 1. 项目定位

### 1.1 项目名称

**基于 vLLM 的私有化大模型网关、RAG 问答与评测监控平台**

英文名称：

> Private LLM Gateway with RAG, Evaluation and Observability Powered by vLLM

### 1.2 项目目标

在单张 22 GB RTX 2080 Ti 上部署 Qwen3-4B FP16 生成服务，并围绕它构建一套
最小但完整的企业大模型应用平台：

- 使用 FastAPI 提供统一模型网关；
- 支持普通问答和复杂推理的能力路由；
- 接入 Embedding、向量检索和 Reranker；
- 实现带来源引用的企业知识库 RAG 问答；
- 实现基于固定数据集的效果评测、失败归因和回归门禁；
- 使用结构化日志和 Prometheus 指标监控线上调用；
- 使用 vLLM benchmark 完成容量规划和推理成本分析。

最终交付物不是一个只能聊天的网页，而是一套能够回答以下企业问题的工程系统：

> 如何把一个会生成文字的模型，变成有企业知识、回答有依据、质量可测试、运行
> 可监控、成本可控制的生产服务？

### 1.3 对目标岗位的覆盖

| 岗位能力 | 项目对应模块 |
| --- | --- |
| 大模型接入 | Qwen3-4B vLLM 服务、统一客户端 |
| 模型路由 | `qwen3-fast`/`qwen3-reasoning` 路由与后端注册 |
| Embedding/Rerank | 向量召回与 CrossEncoder 重排 |
| 推理服务 | vLLM OpenAI-compatible server |
| 模型网关 | FastAPI、鉴权、超时、重试和统一错误 |
| 智能问答 | 带引用的企业知识库 RAG |
| 报告生成 | 基于检索证据的结构化报告接口 |
| 模型评测 | Recall@K、MRR、事实覆盖率、引用正确率 |
| 失败归因 | 检索、重排、生成、格式和服务失败分类 |
| 回归测试 | 固定评测集、质量阈值和 CI 风格门禁 |
| 调用监控 | Prometheus 指标、结构化日志和请求追踪 |
| 私有化适配 | 本地模型、离线知识库和敏感数据控制 |
| 推理成本优化 | 并发、O0/O2、Prefix Cache 和 SLO 容量测试 |

### 1.4 主线之外的内容

半个月内不把以下内容设为必须交付：

- 多模态图片或视频理解；
- LoRA/SFT 训练微调；
- Kubernetes、多节点和多 GPU 部署；
- 复杂的学习型模型路由器；
- 大规模分布式向量数据库；
- 修改 vLLM CUDA/Triton Kernel；
- 为了简历强行提交低价值上游 PR。

这些内容可以在主线完成后作为扩展，不能挤占评测和可复现性建设时间。

## 2. 生成模型与其他模块的关系

### 2.1 为什么不能只有生成模型

单独运行：

```bash
vllm serve /home/aida/vllm/Qwen3-4B
```

只能证明模型能够生成文字。企业应用还需要解决：

- 模型不知道企业私有知识；
- 企业制度变化后，模型训练参数不会自动更新；
- 答案可能流畅但没有证据；
- 多个业务直接连接 GPU 服务，缺少统一治理；
- 模型、Prompt 或知识库变化后可能产生质量回归；
- 请求失败时无法判断是检索、模型还是基础设施问题；
- 线上延迟、错误率和 Token 消耗不可见。

### 2.2 完整关系

```text
用户/业务系统
      │
      ▼
  FastAPI 模型网关
鉴权、路由、限流、超时、日志
      │
      ├───────────────┐
      ▼               ▼
   RAG Pipeline    直接模型调用
      │               │
      ├─ Embedding    │
      ├─ Vector Store │
      └─ Reranker     │
      │               │
      └───────┬───────┘
              ▼
      Qwen3-4B 生成模型
              │
              ▼
  答案、报告、引用和审核建议
              │
       ┌──────┴──────┐
       ▼             ▼
   离线评测       在线监控
```

各组件分工：

| 组件 | 作用 | 与生成模型的关系 |
| --- | --- | --- |
| 生成模型 | 理解、推理、组织语言 | 根据问题和证据生成最终结果 |
| Embedding | 将文本映射为语义向量 | 为生成模型寻找可能相关的知识 |
| Vector Store | 保存并搜索文档向量 | 提供候选证据 |
| Reranker | 精确判断问题与文档相关性 | 减少错误证据进入 Prompt |
| RAG | 编排召回、重排和 Prompt | 在推理时给模型注入企业知识 |
| 模型网关 | 统一接入、路由和治理 | 屏蔽后端差异并保护推理服务 |
| 评测回归 | 上线前检查质量 | 防止模型、Prompt 和检索改动退化 |
| 在线监控 | 上线后观察运行状态 | 发现超时、错误、成本和质量风险 |

### 2.3 项目业务场景

主场景选择“企业制度与技术文档知识助手”。示例能力：

1. 员工询问某项制度，系统返回带文档出处的答案；
2. 用户要求根据多份文档生成结构化摘要报告；
3. 系统根据规则文档给出审核辅助意见和依据；
4. 对资料中没有的信息明确拒答，而不是编造答案。

示例问题：

```text
出差住宿超过标准后应该如何申请报销？
```

典型处理链路：

```text
问题
 ↓ Embedding
从知识库召回 Top-20 文档片段
 ↓ Reranker
选择 Top-5 证据
 ↓ Prompt Builder
问题 + 证据 + 回答约束
 ↓ Qwen3-4B
答案 + 引用 + 置信信息
```

## 3. 总体架构

### 3.1 服务划分

```text
┌──────────────────────────────────────────────────────────┐
│                      Client / Demo                       │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP/SSE
┌──────────────────────────▼───────────────────────────────┐
│                    FastAPI LLM Gateway                   │
│ Auth │ Routing │ Timeout │ Retry │ Request ID │ Metrics │
└───────┬──────────────────┬─────────────────────┬─────────┘
        │                  │                     │
        ▼                  ▼                     ▼
┌──────────────┐  ┌────────────────┐   ┌─────────────────┐
│ vLLM Server  │  │ Retrieval API  │   │ Evaluation API  │
│ Qwen3-4B     │  │ Embed + FAISS  │   │ Dataset + Rules │
│ FP16         │  │ + Reranker     │   │ + Attribution   │
└──────────────┘  └────────────────┘   └─────────────────┘
        │                  │                     │
        └──────────────────┴─────────────────────┘
                           │
                  Logs + Prometheus Metrics
```

### 3.2 API 设计

模型网关提供：

```text
POST /v1/chat/completions
POST /v1/embeddings
POST /v1/rerank
POST /v1/knowledge/chat
POST /v1/report/generate
POST /v1/evaluations/run
GET  /health
GET  /ready
GET  /metrics
```

主要接口职责：

| 接口 | 职责 |
| --- | --- |
| `/v1/chat/completions` | 统一转发普通或推理模式请求 |
| `/v1/embeddings` | 返回文本向量 |
| `/v1/rerank` | 对候选文档重新排序 |
| `/v1/knowledge/chat` | 完整 RAG 问答，返回引用 |
| `/v1/report/generate` | 基于检索材料输出结构化报告 |
| `/v1/evaluations/run` | 运行指定评测集，开发环境使用 |
| `/metrics` | Prometheus 指标 |

### 3.3 模型路由

第一版不训练复杂路由器，而是实现可解释的配置路由：

```text
model=qwen3-fast
  → Qwen3-4B
  → enable_thinking=false
  → 普通问答、摘要、格式转换

model=qwen3-reasoning
  → Qwen3-4B
  → enable_thinking=true
  → 复杂分析、报告和审核辅助

endpoint=/v1/embeddings
  → Embedding Backend

endpoint=/v1/rerank
  → Reranker Backend
```

路由结果必须写入响应元数据和结构化日志，便于统计不同路由的延迟与调用量。

## 4. 硬件与模型选择

### 4.1 硬件

| 项目 | 配置 |
| --- | --- |
| GPU | 22 GB RTX 2080 Ti |
| 架构 | Turing |
| Compute Capability | 7.5 |
| 生成模型计算精度 | FP16 |
| BF16 | 不支持 |
| 原生 FP8 Tensor Core | 不支持 |
| CUDA Graph | 支持 |

正式开发前记录：

```bash
nvidia-smi

.venv/bin/python -c \
'import torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability()); print(torch.cuda.get_device_properties(0).total_memory / 2**30)'
```

### 4.2 生成模型

```text
模型路径：/home/aida/vllm/Qwen3-4B
API 名称：qwen3-4b
架构：Qwen3ForCausalLM
加载 dtype：FP16
最大上下文：4096（项目初始值）
```

模型配置虽然声明 `bfloat16`，但启动时必须显式使用 `--dtype half`。

当前 vLLM 原生支持 Qwen3，配置中也没有自定义 `auto_map`，因此不使用
`--trust-remote-code`。

### 4.3 Embedding 与 Reranker

为了在半个月内完成主线，优先使用较小的中文或多语言模型：

```text
Embedding 候选：BAAI/bge-small-zh-v1.5
Reranker 候选：BAAI/bge-reranker-base
```

第一阶段可以通过 Transformers/Sentence Transformers 在 CPU 或剩余 GPU 显存上运行；
网关通过 Backend 接口隔离实现。主线完成后，再尝试将支持的 Pooling/Rerank 模型
部署为独立 vLLM 服务。

不要为了“所有模型都用 vLLM”导致三个服务争抢同一张 GPU，影响主项目交付。

### 4.4 向量存储

第一版使用本地 FAISS：

- 不依赖外部服务；
- 便于保存和重建索引；
- 足够支撑几十到几千个文档片段；
- 更适合演示和自动化测试。

Qdrant、Milvus 或 Elasticsearch 作为扩展，不进入最小完成标准。

## 5. Qwen3 vLLM 服务

### 5.1 开发基线

```bash
vllm serve /home/aida/vllm/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  -O0
```

先用 O0 验证功能，减少编译与 CUDA Graph 对调试的干扰。

### 5.2 生产实验配置

```bash
vllm serve /home/aida/vllm/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  -O2
```

保存启动日志中的：

- Attention Backend；
- 编译缓存命中情况；
- CUDA Graph 模式和捕获尺寸；
- GPU KV Cache Token 数；
- 理论最大并发；
- fallback、自动降级和 warning。

### 5.3 Thinking 路由

普通问答请求发送：

```json
{
  "model": "qwen3-fast",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

复杂推理请求发送：

```json
{
  "model": "qwen3-reasoning",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": true}
}
```

网关将两个逻辑模型名映射为后端的 `qwen3-4b`，并附加对应模板参数。

## 6. RAG Pipeline

### 6.1 文档处理

支持 Markdown、TXT，时间充足时再加入 PDF。

处理步骤：

```text
读取文档
  ↓
文本清洗
  ↓
按标题和长度分块
  ↓
生成稳定 chunk_id
  ↓
计算 Embedding
  ↓
写入 FAISS 和 Metadata Store
```

每个文档片段至少保存：

```json
{
  "chunk_id": "travel-policy-001",
  "document_id": "travel-policy",
  "document_title": "差旅报销制度",
  "section": "住宿费超标处理",
  "content": "...",
  "source_path": "documents/travel_policy.md",
  "version": "2026-01"
}
```

### 6.2 检索与重排

默认策略：

```text
Embedding 召回 Top-20
        ↓
Reranker 重排
        ↓
保留 Top-5
        ↓
构建带编号引用的 Prompt
```

返回结果必须保留召回分数、重排分数和最终排名，方便失败归因。

### 6.3 Prompt 约束

核心约束：

```text
你是企业知识助手。
只根据给定资料回答，不得使用无法验证的外部事实。
如果资料不足，明确回答“根据当前资料无法确定”。
重要结论后使用 [1]、[2] 标注来源。
不要编造制度名称、金额、日期或审批流程。
```

### 6.4 RAG 响应

```json
{
  "request_id": "req-...",
  "answer": "住宿费超标时需要提交说明并经过审批。[1]",
  "citations": [
    {
      "index": 1,
      "document_id": "travel-policy",
      "chunk_id": "travel-policy-001",
      "title": "差旅报销制度",
      "rerank_score": 0.91,
      "content": "..."
    }
  ],
  "route": "qwen3-fast",
  "usage": {
    "prompt_tokens": 620,
    "completion_tokens": 85
  },
  "latency_ms": {
    "retrieval": 12,
    "rerank": 28,
    "generation": 540,
    "total": 590
  }
}
```

### 6.5 报告生成

`/v1/report/generate` 接收主题、报告模板和文档范围，输出结构化结果：

```json
{
  "title": "差旅制度执行情况分析",
  "summary": "...",
  "findings": ["..."],
  "risks": ["..."],
  "recommendations": ["..."],
  "citations": ["travel-policy-001"]
}
```

必须进行 JSON Schema/Pydantic 校验。格式失败时允许一次受控重试，并记录重试指标。

## 7. 模型网关

### 7.1 核心职责

- 统一模型和能力注册；
- API Key 验证；
- 请求 ID 和链路追踪；
- 模型/能力路由；
- 超时、有限重试和统一错误；
- 流式响应转发；
- Token 用量和延迟统计；
- 敏感日志字段脱敏；
- Prometheus 指标暴露。

### 7.2 配置示例

```yaml
models:
  qwen3-fast:
    backend: qwen3
    upstream_model: qwen3-4b
    enable_thinking: false
    timeout_seconds: 60

  qwen3-reasoning:
    backend: qwen3
    upstream_model: qwen3-4b
    enable_thinking: true
    timeout_seconds: 120

backends:
  qwen3:
    type: openai_compatible
    base_url: http://127.0.0.1:8000/v1

  embedding:
    type: sentence_transformers
    model: BAAI/bge-small-zh-v1.5

  reranker:
    type: cross_encoder
    model: BAAI/bge-reranker-base
```

### 7.3 错误模型

统一返回：

```json
{
  "error": {
    "code": "MODEL_TIMEOUT",
    "message": "Generation backend timed out",
    "request_id": "req-...",
    "retryable": true
  }
}
```

错误类型至少包括：

```text
INVALID_REQUEST
UNAUTHORIZED
MODEL_NOT_FOUND
MODEL_UNAVAILABLE
MODEL_TIMEOUT
RETRIEVAL_FAILED
RERANK_FAILED
OUTPUT_VALIDATION_FAILED
RATE_LIMITED
INTERNAL_ERROR
```

重试只用于明确可重试且幂等安全的失败，不能无限重试生成请求。

## 8. 评测体系

### 8.1 评测集

构造 30～50 条人工样本，覆盖：

- 单文档事实问答；
- 多文档综合问答；
- 时间、金额、条件等精确信息；
- 相似制度之间的区分；
- 无答案问题；
- 需要拒答的问题；
- 报告生成和结构化输出。

样本格式：

```json
{
  "id": "case-001",
  "question": "住宿费超标后如何处理？",
  "reference_answer": "...",
  "relevant_document_ids": ["travel-policy"],
  "relevant_chunk_ids": ["travel-policy-001"],
  "required_facts": ["提交超标说明", "经过审批"],
  "forbidden_facts": ["自动报销"],
  "should_abstain": false,
  "category": "single_document_fact"
}
```

### 8.2 分层评测

不要只给最终答案打一个总分。分别评测：

#### 检索层

- Recall@1/5/20；
- MRR；
- 正确文档首次出现位置；
- 空召回比例。

#### 重排层

- Rerank 前后 Recall@K；
- MRR/NDCG 变化；
- 正确片段是否进入最终 Top-5。

#### 生成层

- 必要事实覆盖率；
- 禁止事实命中数；
- 引用存在率；
- 引用与答案一致性；
- 无答案问题拒答率；
- JSON Schema 通过率。

#### 服务层

- 成功率；
- P50/P95/P99 端到端延迟；
- TTFT、TPOT；
- 超时率和重试率；
- 平均输入/输出 Token 数。

### 8.3 失败归因

每个失败样本归入一个主要原因：

| 类型 | 判断依据 | 优先修复位置 |
| --- | --- | --- |
| `RETRIEVAL_MISS` | 正确文档未进入候选集 | 分块、Embedding、查询改写 |
| `RERANK_FAILURE` | 正确片段召回但未进入 Top-5 | Reranker、Top-K 参数 |
| `KNOWLEDGE_MISSING` | 知识库不存在答案 | 文档治理 |
| `CONTEXT_INSUFFICIENT` | 证据片段不完整 | 分块和上下文拼接 |
| `HALLUCINATION` | 答案包含证据不支持的事实 | Prompt、模型、拒答策略 |
| `CITATION_ERROR` | 引用与结论不对应 | 引用映射和生成约束 |
| `FORMAT_ERROR` | 结构化输出校验失败 | Schema、Prompt、重试 |
| `TIMEOUT` | 检索或生成超过时限 | 容量、并发、超时配置 |
| `BACKEND_ERROR` | 下游模型服务异常 | 网关容错和运维 |

失败归因规则先使用确定性逻辑和人工复核，不急于实现另一个 LLM Judge。

### 8.4 回归门禁

初始门槛应根据第一轮真实基准设置。示例：

```text
Recall@5 >= 0.90
MRR >= 0.80
必要事实覆盖率 >= 0.85
引用正确率 >= 0.90
无答案拒答率 >= 0.85
JSON Schema 通过率 >= 0.98
请求成功率 = 1.00
P95 端到端延迟 <= 实测可行阈值
```

回归测试命令示例：

```bash
.venv/bin/python -m pytest tests/eval/test_rag_regression.py -v
```

质量变化必须能够追溯到具体样本和失败类型，不能只输出一个平均分。

## 9. 在线监控

### 9.1 Prometheus 指标

至少暴露：

```text
llm_gateway_requests_total
llm_gateway_request_errors_total
llm_gateway_request_duration_seconds
llm_gateway_active_requests
llm_gateway_input_tokens_total
llm_gateway_output_tokens_total
llm_gateway_route_total
llm_gateway_retries_total
rag_retrieval_duration_seconds
rag_rerank_duration_seconds
rag_empty_retrieval_total
rag_citations_per_response
rag_output_validation_failures_total
```

标签必须控制基数。可以使用 `route`、`endpoint`、`status_code`、`error_code`，
不能把 `request_id`、用户问题或文档内容作为 Prometheus label。

### 9.2 结构化日志

```json
{
  "timestamp": "...",
  "request_id": "req-...",
  "endpoint": "/v1/knowledge/chat",
  "route": "qwen3-fast",
  "status": "success",
  "input_tokens": 620,
  "output_tokens": 85,
  "retrieval_ms": 12,
  "rerank_ms": 28,
  "generation_ms": 540,
  "total_ms": 590,
  "retrieved_chunks": 20,
  "selected_chunks": 5,
  "failure_type": null
}
```

默认不记录完整 Prompt、答案、API Key 或原始私有文档。需要调试样本时使用显式、
受控的采样开关，并对个人和敏感信息脱敏。

### 9.3 评测与监控闭环

```text
线上指标或用户反馈发现问题
            ↓
保存脱敏失败样本
            ↓
人工分类失败原因
            ↓
加入离线评测集
            ↓
修改检索、Prompt、模型或服务配置
            ↓
运行回归门禁
            ↓
通过后发布并继续监控
```

## 10. vLLM 容量规划与成本优化

性能实验从原项目主线缩减为一个容量规划模块，回答：

> 在满足 P95 TTFT 和 TPOT 要求的前提下，这张 2080 Ti 能稳定承载多少并发？

### 10.1 Benchmark 基线

```bash
vllm bench serve \
  --backend vllm \
  --endpoint /v1/completions \
  --model /home/aida/vllm/Qwen3-4B \
  --served-model-name qwen3-4b \
  --tokenizer /home/aida/vllm/Qwen3-4B \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --request-rate inf \
  --max-concurrency 8 \
  --save-result \
  --save-detailed
```

### 10.2 必做实验

#### 并发度

```text
max_concurrency = 1, 4, 8, 16, 32
```

#### 优化策略

```text
-O0
-O2
```

#### Prefix Caching

```text
开启：--enable-prefix-caching
关闭：--no-enable-prefix-caching
```

使用共享 512～1024 Token 企业 System Prompt 的负载验证缓存收益。每组实验必须
重启或清除缓存，防止上一轮数据污染结果。

#### 典型长度

| 场景 | 输入 Token | 输出 Token |
| --- | ---: | ---: |
| 短问答 | 128 | 128 |
| 常规 RAG | 512 | 128 |
| 长上下文 RAG | 2048 | 128 |
| 报告生成 | 1024 | 512 |

### 10.3 指标

- Request throughput；
- Output token throughput；
- P50/P95/P99 TTFT；
- P50/P95 TPOT；
- 请求成功率；
- GPU 峰值显存；
- 服务启动和编译时间；
- 满足 SLO 的 Goodput。

### 10.4 实验规范

1. 固定模型、dtype、上下文长度和随机种子；
2. 每个正式配置至少重复 3 次；
3. 预热后再测量；
4. 保存原始 JSON 和服务日志；
5. 记录 GPU 温度、功耗和频率；
6. Profiling 数据不作为正式 benchmark 数字；
7. 不把 Turing 结果直接与 H100 数字比较；
8. 所有简历性能数字都能追溯到原始结果。

## 11. 项目结构

建议创建独立仓库，不把业务代码混入 vLLM 上游源码：

```text
private-llm-platform/
├── configs/
│   ├── gateway.yaml
│   ├── models.yaml
│   └── evaluation.yaml
├── documents/
│   ├── policies/
│   └── technical/
├── evals/
│   ├── datasets/
│   ├── baselines/
│   └── reports/
├── src/private_llm_platform/
│   ├── gateway/
│   │   ├── app.py
│   │   ├── auth.py
│   │   ├── routing.py
│   │   ├── errors.py
│   │   └── middleware.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── vllm_backend.py
│   │   ├── embedding_backend.py
│   │   └── reranker_backend.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── chunking.py
│   │   ├── index.py
│   │   ├── retrieval.py
│   │   ├── rerank.py
│   │   └── prompt.py
│   ├── evaluation/
│   │   ├── dataset.py
│   │   ├── retrieval_metrics.py
│   │   ├── answer_metrics.py
│   │   ├── attribution.py
│   │   └── regression.py
│   ├── observability/
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   └── benchmark/
│       ├── runner.py
│       └── analyzer.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
├── results/
│   ├── raw/
│   └── summary/
├── dashboards/
├── scripts/
├── pyproject.toml
├── README.md
└── report.md
```

## 12. 测试策略

### 12.1 单元测试

- 配置解析；
- 路由规则；
- 文档分块稳定性；
- 引用编号映射；
- Recall@K/MRR 计算；
- 失败归因规则；
- SLO 配置选择；
- 日志脱敏。

### 12.2 集成测试

- 网关到 mock vLLM backend；
- 文档导入到 FAISS 检索；
- Embedding 召回到 Reranker；
- RAG 请求返回答案和引用；
- 后端超时和统一错误；
- 流式响应断开处理；
- `/metrics` 指标更新。

### 12.3 回归测试

使用固定评测集连接真实模型运行。测试报告保存每个失败样本，模型型测试与快速单元
测试分开执行，避免普通开发测试每次加载大模型。

## 13. 15 天执行计划

| 天数 | 工作内容 | 当日交付物 |
| --- | --- | --- |
| Day 1 | 固定环境，验证 Qwen3-4B vLLM 服务 | 环境与启动日志 |
| Day 2 | 实现 FastAPI 网关和流式转发 | Chat 接口、健康检查 |
| Day 3 | 实现模型注册、fast/reasoning 路由 | 路由配置和测试 |
| Day 4 | 接入 Embedding、文档分块和 FAISS | 可检索知识库 |
| Day 5 | 接入 Reranker | 召回与重排结果 |
| Day 6 | 完成带引用的 RAG 问答 | `/v1/knowledge/chat` |
| Day 7 | 实现结构化报告接口 | Pydantic Schema 与重试 |
| Day 8 | 制作 30～50 条评测集 | 带证据标注的数据集 |
| Day 9 | 实现 Recall@K、MRR 和重排评测 | 检索评测报告 |
| Day 10 | 实现事实、引用、拒答和格式评测 | 生成评测报告 |
| Day 11 | 实现失败归因和回归门禁 | Regression 测试 |
| Day 12 | 加入 Prometheus 指标与结构化日志 | `/metrics` 与日志样例 |
| Day 13 | 完成并发、O0/O2、Prefix Cache 实验 | 容量与 SLO 报告 |
| Day 14 | 完成 README、架构图和演示脚本 | 可复现项目仓库 |
| Day 15 | 从干净环境复现并准备面试 | 最终数据与讲解稿 |

进度落后时的优先级：

```text
网关 → RAG → Embedding/Rerank → 评测回归 → 监控 → 性能实验 → 报告生成
```

不能为了 Grafana 页面或前端效果，牺牲评测集和失败归因。

## 14. 最终演示流程

演示控制在 8～10 分钟：

1. 展示架构和硬件限制；
2. 启动或展示已运行的 Qwen3 vLLM 服务；
3. 通过网关进行普通问答和 reasoning 路由；
4. 提问企业制度问题，展示召回、重排、答案和引用；
5. 提问知识库不存在的问题，展示拒答；
6. 修改一个检索参数，展示评测回归发现退化；
7. 展示失败归因报告；
8. 展示 Prometheus 指标或 Grafana 面板；
9. 展示并发—延迟曲线和推荐容量；
10. 说明项目限制和下一步计划。

## 15. README 与报告

README 必须包含：

1. 企业问题和项目目标；
2. 系统架构图；
3. 各模块与生成模型的关系；
4. 硬件和软件环境；
5. 安装、模型准备和启动方法；
6. 文档导入和索引构建方法；
7. API 示例；
8. 评测集格式和评测命令；
9. 失败归因说明；
10. 监控指标；
11. 性能实验方法和结果；
12. 安全、隐私和适用范围；
13. 已知限制和后续计划。

最终报告至少包含：

- Reranker 前后的 Recall@K/MRR；
- 必要事实覆盖率和引用正确率；
- 无答案拒答率；
- 失败类型分布；
- 并发—吞吐/TTFT/TPOT 曲线；
- O0/O2 对比；
- Prefix Caching 对共享前缀的收益；
- SLO 下推荐的最大并发配置。

## 16. 简历表述模板

获得真实结果后再替换所有方括号：

> **基于 vLLM 的私有化大模型网关与 RAG 评测平台**
> 基于 vLLM 在单张 22 GB RTX 2080 Ti 上部署 Qwen3-4B FP16 服务，使用
> FastAPI 构建统一模型网关，支持 Chat、Embedding、Rerank、知识问答和报告生成；
> 构建文档分块、向量召回、CrossEncoder 重排及带引用生成的 RAG 链路，并实现
> fast/reasoning 能力路由；建立包含 `[真实数量]` 条样本的回归评测集，通过
> Recall@K、MRR、事实覆盖率、引用正确率和拒答率进行分层评测与失败归因；接入
> Prometheus 指标和结构化日志，并通过并发、CUDA Graph 与 Prefix Caching 实验，
> 在 P95 TTFT `[真实数字]` ms 的约束下实现 `[真实数字]` requests/s。

可拆分为：

- 基于 vLLM 部署 Qwen3-4B OpenAI-compatible 服务，使用 FastAPI 实现统一模型
  网关、能力路由、流式转发、超时控制和统一错误处理；
- 构建 Embedding 召回、FAISS 索引、Reranker 重排和带引用生成的 RAG 链路；
- 建立 `[真实数量]` 条评测样本，通过 Recall@K、MRR、事实覆盖率、引用正确率和
  拒答率评估效果，并实现检索失败、重排失败、幻觉和超时归因；
- 接入 Prometheus 与结构化日志，监控请求量、错误率、端到端延迟、Token 用量和
  模型路由分布；
- 对并发度、O0/O2 和 Prefix Caching 进行容量测试，得到满足延迟 SLO 的部署配置。

禁止填写没有原始数据支持的性能提升比例。

## 17. 面试准备清单

### 17.1 系统设计

1. 为什么企业不能直接让业务调用 `vllm serve`？
2. 模型网关和普通 API Gateway 有什么区别？
3. 为什么选择配置路由，而不是训练一个路由模型？
4. 流式请求超时和断开应该如何处理？
5. 如何避免日志泄露用户问题和企业文档？

### 17.2 RAG

1. Embedding、Reranker 和生成模型分别负责什么？
2. 为什么不能直接把所有文档放进 Prompt？
3. Chunk 大小如何影响 Recall 和上下文完整性？
4. 为什么需要两阶段“召回 + 重排”？
5. 如何判断错误来自检索还是生成？
6. RAG 与微调分别适合解决什么问题？

### 17.3 评测与监控

1. Recall@K、MRR 和 NDCG 的含义是什么？
2. 为什么只用一个 LLM Judge 不够可靠？
3. 如何评估引用正确性和无答案拒答？
4. 离线评测和线上监控有什么区别？
5. 如何把线上失败样本加入回归集？
6. Prometheus 为什么不能用 request ID 作为 label？

### 17.4 vLLM

1. Continuous Batching 与静态 Batch 有什么区别？
2. Prefill 与 Decode 的性能特征有何不同？
3. Paged KV Cache 如何影响显存利用率？
4. TTFT、TPOT、ITL 和 Goodput 分别是什么？
5. `torch.compile` 与 CUDA Graph 有什么区别？
6. Prefix Caching 在什么负载下有效？
7. 2080 Ti 为什么必须使用 FP16 而不是 BF16？

## 18. 项目验收标准

### 18.1 最小完成标准

- [ ] Qwen3-4B FP16 vLLM 服务稳定运行；
- [ ] 网关支持 Chat、Embedding、Rerank 和 RAG 接口；
- [ ] 实现 fast/reasoning 路由；
- [ ] 至少导入 20 份自建或公开文档；
- [ ] RAG 返回答案、引用、召回和重排元数据；
- [ ] 至少 30 条人工评测样本；
- [ ] 实现 Recall@K、MRR、事实覆盖率和拒答率；
- [ ] 每个失败样本具有明确失败分类；
- [ ] 至少一个回归门禁测试；
- [ ] 暴露 Prometheus 指标并输出脱敏结构化日志；
- [ ] 完成并发、O0/O2 和 Prefix Cache 性能对比；
- [ ] README 可供另一位开发者复现；
- [ ] 简历中的所有数字都可追溯到原始结果。

### 18.2 理想完成标准

- [ ] 支持流式 RAG 响应；
- [ ] 报告生成结果通过 JSON Schema 校验；
- [ ] 评测报告能够自动生成 Markdown/HTML；
- [ ] 提供 Grafana Dashboard；
- [ ] 自动运行服务健康检查和 benchmark；
- [ ] 保存版本化评测基线；
- [ ] 完成一次 Profile 并解释 O0/O2 差异；
- [ ] 能在 10 分钟内完整演示和讲解项目。

## 19. 风险与回退方案

| 风险 | 回退方案 |
| --- | --- |
| 三个模型同时占用 GPU 导致 OOM | Qwen3 使用 GPU，Embedding/Reranker 使用 CPU 或按需加载 |
| Qwen3 O2 编译失败 | 功能开发使用 O0，性能实验单独排查 O1/O2 |
| PDF 解析耗时过多 | 主线只支持 Markdown/TXT |
| 文档数据不足 | 创建不含真实隐私信息的模拟企业制度与技术文档 |
| 自动事实评测不可靠 | 使用 required/forbidden facts 加人工复核 |
| LLM Judge 结果不稳定 | 不将 LLM Judge 作为唯一门禁 |
| Prefix Cache 污染实验 | 每组重启服务、重置缓存或使用 sweep 工具 |
| GPU 性能波动 | 记录温度、功耗和频率，增加重复次数 |
| 时间不足 | 先完成网关、RAG、评测回归和监控，再缩减性能矩阵 |

## 20. 安全与隐私

- 网关监听本机或受控内网地址；
- API Key 不写入源码和日志；
- 不使用真实企业机密作为公开项目数据；
- 默认不保存完整 Prompt、答案和文档内容；
- 文档索引与 Metadata Store 位于明确的数据目录；
- 上传文件限制格式、尺寸和解析时间；
- 检索结果需要进行访问控制，不能跨部门泄露文档；
- 模型输出作为辅助信息，不自动执行高风险操作；
- 审核辅助结果必须保留证据并由人最终确认。

## 21. 上游贡献边界

该项目作为独立应用仓库完成，不直接把业务代码写入 vLLM 主仓库。

只有发现真实、可复现且非重复的 vLLM 问题时，才按照以下流程贡献：

```text
最小复现
  → 检查现有 Issue 和 PR
  → 设计行为测试
  → 最小修复
  → 运行相关测试和性能验证
  → 执行 pr-checklist
```

不得为了简历提交 typo、孤立格式修改或其他低价值 PR。
