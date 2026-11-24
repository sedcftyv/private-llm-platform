# Day 4 验收报告

日期：2026-11-24（Asia/Shanghai）

## 结论

项目已完成从 Markdown/TXT 文档到可检索 FAISS 知识库的完整链路，并新增
OpenAI-compatible `/v1/embeddings`。真实 `BAAI/bge-small-zh-v1.5` 在 CPU 上
完成批量向量生成、索引持久化和语义检索，不占用 Qwen3 的 GPU 显存。

## 实现范围

| 能力 | 状态 |
| --- | --- |
| Markdown/TXT 读取和清洗 | 通过 |
| 按标题和字符长度分块 | 通过 |
| 可配置 chunk overlap | 通过 |
| 稳定哈希 `chunk_id` | 通过 |
| BGE 512 维归一化向量 | 通过 |
| FAISS `IndexFlatIP` 余弦检索 | 通过 |
| index/metadata/manifest 持久化 | 通过 |
| OpenAI-compatible Embedding 接口 | 通过 |
| float/base64 输出 | 通过 |
| CPU-only PyTorch 环境 | 通过 |

## 文档处理

每个 chunk 保存：

```text
chunk_id, document_id, document_title, section,
content, source_path, version
```

`chunk_id` 根据 document ID、章节和正文的 SHA-256 摘要生成。相同文档重复构建时
ID 保持稳定，便于增量索引、引用和评测结果对齐。

示例知识库包含：

- `documents/travel_policy.md`；
- `documents/security_policy.md`。

默认 500 字符切块、80 字符重叠。本次两个文档共生成 7 个 chunk。

## Embedding 实现

逻辑模型名：`bge-small-zh-v1.5`

实际模型：`BAAI/bge-small-zh-v1.5`

运行设备：CPU

向量维度：512

查询向量使用中文 retrieval instruction，文档向量不添加 instruction。所有向量在
写入和查询 FAISS 前进行 L2 归一化，因此 `IndexFlatIP` 分数等价于余弦相似度。

项目固定 `torch==2.9.1+cpu` 并使用 PyTorch CPU wheel 索引。相比默认解析，避免
安装 CUDA、cuDNN、NCCL 和 Triton 等与本项目 Embedding 部署策略无关的依赖。

## Embedding API

`POST /v1/embeddings` 支持：

- 单个字符串或字符串数组；
- `encoding_format=float`；
- `encoding_format=base64`；
- `user` 字符串字段；
- 与模型固定维度相等的 `dimensions`；
- OpenAI 风格的 `data/model/object/usage` 响应。

当前明确不支持 token ID 输入和任意维度裁剪，这两种请求返回 400，不会静默忽略。
接口结构参考 OpenAI 官方 OpenAPI/SDK；BGE 固定维度是本项目后端能力边界。

## 自动化测试

执行：

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -v
```

结果：Ruff 通过，17 项测试全部通过。新增测试覆盖：

1. Markdown 标题、章节与 metadata 保留；
2. 重复分块得到稳定 ID；
3. 长段落不超过长度限制；
4. FAISS 构建、保存、加载和检索；
5. Embedding float 响应结构；
6. Embedding base64 编码；
7. token ID 和未知模型错误；
8. `/v1/models` 同时列出生成与 Embedding 模型。

## 真实验收

### Embedding API

真实网关烟雾测试结果见 [`embedding-smoke-test.txt`](embedding-smoke-test.txt)：

```text
模型：bge-small-zh-v1.5
批量大小：2
维度：512
float/base64：通过
prompt tokens：26
```

### FAISS 检索

查询：

```text
住宿费超标时需要提交什么材料并经过谁审批？
```

Top-1 为“差旅报销制度 / 住宿费超标处理”，相似度约 `0.7391`。完整前三条结果
见 [`retrieval-result.json`](retrieval-result.json)。

对更宽泛的“住宿费超过标准后应该怎么报销”查询，目标片段排第 2。这说明
Embedding 召回能够覆盖正确证据，但仍需要 Day 5 的 CrossEncoder Reranker 改善
精细排序。

## 已知边界

- 当前索引面向几十到几千个 chunk，尚未引入增量更新或删除；
- metadata 使用 JSON 文件，不适合多进程并发写入；
- 首次运行需要从 Hugging Face 下载 BGE 模型；
- `/ready` 目前仍表示生成后端就绪，不会为了探活而提前加载 BGE；
- Reranker 将在 Day 5 接入。

## Day 4 验收状态

- [x] 文档清洗和稳定分块；
- [x] BGE CPU Embedding；
- [x] OpenAI-compatible Embedding API；
- [x] FAISS 索引构建和持久化；
- [x] 查询返回 chunk、分数与排名；
- [x] CPU-only 可复现依赖；
- [x] 17 项测试通过；
- [x] 真实模型与真实索引验收通过。
