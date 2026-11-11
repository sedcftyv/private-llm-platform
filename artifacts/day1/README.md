# Day 1 验收报告

日期：2026-11-12（Asia/Shanghai）

## 结论

Qwen3-4B 已在 22 GB RTX 2080 Ti 上以 FP16、4096 Token 上下文和 vLLM O0
模式成功启动。健康检查、模型发现和 Chat Completions 生成请求均通过。

## 固定环境

| 项目 | 结果 |
| --- | --- |
| 操作系统 | WSL2 Linux 6.18.58.1 |
| Python | 3.12.16 |
| PyTorch | 2.14.0+cu130 |
| PyTorch CUDA | 13.0 |
| vLLM | 0.33.0rc1.dev1051+gfd0c471b3 |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| 显存 | 22.0 GiB / 22528 MiB |
| Compute Capability | 7.5 |
| NVIDIA Driver | 610.88 |
| 模型 | `/home/aida/vllm/Qwen3-4B` |
| 模型架构 | `Qwen3ForCausalLM` |
| 模型文件大小 | 7.6 GiB |

完整信息见 [`environment.txt`](environment.txt)。

## 服务配置

```text
served model name: qwen3-4b
listen address: 127.0.0.1:8000
dtype: float16
max model length: 4096
gpu memory utilization: 0.75
optimization level: O0
torch.compile: disabled
CUDA Graph: disabled
prefix caching: enabled
chunked prefill: enabled
```

## 启动结果

| 指标 | 结果 |
| --- | ---: |
| 权重加载耗时 | 24.65 s |
| 模型加载总耗时 | 27.05 s |
| 模型权重相关显存 | 7.56 GiB |
| 可用 KV Cache 显存 | 8.49 GiB |
| KV Cache 容量 | 60,928 tokens |
| 4096 Token 理论最大并发 | 14.88x |
| 首次启动到 API Ready | 约 41 s |
| 单请求后 GPU 已用显存 | 18,167 MiB |
| 单请求后 GPU 剩余显存 | 4,064 MiB |

完整服务日志保存在本地 `vllm-o0.log`，该文件因体积和运行时属性不提交 Git。

## 烟雾测试

检查项：

- [x] `GET /health` 返回 HTTP 200；
- [x] `GET /v1/models` 包含 `qwen3-4b`；
- [x] `POST /v1/chat/completions` 返回 HTTP 200；
- [x] `enable_thinking=false` 生效并返回非空正文；
- [x] 没有 OOM；
- [x] 没有模型架构错误；
- [x] BF16 checkpoint 成功转换为 FP16 运行。

测试输出见 [`smoke-test.txt`](smoke-test.txt)。

## 已识别现象

1. 模型 checkpoint 声明 BF16，Turing 不支持 BF16。vLLM 按启动参数将权重转换为
   FP16，属于预期行为。
2. FlashInfer Top-K/Top-P sampler 不支持 SM 7.5，已在启动脚本中设置
   `VLLM_USE_FLASHINFER_SAMPLER=0`，使用兼容 fallback。
3. 第一个真实请求触发了 `kernel_unified_attention` Triton JIT，产生一次性延迟尖峰。
   后续性能测试必须先 warm-up，不能把首请求延迟计入正式数据。
4. O0 正确关闭了 Inductor compilation 和 CUDA Graph，适合作为功能开发基线。

## Day 1 验收状态

- [x] 硬件和软件版本已固定并保存；
- [x] 本地模型配置和 Tokenizer 可离线加载；
- [x] 可重复的环境采集脚本已创建；
- [x] 可重复的 O0 服务启动脚本已创建；
- [x] 自动烟雾测试脚本已创建并通过；
- [x] 启动日志和测试结果已保存；
- [x] 临时验证服务已关闭，不继续占用 GPU。

Day 2 将实现 FastAPI 网关、健康检查和 Chat Completions 流式转发。
