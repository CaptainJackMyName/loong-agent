# 链路追踪

SDK 采用 [OpenTelemetry](https://opentelemetry.io/) 提供 Traces、Metrics 与 Events。

## Span 命名规范

```
loong_agent.<layer>.<operation>
```

| Span 名称 | 描述 |
|-----------|------|
| `loong_agent.loop.turn` | 单次 Turn 执行 |
| `loong_agent.llm.request` | LLM API 调用 |
| `loong_agent.tool.execute` | 工具执行 |
| `loong_agent.hook.run` | 钩子执行 |
| `loong_agent.mcp.call` | MCP 工具调用 |
| `loong_agent.subagent.run` | 子智能体执行 |
| `loong_agent.skill.execute` | Skill 执行 |

## 配置导出器

```python
from loong_agent.tracing import configure_tracing

# 默认使用 ConsoleSpanExporter（本地开发）
configure_tracing()

# 生产环境使用 OTLP 导出器
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
configure_tracing(exporter=OTLPSpanExporter(endpoint="http://localhost:4317"))
```

## 手动使用 Tracer

```python
from loong_agent.tracing import LoongAgentTracer

tracer = LoongAgentTracer()

async with tracer.span("loong_agent.custom.operation"):
    tracer.set_attribute("key", "value")
    tracer.add_event("step", {"index": 1})
```

## 无依赖降级

未安装 OpenTelemetry 时，`LoongAgentTracer` 自动退化为 no-op，不影响 SDK 其他功能。
