# loong-agent 项目开发计划书

> 基于 ReAct 论文理论框架的自主 Agent SDK 实现

---

## 一、项目概述

### 1.1 项目背景

loong-agent 是一个开源的自主 AI Agent 开发套件，参考 ReAct（Reasoning + Acting）论文的理论框架构建。该项目旨在为开发者提供一个可自由定制、完全开源、不依赖任何商业闭源二进制文件的 Agent SDK。

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| **功能实现** | 思考-行动-观察循环、插件系统、MCP、Skills、子智能体、钩子系统、链路追踪 |
| **模型无关** | 支持任意 OpenAI-compatible 的大模型接口接入，不绑定任何特定 LLM 提供商 |
| **纯代码实现** | 所有逻辑用 Python 源代码实现，不调用任何外部可执行文件 |

### 1.3 项目信息

- **项目名称**：loong-agent
- **源码路径**：`src/loong_agent`
- **包管理**：uv（已通过 `uv init` 初始化）

---

## 二、理论基础

### 2.1 ReAct 框架

ReAct 论文的核心思想是让 LLM **交替生成推理轨迹（Reasoning Traces）和任务特定动作（Actions）** 。这种协同带来三大优势：

1. **推理轨迹**帮助模型诱导、追踪和更新动作计划，以及处理异常
2. **动作**使模型能够与外部知识库或环境交互以收集额外信息
3. 生成类人的任务解决轨迹，比无推理轨迹的基线更具可解释性

### 2.2 Agent Loop

Free Agent SDK 的核心执行循环：

```
接收提示词 → 模型评估与响应 → 执行工具 → 重复 → 返回结果
```

每个完整循环称为一个 **Turn**（回合），包含：
- **SystemMessage**（init）：会话元数据
- **AssistantMessage**：包含文本和工具调用请求
- **UserMessage**：工具执行结果反馈

---

## 三、核心架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Public API Layer                            │
│              query() / LoongAgentClient / Streaming                  │
├─────────────────────────────────────────────────────────────────────┤
│                        Agent Loop Engine                            │
│         ┌─────────┐   ┌─────────┐   ┌─────────┐                    │
│         │ Think   │ → │  Act    │ → │ Observe │  (ReAct Loop)      │
│         └─────────┘   └─────────┘   └─────────┘                    │
├─────────────────────────────────────────────────────────────────────┤
│                         Core Components                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  Tools   │ │  Hooks   │ │   MCP    │ │  Skills  │ │Subagents │ │
│  │  System  │ │  System  │ │ Gateway  │ │  System  │ │  System  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                      Plugin & Extension Layer                       │
│              Plugin Loader / Discovery / Lifecycle                  │
├─────────────────────────────────────────────────────────────────────┤
│                        LLM Provider Layer                           │
│              OpenAI-Compatible Adapter / Model Router               │
├─────────────────────────────────────────────────────────────────────┤
│                    Observability Layer                              │
│           OpenTelemetry Tracing / Metrics / Logging                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块依赖关系

```
loong_agent/
├── __init__.py          # 公开 API 导出
├── client.py            # LoongAgentClient 主客户端
├── loop/                # Agent Loop 引擎
│   ├── __init__.py
│   ├── engine.py        # 主循环控制器
│   ├── turn.py          # 单回合执行
│   └── context.py       # 会话上下文管理
├── llm/                 # LLM 适配层
│   ├── __init__.py
│   ├── base.py          # LLM 基类接口
│   ├── openai_compatible.py  # OpenAI-compatible 适配器
│   └── registry.py      # 模型注册与路由
├── tools/               # 工具系统
│   ├── __init__.py
│   ├── base.py          # Tool 基类
│   ├── builtin/         # 内置工具 (Read, Write, Bash, Grep, Glob)
│   └── registry.py      # 工具注册表
├── hooks/               # 钩子系统
│   ├── __init__.py
│   ├── events.py        # 事件定义 (PreToolUse, PostToolUse, etc.)
│   ├── matcher.py       # 钩子匹配器
│   └── registry.py      # 钩子注册
├── mcp/                 # MCP (Model Context Protocol) 网关
│   ├── __init__.py
│   ├── gateway.py       # MCP 服务器管理
│   ├── transports/      # stdio / HTTP / SSE / in-process
│   └── client.py        # MCP 客户端
├── skills/              # Skills 系统
│   ├── __init__.py
│   ├── loader.py        # SKILL.md 文件加载器
│   ├── discovery.py     # 文件系统发现
│   └── executor.py      # Skill 执行器
├── subagents/           # 子智能体系统
│   ├── __init__.py
│   ├── definition.py    # AgentDefinition
│   ├── orchestrator.py  # 子智能体编排
│   └── runner.py        # 子智能体运行器
├── plugins/             # 插件系统
│   ├── __init__.py
│   ├── loader.py        # 插件加载器
│   ├── manifest.py      # 插件清单解析
│   └── lifecycle.py     # 插件生命周期管理
├── tracing/             # 链路追踪
│   ├── __init__.py
│   ├── tracer.py        # OpenTelemetry Tracer 封装
│   ├── spans.py         # Span 定义
│   └── export.py        # 导出器配置
├── permissions/         # 权限系统
│   ├── __init__.py
│   ├── policy.py        # 权限策略
│   └── decision.py      # 权限决策
└── types/               # 类型定义
    ├── __init__.py
    ├── messages.py      # Message 类型 (System, Assistant, User, Result)
    └── options.py       # 配置选项 (LoongAgentOptions)
```

---

## 四、核心模块详细设计

### 4.1 Agent Loop 引擎（`loop/`）

**设计依据**：Claude Agent SDK 的循环机制与 ReAct 的推理-行动交织模式

**核心接口**：

```python
# 主查询接口
async def query(
    prompt: str,
    options: LoongAgentOptions = None
) -> AsyncIterator[Message]:
    """流式执行 Agent 任务，逐步返回消息"""
    
# 客户端类
class LoongAgentClient:
    async def send(self, prompt: str) -> None
    async def receive(self) -> AsyncIterator[Message]
    async def interrupt(self) -> None
    async def close(self) -> None
```

**Loop 执行流程**：

```python
class AgentLoop:
    async def run(self, prompt: str) -> AsyncIterator[Message]:
        # 1. 初始化：发送 SystemMessage (init)
        yield SystemMessage(subtype="init", session_id=..., ...)
        
        while True:
            # 2. Think: 调用 LLM，获取推理和动作
            response = await self._think()
            
            # 3. Act: 执行工具调用
            if response.has_tool_calls():
                yield AssistantMessage(content=response.content)
                results = await self._execute_tools(response.tool_calls)
                yield UserMessage(content=results)
                # 4. Observe: 将观察结果反馈给模型，继续循环
                continue
            else:
                # 5. 无工具调用，返回最终结果
                yield AssistantMessage(content=response.content)
                yield ResultMessage(text=response.text, usage=..., cost=...)
                break
```

### 4.2 LLM 适配层（`llm/`）

**设计目标**：支持任意 OpenAI-compatible API 的大模型接入。

**核心接口**：

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        **kwargs
    ) -> ChatResponse:
        """调用 LLM 进行对话补全"""
    
    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称"""

class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0
    ):
        ...
```

**配置示例**：

```python
options = LoongAgentOptions(
    llm=OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4"
    )
)
# 或使用其他兼容服务
options = LoongAgentOptions(
    llm=OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1",  # Local LLM
        api_key="not-needed",
        model="llama-3.3-70b"
    )
)
```

### 4.3 工具系统（`tools/`）

**设计依据**：Claude Agent SDK 内置工具集。

**内置工具**：

| 工具名 | 功能 | 对应 Claude SDK |
|--------|------|-----------------|
| `Read` | 读取文件 | Read |
| `Write` | 写入文件 | Write |
| `Edit` | 编辑文件 | Edit |
| `Bash` | 执行 Shell 命令 | Bash |
| `Grep` | 搜索文件内容 | Grep |
| `Glob` | 文件模式匹配 | Glob |
| `Agent` | 调用子智能体 | Agent |
| `Skill` | 调用 Skill | Skill |

**工具定义接口**：

```python
from typing import Any, Dict, Callable, Awaitable

class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具并返回结果"""

# 装饰器方式定义
@tool("search_web", "Search the web for information")
async def search_web(query: str, max_results: int = 10) -> str:
    ...
```

### 4.4 钩子系统（`hooks/`）

**设计依据**：Claude Agent SDK 的钩子机制。

**支持的事件类型**：

| 事件 | 触发时机 |
|------|----------|
| `PreToolUse` | 工具执行前 |
| `PostToolUse` | 工具执行后 |
| `SubagentStart` | 子智能体启动 |
| `SubagentStop` | 子智能体停止 |
| `SessionStart` | 会话开始 |
| `SessionEnd` | 会话结束 |
| `Idle` | Agent 空闲 |

**钩子定义接口**：

```python
class HookMatcher:
    """匹配器，用于过滤钩子触发条件"""
    event: HookEvent
    matcher: Optional[str] = None  # 正则表达式匹配

async def my_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: HookContext
) -> HookOutput:
    """钩子回调函数"""
    # 可以返回:
    # - allow: 允许操作
    # - deny: 阻止操作
    # - modify: 修改输入
    # - inject: 注入上下文
    return HookOutput(permission_decision="allow")
```

### 4.5 MCP 网关（`mcp/`）

**设计依据**：Claude Agent SDK 的 MCP 集成。

MCP（Model Context Protocol）是连接 AI Agent 到外部工具和数据源的开放标准。

**支持的传输类型**：

- **stdio**：本地子进程
- **HTTP/SSE**：远程 HTTP 服务
- **In-process**：进程内直接执行

**配置接口**：

```python
options = LoongAgentOptions(
    mcp_servers={
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        },
        "github": {
            "type": "http",
            "url": "https://api.github.com/mcp"
        }
    },
    allowed_tools=["mcp__filesystem__*", "mcp__github__*"]
)
```

**MCP Gateway 设计**：

```python
class MCPGateway:
    """管理所有 MCP 服务器连接"""
    
    async def connect(self, name: str, config: MCPServerConfig) -> None
    async def disconnect(self, name: str) -> None
    async def list_tools(self, server: str) -> List[Tool]
    async def call_tool(self, server: str, tool: str, args: Dict) -> Any
    async def close_all(self) -> None
```

### 4.6 Skills 系统（`skills/`）

**设计依据**：Claude Agent SDK 的 Skills 机制。

Skills 以 `SKILL.md` 文件形式存在，包含指令、描述和可选资源。

**文件系统发现**：

```
.claude/skills/
├── pdf/
│   └── SKILL.md
├── docx/
│   └── SKILL.md
└── web-search/
    └── SKILL.md
```

**Skill 加载与执行**：

```python
class SkillLoader:
    """从文件系统加载 SKILL.md"""
    async def discover(self, paths: List[str]) -> List[Skill]
    async def load(self, skill_name: str) -> SkillContent

class SkillExecutor:
    """执行 Skill（通过 LLM 调用或直接执行）"""
    async def execute(self, skill: Skill, context: Dict) -> str
```

**使用方式**：

```python
# 启用所有发现的 Skills
options = LoongAgentOptions(skills="all")

# 仅启用特定 Skills
options = LoongAgentOptions(skills=["pdf", "docx"])

# 禁用所有 Skills
options = LoongAgentOptions(skills=[])
```

### 4.7 子智能体系统（`subagents/`）

**设计依据**：Claude Agent SDK 的子智能体机制。

**核心优势**：

1. **上下文隔离**：每个子智能体运行在独立的对话中
2. **并行执行**：多个子智能体可并发运行
3. **专用指令**：每个子智能体可定制系统提示
4. **工具限制**：子智能体可限制特定工具

**定义接口**：

```python
@dataclass
class AgentDefinition:
    """子智能体定义"""
    description: str          # 描述，用于主 Agent 决定何时调用
    system_prompt: str        # 专用系统提示
    tools: List[str]          # 允许的工具列表
    model: Optional[str] = None  # 可选，使用不同模型
    parallel: bool = False    # 是否允许并行执行

# 使用示例
options = LoongAgentOptions(
    agents={
        "code-reviewer": AgentDefinition(
            description="Expert code reviewer for security and style",
            system_prompt="You are a senior code reviewer...",
            tools=["Read", "Grep", "Glob"]
        ),
        "test-runner": AgentDefinition(
            description="Runs tests and reports results",
            system_prompt="You run tests and analyze output...",
            tools=["Bash"]
        )
    }
)
```

### 4.8 插件系统（`plugins/`）

**设计依据**：Claude Agent SDK 的插件机制。

一个插件可以包含：

- **Skills**：Claude 自主调用的能力
- **Agents**：专用子智能体
- **Hooks**：事件处理器
- **MCP Servers**：外部工具集成

**插件目录结构**：

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json        # 插件清单
├── skills/
│   └── my-skill/
│       └── SKILL.md
├── agents/
│   └── my-agent.md
├── hooks/
│   └── my-hook.py
└── mcp/
    └── my-server.json
```

**加载插件**：

```python
options = LoongAgentOptions(
    plugins=[
        {"type": "local", "path": "./my-plugin"},
        {"type": "local", "path": "/absolute/path/to/another-plugin"}
    ]
)
```

**插件验证**：

加载成功后，`SystemMessage` 会包含加载的插件、Skills 和命令信息。

### 4.9 链路追踪（`tracing/`）

**设计依据**：Claude Agent SDK 的 OpenTelemetry 支持。

采用 **OpenTelemetry** 作为可观测性标准，提供：

- **Traces**：关键操作的 Span（查询、工具执行、钩子、MCP 调用）
- **Metrics**：性能指标
- **Events**：关键事件日志

**Span 命名规范**：

```
claude_agent_sdk.<layer>.<operation>
```

**Tracer 接口**：

```python
from opentelemetry.trace import Tracer

class LoongAgentTracer:
    def __init__(self, tracer: Optional[Tracer] = None):
        self._tracer = tracer or get_tracer("loong")
    
    def start_span(self, name: str, attributes: Dict = None):
        """启动一个新的 Span"""
    
    def add_event(self, name: str, attributes: Dict = None):
        """添加事件到当前 Span"""
    
    def set_attribute(self, key: str, value: Any):
        """设置当前 Span 属性"""
```

**关键追踪点**：

| Span 名称 | 描述 |
|-----------|------|
| `loong.loop.turn` | 单次 Turn 执行 |
| `loong.llm.request` | LLM API 调用 |
| `loong.tool.execute` | 工具执行 |
| `loong.hook.run` | 钩子执行 |
| `loong.mcp.call` | MCP 工具调用 |
| `loong.subagent.run` | 子智能体执行 |

### 4.10 权限系统（`permissions/`）

**设计依据**：Claude Agent SDK 的权限模型。

```python
@dataclass
class PermissionPolicy:
    """权限策略"""
    allow: List[str] = field(default_factory=list)   # 允许的工具/操作
    deny: List[str] = field(default_factory=list)    # 拒绝的工具/操作
    ask: List[str] = field(default_factory=list)     # 需要询问的工具/操作

class PermissionManager:
    async def check(self, tool: str, context: Dict) -> PermissionDecision:
        """检查权限"""
        if tool in policy.deny:
            return PermissionDecision.DENY
        if tool in policy.ask:
            return PermissionDecision.ASK
        return PermissionDecision.ALLOW
```

---

## 五、开发计划

### 5.1 里程碑规划

| 阶段 | 内容 | 预估工时 |
|------|------|----------|
| **Phase 0** | 项目初始化、类型定义、基础架构 | 3 天 |
| **Phase 1** | LLM 适配层 + Agent Loop 引擎（核心） | 7 天 |
| **Phase 2** | 工具系统（内置工具 + 注册机制） | 5 天 |
| **Phase 3** | 钩子系统 + 权限系统 | 5 天 |
| **Phase 4** | MCP 网关 | 5 天 |
| **Phase 5** | Skills 系统 | 4 天 |
| **Phase 6** | 子智能体系统 | 5 天 |
| **Phase 7** | 插件系统 | 4 天 |
| **Phase 8** | 链路追踪（OpenTelemetry） | 3 天 |
| **Phase 9** | 文档、示例、测试 | 5 天 |
| **总计** | | **约 46 天** |

### 5.2 详细任务分解

#### Phase 0：项目初始化与基础架构（3 天）

- [ ] 确认 `uv` 项目结构，配置 `pyproject.toml`
- [ ] 定义核心类型：`Message`、`Tool`、`Hook`、`Options`
- [ ] 建立模块目录结构
- [ ] 配置开发环境（ruff、mypy、pytest）

#### Phase 1：LLM 适配层 + Agent Loop（7 天）

- [ ] 实现 `LLMProvider` 基类
- [ ] 实现 `OpenAICompatibleProvider`
- [ ] 实现 `AgentLoop` 主控制器
- [ ] 实现 `Turn` 单回合执行
- [ ] 实现 `Context` 会话上下文
- [ ] 单元测试：LLM 适配器、Loop 流程

#### Phase 2：工具系统（5 天）

- [ ] 实现 `Tool` 基类与 `@tool` 装饰器
- [ ] 实现内置工具：Read、Write、Edit、Bash、Grep、Glob
- [ ] 实现 `ToolRegistry`
- [ ] 实现工具调用解析与结果处理
- [ ] 单元测试：每个内置工具

#### Phase 3：钩子系统 + 权限系统（5 天）

- [ ] 定义 Hook 事件类型
- [ ] 实现 `HookMatcher` 匹配器
- [ ] 实现 `HookRegistry` 注册管理
- [ ] 实现 `PermissionManager`
- [ ] 集成测试：钩子拦截与权限决策

#### Phase 4：MCP 网关（5 天）

- [ ] 实现 MCP 协议基础（参考 `mcp` 规范）
- [ ] 实现 stdio 传输
- [ ] 实现 HTTP/SSE 传输
- [ ] 实现 MCP Gateway 管理
- [ ] 集成测试：MCP 服务器连接与工具调用

#### Phase 5：Skills 系统（4 天）

- [ ] 实现 `SKILL.md` 解析器
- [ ] 实现文件系统发现
- [ ] 实现 Skill 加载器
- [ ] 实现 Skill 执行器
- [ ] 单元测试：Skill 解析与执行

#### Phase 6：子智能体系统（5 天）

- [ ] 实现 `AgentDefinition` 数据类
- [ ] 实现子智能体运行器
- [ ] 实现子智能体编排器（并行执行）
- [ ] 实现上下文隔离机制
- [ ] 集成测试：子智能体委托与结果返回

#### Phase 7：插件系统（4 天）

- [ ] 定义插件清单格式
- [ ] 实现插件加载器
- [ ] 实现插件生命周期管理
- [ ] 实现插件 Skills/Agents/Hooks/MCP 的集成加载
- [ ] 集成测试：完整插件加载

#### Phase 8：链路追踪（3 天）

- [ ] 集成 OpenTelemetry SDK
- [ ] 实现 `LoongAgentTracer`
- [ ] 在关键路径添加 Span
- [ ] 实现 Span 导出器配置
- [ ] 测试：追踪数据输出

#### Phase 9：文档与示例（5 天）

- [ ] API 文档（使用 mkdocs 或 sphinx）
- [ ] 快速开始指南
- [ ] 完整示例（每个功能模块）
- [ ] README 与贡献指南

---

## 六、技术选型

| 类别 | 技术 | 说明 |
|------|------|------|
| **语言** | Python 3.10+ | 类型注解、async/await |
| **包管理** | uv | 已初始化 |
| **HTTP 客户端** | httpx | 异步 HTTP 请求 |
| **LLM SDK** | openai | OpenAI-compatible 接口 |
| **MCP** | mcp 或自实现 | 参考 Model Context Protocol 规范 |
| **OpenTelemetry** | opentelemetry-api + opentelemetry-sdk | 链路追踪 |
| **测试** | pytest + pytest-asyncio | 异步测试 |
| **类型检查** | mypy | 静态类型 |
| **代码格式化** | ruff | 快速 lint + format |
| **文档** | mkdocs | 项目文档 |

---

## 七、API 设计概览

### 7.1 快速开始

```python
import asyncio
from loong_agent import query, LoongAgentOptions
from loong_agent.llm import OpenAICompatibleProvider

async def main():
    options = LoongAgentOptions(
        llm=OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4"
        ),
        allowed_tools=["Read", "Write", "Bash"],
        skills=["pdf", "docx"],
        hooks={
            "PreToolUse": [my_security_hook]
        }
    )
    
    async for message in query(
        prompt="Review the code in src/ and fix any issues",
        options=options
    ):
        if message.type == "assistant":
            print(f"Claude: {message.text}")
        elif message.type == "result":
            print(f"Done! Usage: {message.usage}")
```

### 7.2 高级用法

```python
# 使用客户端进行流式交互
from loong_agent import LoongAgentClient

client = LoongAgentClient(options=options)
await client.send("Analyze this project structure")

async for message in client.receive():
    if message.type == "assistant" and message.has_tool_calls():
        # 处理工具调用
        pass

await client.close()
```

---

## 八、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| MCP 协议复杂度高 | 开发周期延长 | 优先实现 stdio/HTTP，逐步完善 |
| OpenAI-compatible 接口差异 | 兼容性问题 | 实现适配器模式，支持主流提供商 |
| ReAct 循环收敛问题 | 无限循环 | 设置最大 Turn 数、超时控制 |
| 子智能体上下文管理 | 内存/性能 | 实现上下文隔离与垃圾回收 |
| 依赖 LLM 的 tool calling 能力 | 功能受限 | 对不支持 tool calling 的模型提供 fallback |

---

## 九、总结

loong 是一个**完全开源、模型无关、纯代码实现**的自主 Agent 开发套件。它融合了：

1. **Agent SDK 的设计理念**：完整的 Agent Loop、工具系统、钩子、MCP、Skills、子智能体、插件、权限、追踪
2. **ReAct 论文的理论框架**：推理与行动的协同交织
