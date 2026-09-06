# Skills 与子智能体

## Skills

Skills 以 `SKILL.md` 文件形式存在，是 Agent 可自主调用的指令包。

### 目录结构

```
.claude/skills/
├── pdf/
│   └── SKILL.md
├── docx/
│   └── SKILL.md
└── web-search/
    └── SKILL.md
```

### SKILL.md 格式

```markdown
---
name: pdf
description: 处理 PDF 文件
---

# PDF 处理流程

1. 读取 PDF
2. 提取文本
...
```

Front-matter 中的 `name` 与 `description` 用于注册与描述。

### 启用方式

```python
# 启用所有发现的 Skills
options = LoongAgentOptions(skills="all")

# 仅启用特定 Skills
options = LoongAgentOptions(skills=["pdf", "docx"])

# 禁用所有 Skills（默认）
options = LoongAgentOptions(skills=[])
```

启用后，循环中会注册一个 `Skill` 工具，模型可调用它加载某技能的指令内容。

## 子智能体

子智能体在独立上下文中运行，支持专用系统提示、工具限制与可选模型。

### 定义

```python
from loong_agent import AgentDefinition

options = LoongAgentOptions(
    llm=provider,
    agents={
        "code-reviewer": AgentDefinition(
            description="Expert code reviewer for security and style",
            system_prompt="You are a senior code reviewer...",
            tools=["Read", "Grep", "Glob"],
        ),
        "test-runner": AgentDefinition(
            description="Runs tests and reports results",
            system_prompt="You run tests and analyze output...",
            tools=["Bash"],
            parallel=True,
        ),
    },
)
```

### 委托

启用子智能体后，循环中会注册一个 `Agent` 工具，主 Agent 可调用它把任务委托给指定子智能体：

```python
# 主 Agent 内部的工具调用：
# Agent(agent_name="code-reviewer", prompt="Review src/ for security issues")
```

### 并行执行

编排器支持并发运行多个子智能体：

```python
from loong_agent.subagents import SubagentOrchestrator

orchestrator = SubagentOrchestrator(options)
results = await orchestrator.run_parallel([
    ("code-reviewer", "Review src/"),
    ("test-runner", "Run the test suite"),
])
```
