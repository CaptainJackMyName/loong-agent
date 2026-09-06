# 插件系统

一个插件可以打包 **Skills**、**Agents**、**Hooks** 和 **MCP Servers**。

## 插件目录结构

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

## 插件清单（plugin.json）

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "An example plugin"
}
```

## 加载插件

```python
from loong_agent import LoongAgentOptions

options = LoongAgentOptions(
    llm=provider,
    plugins=[
        {"type": "local", "path": "./my-plugin"},
        {"type": "local", "path": "/absolute/path/to/another-plugin"},
    ],
)
```

## Agent 定义文件

`agents/*.md` 使用与 SKILL.md 类似的 front-matter：

```markdown
---
description: reviews code
tools: [Read, Grep]
---

You are a code reviewer...
```

## 直接使用插件管理器

```python
from loong_agent.plugins import PluginManager
from loong_agent.types.options import PluginRef

manager = PluginManager()
manager.load_all([PluginRef(type="local", path="./my-plugin")])

print(manager.summary())
# {'plugins': [...], 'skills': [...], 'agents': [...], 'mcp_servers': [...]}
```

加载成功后，插件的 Skills / Agents / MCP 会被聚合，并纳入会话的 SystemMessage 信息中。
