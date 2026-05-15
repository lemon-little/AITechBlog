---
layout:     post
title:      "我是怎么用 Skills 沉淀工作流的"
subtitle:   "把每一次重复都变成一份可复用的 SKILL.md"
date:       2026-05-15
author:     "刘晨阳"
image:      "https://img.lemon-little.com/post-bg-rwd.jpg"
tags:
    - skills
    - 工作流
    - Agent
    - Claude
---

# 我是怎么用 Skills 沉淀工作流的

> 这是 `skill-workflow/` 这个新版块的第一篇文章。后续我会把每一个稳定下来的 Skill / 工作流都补到这里来。

## 为什么需要 Skills

当我用 Claude / Agent 处理重复性任务（写邮件、整理日报、做技术总结、读论文、复盘 MR）时，常常发现：

1. **同样的提示语我每次都在重写**，但每次又比上次差一点点。
2. **同样的踩坑我每次都要重新踩一遍**，没有地方可以累积"如何让它这次别再翻车"。
3. 把这些经验放在 `notes/` 里，最后变成"看上去很多但用不起来"的笔记。

Skills 是把这些工作流**显式化**的形式：

- 一个 `SKILL.md`（怎么用、什么时候用、不用的情况）
- 配套的提示模板、参考资料、可执行片段
- 一句能让 Agent 自己识别"这个场景该用我"的描述

## 我用什么样的结构沉淀一个 Skill

```
skill-workflow/
└── <skill-name>/
    ├── SKILL.md         # 入口，what / when / how
    ├── prompt.md        # 核心提示词模板
    ├── examples/        # 真实使用案例
    └── references/      # 关联的论文、文档、设计稿
```

每一篇都遵循同一个三段：

1. **问题（Why）**：原本的任务有什么痛点
2. **解法（How）**：这个 skill 是怎么解决的，关键决策有哪些
3. **价值（Value）**：把它沉淀下来后省了多少事，有没有被验证过

## 接下来会写什么

- 怎么把"周报整理"做成一个稳定的 skill
- 怎么把"读论文 → 写笔记 → 写博客"串成一条自动化链路
- 怎么用 Skills 帮我每周自动整理 Claude Code session 的高价值片段

> 如果你也在沉淀自己的工作流，欢迎在 GitHub 上 fork 这个仓库，或者直接联系我交流。
