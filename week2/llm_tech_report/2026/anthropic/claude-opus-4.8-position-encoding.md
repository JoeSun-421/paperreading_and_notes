# Claude Opus 4.8：位置编码分析

## 资料与结论

- 主要资料：`claude-opus-4.8-system-card.pdf`。
- 原文：[Anthropic Transparency](https://www.anthropic.com/transparency)；[官方 System Card PDF](https://cdn.sanity.io/files/4zrzovbb/website/c886650a2e96fc0925c805a1a7ca77314ccbf4a6.pdf)。
- 结论：这份文档是安全 System Card，不是架构 Technical Report。公开内容主要是能力、安全评测和风险缓解；没有披露 RoPE、位置 embedding、theta、rotary dimension 或 position ID 构造。

## 能确认和不能确认的内容

能确认的是 Claude Opus 4.8 的部署能力、评测范围和安全分析。不能从 System Card 确认：

- 是否使用标准 RoPE、partial RoPE、NoPE 或其他位置机制；
- RoPE 施加在 Q/K 的全部维度还是子空间；
- 是否采用 YaRN、NTK scaling、ALiBi 或训练长度外推；
- 多模态/工具调用输入如何分配 position IDs。

System Card 中的长上下文评测也不能反推具体位置编码。长上下文表现可能由架构、训练数据、缓存、检索和上下文管理共同决定。

## 研究记录应如何写

```text
position encoding : 未披露
RoPE parameters    : 未披露
long-context method: 公开评测有，底层实现未披露
evidence type      : safety System Card, not architecture report
```

这类“未披露”是有价值的结果：它说明闭源 frontier model 的安全披露和架构披露是两个不同层次。分析时应避免用其他开源模型的 config 猜测 Claude 的实现。

## 对你研究的启发

如果老师要求比较各家位置编码，Claude Opus 4.8 可以作为“不可验证组”。你可以比较公开的 context benchmark 和已披露的系统级技术，但必须单独标注：没有公开公式，不能做实现级结论。

## 披露边界

本文只基于公开 System Card 做证据化总结，不把第三方实现、社区猜测或 benchmark 名称当成 Anthropic 的位置编码说明。
