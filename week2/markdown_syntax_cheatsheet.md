# Markdown 常用语法速查

## 1. 标题

```markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
```

建议在 `#` 后留一个空格，并让标题层级连续。

## 2. 段落与换行

不同段落之间空一行：

```markdown
这是第一段。

这是第二段。
```

需要在同一段内强制换行时，可在行末添加两个空格，或者使用 `<br>`：

```markdown
第一行  
第二行

第一行<br>
第二行
```

## 3. 强调文本

```markdown
*斜体*
_斜体_

**粗体**
__粗体__

***粗斜体***

~~删除线~~
```

显示效果：*斜体*、**粗体**、***粗斜体***、~~删除线~~。

## 4. 无序列表

```markdown
- Tokenization
- Transformer
  - Attention
  - Feed-forward network
- Optimization
```

也可以使用 `*` 或 `+`，但同一篇笔记中最好保持统一。

## 5. 有序列表

```markdown
1. 准备训练数据
2. 训练 tokenizer
3. 训练语言模型
```

嵌套列表需要缩进：

```markdown
1. Transformer block
   1. RMSNorm
   2. Attention
   3. SwiGLU
2. Output layer
```

## 6. 任务列表

```markdown
- [x] 阅读论文
- [x] 整理笔记
- [ ] 完成实验
- [ ] 分析结果
```

显示效果：

- [x] 阅读论文
- [ ] 完成实验

## 7. 引用

```markdown
> Attention is all you need.
>
> 第二段引用内容。
```

显示效果：

> Attention is all you need.

嵌套引用：

```markdown
> 第一层引用
>> 第二层引用
```

## 8. 行内代码与代码块

行内代码使用一个反引号：

```markdown
使用 `uv run pytest` 运行测试。
```

代码块使用三个反引号，并推荐标明语言以启用语法高亮：

````markdown
```python
tokens = tokenizer.encode(text)
print(tokens)
```
````

常见语言标识包括 `python`、`bash`、`json`、`yaml`、`text` 和 `markdown`。

## 9. 链接

```markdown
[CS336 课程主页](https://cs336.stanford.edu/)
```

直接显示 URL：

```markdown
<https://cs336.stanford.edu/>
```

带提示文字的链接：

```markdown
[CS336](https://cs336.stanford.edu/ "Stanford CS336")
```

## 10. 图片

```markdown
![图片的替代文字](images/attention.png)
```

带标题的图片：

```markdown
![Attention 示意图](images/attention.png "Attention")
```

替代文字用于图片无法显示时的说明，也有助于无障碍阅读。

## 11. 水平分隔线

```markdown
---
```

也可以使用 `***` 或 `___`。分隔线前后最好各留一个空行。

## 12. 表格

```markdown
| 模块 | 输入形状 | 输出形状 |
|---|---:|---:|
| Embedding | `(B, T)` | `(B, T, D)` |
| RMSNorm | `(B, T, D)` | `(B, T, D)` |
| LM head | `(B, T, D)` | `(B, T, V)` |
```

显示效果：

| 模块 | 输入形状 | 输出形状 |
|---|---:|---:|
| Embedding | `(B, T)` | `(B, T, D)` |
| RMSNorm | `(B, T, D)` | `(B, T, D)` |
| LM head | `(B, T, D)` | `(B, T, V)` |

对齐方式：

```markdown
| 左对齐 | 居中 | 右对齐 |
|:---|:---:|---:|
| A | B | C |
```

## 13. 转义特殊字符

在 Markdown 特殊字符前添加反斜杠 `\`，可以按普通字符显示：

```markdown
\*这不是斜体\*
\# 这不是标题
\[这不是链接\]
```

常见需要转义的字符包括：

```text
\ ` * _ { } [ ] < > ( ) # + - . ! |
```

## 14. 数学公式

数学公式不是基础 CommonMark 的一部分，但 Obsidian、Typora、Jupyter 和许多 Markdown 渲染器支持 LaTeX 语法。

行内公式：

```markdown
交叉熵为 $L=-\log p(y)$。
```

独立公式块：

```markdown
$$
\operatorname{RMS}(x)
=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$
```

常用写法：

| 目标 | LaTeX |
|---|---|
| 下标 | `x_i` |
| 上标 | `x^2` |
| 分数 | `\frac{a}{b}` |
| 求和 | `\sum_{i=1}^{n}` |
| 根号 | `\sqrt{x}` |
| 矩阵乘法 | `QK^\top` |
| 希腊字母 | `\alpha`、`\epsilon` |

## 15. 脚注

部分 Markdown 渲染器支持脚注：

```markdown
RMSNorm 不执行均值中心化。[^rmsnorm]

[^rmsnorm]: RMSNorm 只使用均方根控制向量尺度。
```

## 16. 可折叠内容

这是 HTML 扩展，不是所有渲染器都支持：

```html
<details>
<summary>点击展开</summary>

这里是详细内容。

</details>
```

## 17. Obsidian 常用扩展

以下语法适合 Obsidian，但不属于通用 Markdown。

内部链接：

```markdown
[[另一篇笔记]]
[[另一篇笔记#某个标题]]
[[另一篇笔记|显示名称]]
```

嵌入其他笔记或图片：

```markdown
![[另一篇笔记]]
![[attention.png]]
```

标签：

```markdown
#cs336 #transformer #paper-reading
```

高亮：

```markdown
==重点内容==
```

Callout：

```markdown
> [!note] 笔记
> 这里记录补充信息。

> [!warning] 注意
> 这里记录容易出错的地方。
```

## 18. 推荐的论文笔记结构

```markdown
# 论文标题

## 1. 基本信息

- 作者：
- 会议/年份：
- 链接：

## 2. 研究问题

## 3. 核心方法

## 4. 实验设置

## 5. 主要结果

## 6. 我的理解

## 7. 疑问与局限

## 8. 相关工作
```

## 19. 常见注意事项

- 标题标记、列表标记后通常需要一个空格。
- 不同段落之间留一个空行，兼容性更好。
- 嵌套列表使用一致的缩进。
- 代码块注明语言，便于阅读和语法高亮。
- 图片尽量使用相对路径，移动整个笔记目录时不容易失效。
- 表格、脚注、公式和删除线属于扩展语法，不同渲染器的支持可能不同。
- 文件名尽量简洁稳定，避免频繁修改导致内部链接失效。
