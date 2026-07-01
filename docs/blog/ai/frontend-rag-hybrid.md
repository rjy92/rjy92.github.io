# 前端 RAG 工程化：混合检索 + 重排序 + 多轮，把"能跑"调成"能用"

> [第一篇](/blog/ai/frontend-rag)把最小 RAG 链路**跑通**、[第二篇](/blog/ai/frontend-rag-pitfalls)把单路向量检索的坑**踩透**。这一篇是真刀真枪的续集：一个"数据治理 AI 助手"，怎么从"能跑"调到"能用"。
> 全程还是 Node.js + 原生 `fetch`，但这次的主角是三块工程化拼图——**混合检索、规则重排序、多轮记忆**。

::: tip 🎯 本文目标
读完你应该能讲清楚三件前两篇没讲过的事：为什么纯向量检索对 `amount` 这种字段名会失灵、`RRF` 凭什么比加权融合好；重排序之后拒答阈值为什么**必须用余弦分而不是重排分**；RAG 多轮的真难点为什么不是"存历史"而是"指代消解"。
:::

---

## 前两篇讲了什么，这篇补什么

这个系列到这里是第三篇了，三篇层层递进：

- **第一篇《[前端也能搞懂 RAG：用 JS 手写一条最小检索增强链路](/blog/ai/frontend-rag)》** —— 手写最小链路：embedding、余弦相似度、迷你向量库、拒答兜底。解决"RAG 是什么、链路长什么样"。
- **第二篇《[前端手写 RAG 踩坑实录](/blog/ai/frontend-rag-pitfalls)》** —— 单路向量检索接上真实文档后的四个坑：切太碎、切太大、连接被重置、高分≠能回答。
- **本篇** —— 前两篇的检索只有"向量"一条腿，且都是单轮问答。做一个真实可用的**元数据治理助手**，这两点都不够，本篇讲怎么补上。

前两篇的基础概念（RAG 原理、余弦相似度、分块策略）本篇不再重复，涉及处直接给出链接。

::: info 🧭 本篇的场景
在有一定数据规模的团队里，字段口径散落在建表注释、需求文档和同事的记忆里。写 SQL 取数、做报表时，一个高频阻塞是搞不清口径：`amount` 含不含税？`order_date` 记的是下单时间还是支付时间？口径不清，算出来的数就是错的。本篇要做的，是把**数据字典（表结构 + 字段口径）**做成一个 RAG 助手，让需要用数的人直接问。文中所有设计，都由这个场景推导而来。
:::

---

## 🧱 第 1 章：元数据的分块——一张表要能"整表查"也能"单字段查"

::: tip 🎯 本章目标
理解"结构化元数据"这种数据，分块矛盾和普通文档不一样在哪。
:::

第二篇讲过一个大原则：**切块要顺着文档的"语义边界"切，块质量是检索的天花板**（不记得的看[踩坑实录·坑1](/blog/ai/frontend-rag-pitfalls)）。那篇的对象是 markdown 技术文档，边界是"标题层级"。

但元数据不是文章，它的边界是**表和字段的结构层级**，而且有个普通文档没有的矛盾：**同一份数据，要同时支持两种粒度的提问**。

| 用户问法 | 需要的检索粒度 |
|---|---|
| "订单表里都有哪些字段？" | **表级**——要召回整张表的概览 |
| "`amount` 是什么意思、含不含税？" | **字段级**——要精准召回单个字段的口径 |

如果只按表切成一大块，问单个字段时，这一大块里相关的只有一行，相似度被整块稀释（正是踩坑实录·坑2 的"切太大"）；如果只按字段切碎，问"整张表有啥"时又拼不出全貌。

本文所有例子都基于同一张订单表 `orders`（含 `amount`、`original_amount`、`order_date` 等字段）。完整表结构见 👉 [orders-schema.json](/orders-schema.json)，下面的代码和检索示例都以它为输入。

**解法是双粒度分块**：同一份表结构，既生成一个"表级 chunk"（表名 + 所有字段清单），又给每个字段生成一个"字段级 chunk"（字段名 + 类型 + 口径说明）。两种块都进库，让检索自己按问题匹配到合适的粒度。

```javascript
// chunk.js —— 一张表拆成「1 个表级块 + N 个字段级块」
function chunkTable(table) {
  const chunks = []

  // 表级块：回答"这张表有哪些字段"
  chunks.push({
    text: `表 ${table.name}（${table.comment}）包含字段：${table.fields.map(f => f.name).join('、')}`,
    metadata: { type: 'table', table: table.name },
  })

  // 字段级块：回答"某个字段是什么意思"
  for (const f of table.fields) {
    chunks.push({
      text: `字段 ${f.name}（${f.type}）：${f.comment}`,
      metadata: { type: 'field', table: table.name, field: f.name },
    })
  }
  return chunks
}
```

注意每个 chunk 都带了 `metadata`（`type` / `table` / `field`）。**这几个结构化字段现在看只是顺手写的，但它们是第 4 章"规则重排序"的全部弹药**——先埋在这。

::: info 🧭 承上启下
块切好了、进了向量库。但接下来第一个真问题就来了：用户问 `amount`，向量检索居然把 `original_amount`、`payment` 也排在前面——对"字段名"这种专有名词，纯向量为什么会犯迷糊？
:::

---

## 🦿 第 2 章：混合检索——给向量补一条"精确匹配"的腿

::: tip 🎯 本章目标
搞懂纯向量对专有名词为什么弱，以及 BM25 + RRF 怎么补上这条腿。
:::

### 现象：问 `amount`，它把 `original_amount` 也端上来

向量检索的长处是"懂语义"，但这恰恰是它对**专有名词**的短处。用户问 `amount` 字段，向量觉得 `original_amount`、`payment`、`total_price` 语义上都挺近，一股脑排进来。可用户要的就是**那个叫 `amount` 的字段**，一个字都不能差。

**向量比的是"意思像不像"，但字段名要的是"字对不对"**。这两件事，得用两种检索。

### 补一条腿：BM25 数关键词

BM25 是搜索引擎用了几十年的经典算法，干的正是向量的反面：**按关键词精确匹配**。`amount` 这种字段名，它一抓一个准。

它内部有个打分公式（词越稀有、命中越多分越高），但**那不是这篇的重点**——公式背了就会，面试也很少细抠。真正决定"能不能用"的，是它的第一步：**分词**，把一句话切成一个个"词"再去匹配。而分词这一步，藏着一个**为字段名特意设计的细节**。

先看最常见的"普通分词"怎么写——按"非字母数字的字符"切开：

```javascript
// 普通分词：遇到字母数字以外的字符就切开
'查 order_date 字段'.toLowerCase().match(/[a-z0-9]+/g)
// 结果：['order', 'date']  ❌ order_date 从下划线处被劈成了两个词
```

问题就在**下划线 `_` 既不是字母也不是数字**，普通规则把它当成了分隔符，于是 `order_date` 被切成 `order` + `date`。这样一来，用户搜完整的 `order_date`，库里已经没有这个词了，BM25 永远匹配不到。

修法很简单：把 `_` 也塞进"算作词的一部分"的字符集里（即 `[a-z0-9_]`）：

```javascript
// bm25.js —— 关键词精确匹配，专治向量搞不定的专有名词
tokenize(text) {
  // [a-z0-9_]+ 把下划线也算进词里 → order_date、user_id 保持完整的一个词
  // 后面的 [一-龥] 是为了兼容中文（一个汉字算一个词）
  return text.toLowerCase().match(/[a-z0-9_]+|[一-龥]/g) ?? []
  // 现在 '查 order_date 字段' → ['查', 'order_date', '字段']  ✅
}
```

::: tip 💡 这一行很值钱
差别只在正则里多了一个 `_`，效果却是天壤之别：用户搜 `order_date` 到底能不能命中，全系于此。**这条规则是"看懂了自己的数据里全是 `snake_case` 字段名"才写得出来的**——面试讲这个细节，比背 BM25 公式打动人得多。
:::

### 两路怎么合：为什么用 RRF，不用加权相加

现在有两路结果：向量的分是 `0~1`（就是[第一篇](/blog/ai/frontend-rag)那个余弦相似度——夹角余弦严格来说是 `[-1, 1]`，但文本 embedding 几乎不会出现方向相反的负值，工程上就当 `0~1` 用），BM25 的分是 `0~十几`。**两把尺子的刻度根本不一样**（一个满分是 1，一个能到十几），你不能直接 `0.5×向量 + 0.5×BM25`——这就像拿"1 米"加"1 公斤"，数值上 BM25 天然大一截，一相加就把向量那点分压得没影了。

那先归一化再加权？也不行——归一化对异常值极敏感，一条特别高的 BM25 分会把其他全压扁。

**RRF（Reciprocal Rank Fusion，倒数排名融合）干脆不看分数、只看排名：**

```
RRF_score(d) = Σ  1 / (k + rank(d))       k 取 60（业界惯例）
```

一条结果在向量里排第 1、在 BM25 里排第 3，它的融合分就是 `1/(60+1) + 1/(60+3)`。**两路都靠前的结果，最终分最高**——这正是我们要的"双保险"。RRF 不需要归一化、不需要调参，是目前最常用的融合法。

```javascript
// hybrid-search.js —— RRF 融合，只看排名位置，不看原始分数
const RRF_K = 60

function rrfFuse(vectorHits, bm25Hits) {
  const map = new Map()   // key = text

  for (const [tag, hits] of [['vector', vectorHits], ['bm25', bm25Hits]]) {
    hits.forEach((r, i) => {
      const rank = i + 1
      const cur = map.get(r.text) ?? {
        text: r.text, metadata: r.metadata, rrfScore: 0, vectorScore: undefined,
      }
      cur.rrfScore += 1 / (RRF_K + rank)         // 累加两路的倒数排名
      if (tag === 'vector') cur.vectorScore = r.score  // ★ 单独把余弦分留住！
      map.set(r.text, cur)
    })
  }
  return [...map.values()].sort((a, b) => b.rrfScore - a.rrfScore)
}
```

注意那行加 ★ 的：**融合时我特意把向量的余弦分 `vectorScore` 单独存了下来**。为什么要留它、留着干嘛用——这是第 3 章整章的伏笔，先记住这里埋了一手。

::: info 🧭 承上启下
混合检索粗召回了 Top-20，两条腿都照顾到了。但"粗召回"里排序还不够精——明明字段名精确命中的，可能没排到最前。我们需要再精排一次，而且这次要动用第 1 章埋下的 `metadata`。
:::

---

## 🎚️ 第 3 章：规则重排序——用 metadata 精排，为什么先不上模型

::: tip 🎯 本章目标
理解重排序在干嘛，以及"先用规则、不上模型"是一个怎样的工程判断。
:::

混合检索给了 20 条候选，但它们的排序还只是"排名融合"的结果。真正"最该看的那条"——比如字段名和用户问的**一字不差**的那条——应该被顶到最前。这一步叫**重排序（rerank）**：粗召回 Top-20 → 精排 Top-5。

一提 rerank，很多人第一反应是上 Cross-Encoder 模型。但我这个场景，**规则就够了，而且更好**——因为第 1 章我给每个 chunk 都写了 `metadata`（`type` / `table` / `field`），这些结构化信息本身就是最硬的排序信号：

```javascript
// rerank.js —— 用 metadata 的结构化信息给候选加分（每条规则返回 0~1，再乘权重）
const RULES = [
  {
    weight: 0.5,   // 字段名精确命中，权重最高
    apply: (query, c) =>
      c.metadata.field && query.toLowerCase().includes(c.metadata.field) ? 1 : 0,
  },
  {
    weight: 0.3,   // 表名命中
    apply: (query, c) =>
      c.metadata.table && query.includes(c.metadata.table) ? 1 : 0,
  },
  {
    weight: 0.1,   // 保留一点混合检索本身的排名信号，兜底
    apply: (query, c) => Math.min(1, (c.rrfScore ?? 0) * 30),
  },
  // …还可以按 type、问法意图继续加规则
]

function rerank(query, candidates, topK = 5) {
  return candidates
    .map(c => ({
      ...c,
      rerankScore: RULES.reduce((s, r) => s + r.weight * r.apply(query, c), 0),
    }))
    .sort((a, b) => b.rerankScore - a.rerankScore)
    .slice(0, topK)
}
```

**为什么这个场景先用规则、不上模型？** 三个理由，全是工程判断：

1. **规则的信号本就存在。** 字段名、表名是结构化的，"精确命中就该排前面"是确定性事实，用规则一句话就表达了，模型反而要"学"这件它本该确定的事。
2. **零成本、可解释、可调权重。** 规则不额外调一次模型（省延迟省钱），每条候选为什么排这个位置一目了然，权重不满意随手调。
3. **Cross-Encoder 是"后手"不是"起手"。** 等规则真的覆盖不了（比如复杂自然语言问法），再上模型精排也不迟。**按需求选复杂度，不是越复杂越显水平**——这条判断会贯穿到本文结尾。

注意最后那条权重 0.1 的规则：万一某条候选一个结构化规则都没命中（模糊问法），还能靠混合检索本身的排名兜底，不至于乱排。

::: info 🧭 承上启下
现在排序很准了。但一个新问题冒出来了，而且是本文最容易踩、最少人讲的坑——**我手上现在有三种分数**（余弦分、RRF 分、重排分），当用户问一个库里根本没有的问题、需要"拒答"时，我该拿**哪个分**去和阈值比？
:::

---

## ⚖️ 第 4 章：重排分 ≠ 余弦分——拒答阈值必须用余弦（全文最硬的一节）

::: tip 🎯 本章目标
理解引入混合检索和重排序后，"拒答判断"为什么会悄悄失灵，以及怎么修。
:::

### 坑：三种分数混在一起，阈值不知道该信谁

第一篇讲过"拒答"：问一个库里没有的问题，所有段落相似度都低、被阈值滤空，助手就老实说"无法回答"（不记得的看[原理篇·第7章](/blog/ai/frontend-rag)）。那时候只有一种分——余弦相似度，阈值 `0.6` 是拿它校准的。

可现在，经过混合检索和重排序，一条结果身上挂了**三种分**：

| 分数 | 量纲 | 能不能当拒答阈值 |
|---|---|---|
| 余弦相似度 `vectorScore` | `0~1`，被校准过 | ✅ 只有它能 |
| RRF 分 `rrfScore` | `1/61` 这种极小的数 | ❌ 跟 0.6 没可比性 |
| 重排分 `rerankScore` | 规则加权，可能 0.7 | ❌ 那是"排序分"不是"相关度" |

**最坑的是重排分**：重排序后排第一的那条，`rerankScore` 可能是 0.7，看着比阈值 0.6 高，好像"有相关内容"。但那 0.7 是"字段名命中规则"给的排序分，**跟"这条内容到底和问题相不相关"是两码事**。如果拿它当阈值，一个库里根本没有的问题也会因为某条规则碰巧命中而"看起来够格"，于是助手不拒答、开始硬编——第一篇辛苦建立的拒答能力，就这么被悄悄废掉了。

### 修：让余弦分一路"幸存"到拒答判断

这就是第 2 章那行 ★ 的用意——**融合时死死把 `vectorScore` 单独带着**，一路传到主流程。拒答判断只认它：

```javascript
// metadata-qa-hybrid.js —— 主流程：先判要不要答，再排怎么答
const SIMILARITY_THRESHOLD = 0.6   // 和基础版一致：这是【余弦】阈值

async function answerWithHybrid(searcher, question) {
  // 1. 混合检索，粗召回 20 条候选（每条都带着 vectorScore）
  const candidates = await searcher.search(question, 20)

  // 2. 拒答判断：用"余弦分最高的那条"看库里到底有没有相关内容
  //    ★ 必须用 vectorScore，不能用 rerankScore / rrfScore —— 量纲不对
  const bestVectorScore = Math.max(...candidates.map(c => c.vectorScore ?? 0))
  if (bestVectorScore < SIMILARITY_THRESHOLD) {
    return { text: '根据现有资料无法回答', hits: [] }   // 库里没有，直接拒答
  }

  // 3. 通过了才重排序，精排 Top-5 喂给 LLM
  const hits = rerank(question, candidates, 5)
  return { text: await chat(question, hits), hits }
}
```

这里还有个顺序讲究：**先判要不要答（余弦阈值），再排怎么答（重排序）**。如果库里压根没相关内容，重排序也是白排——直接拒答更省。

::: danger 🏔️ 这一节的一句话
**引入越多"排序技巧"，就越要守住那个"被校准过的相关度分"。** 混合检索、重排序都是为了让"该排前面的排前面"，但"到底该不该答"这个判断，从头到尾只有余弦分有资格。谁负责排序、谁负责拒答，权责必须分清——这是整章最容易踩的坑，也是最能讲出工程成熟度的地方。
:::

::: info 🧭 承上启下
到这，单轮问答已经又准又稳。但真到运营手里，对话是连着问的：问完"`amount` 是什么"，接一句"**它**含税吗？"——这个"它"，会让前面所有的检索努力瞬间归零。为什么？
:::

---

## 💬 第 5 章：多轮记忆——真难点不是"存历史"，是"指代消解"

::: tip 🎯 本章目标
理解 RAG 多轮为什么比纯聊天多一道坎，以及"问题改写"怎么解。
:::

### 为什么"存历史"解决不了这个问题

一说多轮，很多人以为就是"把历史 messages 一起发给模型"。纯聊天确实够了——模型自己看得到上文，"它"指谁它自己会消解。

但 **RAG 多了一道检索，而检索器是个"瞎子"，它看不到对话历史**。用户第二轮问"它含税吗？"，拿去检索的就是"它含税吗"这四个字。"它"是个代词，**本身没有任何检索价值**，向量化之后只剩"含税"这个弱信号，检索器根本不知道"它"指的是上一轮的 `amount`，于是召回一堆跟当前字段无关的东西，答非所问。

**这才是 RAG 多轮的真难点：指代消解（coreference resolution）。** 存历史解决的是"模型记得上文"，解决不了"检索看得懂代词"。

### 解法：检索之前，先把问题"洗"成自包含的

正确顺序是：**先结合历史把代词还原成具体实体，得到一个独立问题，再去检索。** 这一步叫问题改写（query rewriting）：

```
用户："它含税吗？"
        │  ① 结合历史改写（关键的一步）
        ▼
独立问题："amount 含税吗？"
        │  ② 走第 2~4 章的混合检索 + 重排序（一行没改，原样复用）
        ▼
      答案
        │  ③ 记录这一轮，给下一轮改写用
        ▼
```

改写交给 LLM，不用正则——代词太灵活（"它/这个/那张表/该字段"），正则永远有漏网的。但每句都调一次改写 API 又浪费，所以加个**省钱开关：只有问题里含代词才改写，自包含的问题直接放行**：

```javascript
// rewrite-query.js
const PRONOUN = /它|这个|那个|这张表|那张表|上面|刚才|该字段|这列|上述/

export async function rewriteQuery(question, history) {
  // 没历史、或问题本身就独立 → 直接返回，省一次调用
  if (!history.length || !PRONOUN.test(question)) return question

  const historyText = history.map(t => `用户：${t.question}\n助手：${t.answer}`).join('\n')
  const sys = `根据对话历史，把用户最新问题改写成一个不依赖上下文也能独立理解的问题。
把"它/这个/该字段"等代词替换成历史里它指代的具体字段名或表名。只输出改写后的问题本身。

对话历史：
${historyText}`

  return (await chat(`最新问题：${question}`, [], sys)).trim()
}
```

再配上下文长度的两道闸，防止对话越聊越长撑爆 prompt：**滑动窗口**（只留最近 3 轮）+ **token 上限**（拼历史时超了就丢最早的）。两道闸各管一件事——窗口管"看几轮"，token 管"拼多长"。

```javascript
// conversation.js
const MAX_TURNS = 3           // 滑动窗口：最多看最近 3 轮
const MAX_HISTORY_TOKENS = 800

recentHistory() {
  const recent = this.turns.slice(-MAX_TURNS)
  const kept = []
  let tokens = 0
  for (let i = recent.length - 1; i >= 0; i--) {
    const cost = estimateTokens(recent[i].question) + estimateTokens(recent[i].answer)
    if (tokens + cost > MAX_HISTORY_TOKENS) break   // 再加就超了，停
    kept.unshift(recent[i])
    tokens += cost
  }
  return kept
}
```

最后串起来，多轮的全部新增逻辑只在检索的"前面"（改写）和"后面"（记录）——**第 2~4 章的检索器一行都不用动**：

```javascript
// metadata-qa-multiturn.js
async function answerInConversation(searcher, conversation, question) {
  const history = conversation.recentHistory()
  const standalone = await rewriteQuery(question, history)   // 前：改写
  const result = await answerWithHybrid(searcher, standalone) // 中：复用检索（没改）
  conversation.addTurn(standalone, result.text)               // 后：记录改写后的独立问题
  return { ...result, standalone }
}
```

::: tip 💡 一个防套娃的小心思
记录历史时，存的是**改写后的独立问题**（`standalone`），不是用户原话。否则下一轮改写时，历史里又冒出一个"它"，"它套它"没完没了。存独立问题，让历史天然自包含。
:::

::: info 🧭 承上启下
三块拼图——混合检索、规则重排序、多轮记忆——都装上了。回头看，会发现它们背后是同一种思维方式。
:::

---

## 🎁 结语：工程化的分寸感，是"知道什么时候该停"

::: tip 💎 只记一句话也够
**把 RAG 从"能跑"调到"能用"，靠的不是堆最复杂的技术，而是每一步都问一句"这个场景，到这个程度够不够"。**
:::

回头数一遍这篇里所有的"我没有用更复杂的方案"：

- 融合两路检索，用 **RRF 只看排名**，没上"归一化 + 加权调参"那套；
- 重排序用 **metadata 规则**，没上 Cross-Encoder 模型；
- 多轮的 token 控制，用 **糙估算**（中文 1 字≈1 token），没引 tiktoken；
- 改写加了个**代词开关**，独立问题不白调 API。

每一个选择，都不是"我只会简单的"，而是"这个场景，简单的刚好够、且更可控"。**这恰恰是前两篇那条主线的延续**：第一篇说 RAG 没有魔法、是条你看得懂的数据流；第二篇说跑通只是起点、能拆开看"为什么不准"才算会；到这一篇——**能判断"什么时候该加复杂度、什么时候该收手"，才算把 RAG 做成了工程，而不是堆技术。**

而这套检索器的价值还没榨干：同一个 `searcher`，前面接改写就成了多轮记忆，后面换一段 prompt，还能长出 **Text-to-SQL**（把真实字段塞进 prompt，让 LLM 别猜列名）和**报表解读**（结合字段口径解释数字）——**一套地基，三个能力**。那两块怎么做，留到下一篇。

::: tip 🚀 最后的最后
这三篇代码加起来也就几百行。真正稀缺的从来不是"知道有 RRF、有 rerank"，而是**在自己的场景里，讲得清每一步为什么这么选、为什么不那么选**。那才是面试官眼里，你和"背过 RAG 八股"的人之间的差距。
:::
