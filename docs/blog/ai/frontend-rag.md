# 前端也能搞懂 RAG：用 JS 手写一条最小检索增强链路

> 不调框架、不碰向量数据库，只用 100 来行原生 JS，把 RAG 从"听过这个词"变成"我亲手跑通过"。
> 全程 Node.js + 原生 `fetch`，一个硅基流动的免费 API Key 就能复现。

::: tip 🎯 本文目标
读完你应该能：用自己的话讲清 RAG 是什么、解决什么问题；知道一条 RAG 链路由哪几个零件组成、每个零件为什么不可省略；手里有一份能跑、能改、能在面试时打开演示的代码。
:::

---

## 写在前面：这篇文章想帮你解决什么

很多前端同学对 RAG 的认知停留在两句话："不就是把文档喂给大模型嘛""那是后端/算法的活"。

但真到面试被追问"RAG 的检索是怎么做的""为什么不直接 fine-tune""相似度阈值怎么定"，就答不上来了——因为**没亲手拆过**。

这篇文章的目标，是带你用最朴素的方式手写一遍 RAG 的核心链路。每一章只解决一个问题，并且会明确告诉你：**这一章要达成什么、为什么做完这一章会自然引出下一章**。

---

## 💡 第 1 章：RAG 是什么 + 前端为什么值得学

::: tip 🎯 本章目标
建立"为什么需要 RAG"的直觉，而不是背定义。
:::

### 先看一个大模型答不好的问题

直接问模型"我们店的珍珠煮好后能保存多久"，它会给你啰嗦一大段，甚至自相矛盾——一会儿说 2 小时、一会儿说 24 小时、还能编出个"每锅 4 小时冷藏"。

原因很简单：**这是你们店的内部规范，模型训练时根本没见过**。它不是不会说，是没有这个知识，只能猜。这种"一本正经地编"，就是所谓的**幻觉**。

### RAG 干的就是这件事

RAG 全称 Retrieval-Augmented Generation，检索增强生成。拆成大白话：

> **先去你的资料库里"搜"出最相关的几段，把它们塞进 prompt，再让模型"照着资料回答"。**

这里其实藏着一个不简单的问题：**"搜出最相关的几段"，计算机凭什么知道哪段"相关"？** 比如用户问"退货"，论意思最贴近的其实是"退款"，可它俩只共享一个"退"字；而共享了"退货"两个字的"退货流程"，意思反而没那么贴。**字面上重叠多少，根本不等于意思上有多近**——可关键词搜索偏偏只会数字面。

::: warning 🧩 先埋个伏笔
"退款 / 退货 / 退货流程"谁和谁更近，这个问题先记在心里——它是后面整条链路的起点，**第 3 章会专门回收它**。
:::

一句话总结它的本质：

> **RAG 不改模型的脑子，只改喂给模型的那段输入。**

这也是 RAG 和 fine-tune 最关键的区别——fine-tune 是改权重（重新训练），成本高、更新慢；RAG 是改输入（拼上下文），随时换知识库、随时生效。对绝大多数"让模型懂我的私有知识"的需求，RAG 才是性价比之选。

### 前端为什么值得学

三个非常实际的理由：

1. **它本质是一条数据流，不是黑盒算法。** `取文档 → 算向量 → 比相似度 → 拼 prompt → 调接口`，每一步都是你熟悉的"输入输出 + 数组操作 + fetch"。没有梯度、没有反向传播。
2. **它正在变成前端的活。** AI 应用的"检索 + 拼上下文"这层越来越多落在 BFF / Node 层，前端工程师离用户最近，最适合做这层编排。
3. **它是面试高频考点，且容易讲出深度。** 只要你亲手跑过，"高分≠能回答""阈值怎么定""chunk 切多大"这些追问，你都能用自己的实测数据回答。

::: info 🧭 承上启下
现在你知道了 RAG 值得学，但很容易一上来就扎进"Embedding 模型怎么训练""Transformer 怎么推导"里出不来。动手之前，得先回答一个更现实的问题：**前端到底要学到哪一层、学到什么程度才够用？** 先把边界划清楚，再动手才不会迷路。
:::

---

## 📐 第 2 章：应该学到什么程度

::: tip 🎯 本章目标
在动手前先划好边界，避免一头扎进算法细节里出不来。
:::

前端学 RAG，不是要你去训练 embedding 模型、推导 Transformer。你要掌握的是**工程链路**这一层。给个明确的分层：

| 层次 | 要不要深入 | 学到什么程度 |
|------|-----------|-------------|
| Embedding 模型内部怎么训练的 | ❌ 不用 | 知道它"把文本变成一串数字、语义近的数字也近"即可 |
| **调用 embedding 接口、理解它的输入输出** | ✅ 必须 | 能独立调通、知道维度是什么、为什么要分批 |
| **余弦相似度、Top-K、阈值过滤** | ✅ 必须 | 能手写、能解释为什么除以模长、阈值怎么定 |
| **chunk 切分、prompt 注入、拒答兜底** | ✅ 必须 | 踩过坑、能讲出"切太长会稀释相似度"这类实测结论 |
| 向量数据库（Milvus/pgvector）调优 | 🔶 了解 | 知道生产上用它替换"内存数组"，原理是同一套 |

一句话标准：**链路里的每一步你都能手写一个最小版，并解释它为什么存在。** 达到这个程度，框架（LangChain、LlamaIndex）对你就只是"把这些步骤封装了一下"，而不是黑魔法。

::: info 🧭 承上启下
边界划清了——核心就是 Embedding、相似度、链路编排这三块。那我们就从最核心的 Embedding 开始，先把"文本变数字"这一步亲手跑通。
:::

---

## 🔢 第 3 章：Embedding 调用——为什么需要它，以及怎么调

::: tip 🎯 本章目标
理解 embedding 是整条 RAG 的地基，并跑通第一个接口调用。
:::

### 为什么一定要 Embedding？

回到第 1 章留下的问题：用户问"退货"，计算机怎么知道"退款"才是意思最近的那个？

- **关键词匹配**会失败：它只会数字面重叠。"退货流程"和"退货"共享两个字、"退款"只共享一个字，按字面排序"退货流程"会被排在前面——可论意思"退款"才更贴。**字面多 ≠ 语义近**，关键词搜索从根上就抓错了维度。
- **Embedding 不会**：它把每段文本映射成一个高维向量（这里是 1024 维），**语义相近的文本，向量在空间里也靠得近**。

所以 embedding 的作用，就是把"语义相关"这个模糊的人类概念，翻译成"向量距离"这个计算机能算的数字。

::: tip 💡 一句话抓住本质
**没有 embedding 这一步，后面所有的"检索"都无从谈起**——这就是它是整条链路地基的原因。
:::

### 怎么调（真实可跑的代码）

我用的是硅基流动的 `BAAI/bge-m3` 模型，OpenAI 兼容接口，原生 fetch 就能调：

```javascript
// embed.js
import 'dotenv/config'
const API_KEY = process.env.SILICONFLOW_API_KEY
const EMBED_URL = 'https://api.siliconflow.cn/v1/embeddings'

const BATCH_SIZE = 16   // 每批最多发多少条，避免单请求体过大被服务端重置连接

// 发一批（带一次重试，应对 ECONNRESET 等瞬时网络错误）
async function embedBatch(input, retry = 1) {
  try {
    const res = await fetch(EMBED_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model: 'BAAI/bge-m3', input }),
    })
    if (!res.ok) throw new Error(`Embedding 失败: ${res.status} ${await res.text()}`)
    const data = await res.json()
    return data.data.map(item => item.embedding)
  } catch (err) {
    if (retry > 0) {
      await new Promise(r => setTimeout(r, 1000))
      return embedBatch(input, retry - 1)
    }
    throw err
  }
}

// 把一段或多段文本转成向量（自动分批）
export async function embed(texts) {
  const input = Array.isArray(texts) ? texts : [texts]
  const out = []
  for (let i = 0; i < input.length; i += BATCH_SIZE) {
    const vecs = await embedBatch(input.slice(i, i + BATCH_SIZE))
    out.push(...vecs)
  }
  return out
}

// 直接 `node embed.js` 时跑个小测试（被 import 时不执行）
if (import.meta.url === `file://${process.argv[1]}`) {
  const vecs = await embed(['hello'])
  console.log('维度:', vecs[0].length)        // 应该是 1024
  console.log('前 5 个数:', vecs[0].slice(0, 5))
}
```

跑一下 `node embed.js`，你会看到：

```
维度: 1024
前 5 个数: [ -0.013, 0.042, -0.006, 0.038, 0.011 ]
```

**这一串 1024 个数字，就是这段文本的"语义坐标"。** 这是整篇文章最关键的一步——理解了这串数字，RAG 就不再神秘。

::: warning ⚠️ 踩坑预告
这里的 `BATCH_SIZE = 16` 和重试不是凑数的。我第一次把 30 段较长的文本一次性发出去，body 太大直接 `ECONNRESET` 连接被重置。分批 + 重试是踩坑后加的——这段故事我放在另一篇《踩坑实录》里展开。
:::

::: info 🧭 承上启下
现在每段文本都有了自己的 1024 维向量。但两个向量摆在面前，怎么判断它俩"近不近"？我们需要一把量"语义距离"的尺子。
:::

---

## 📊 第 4 章：余弦相似度——给"像不像"一个分数

::: tip 🎯 本章目标
手写一把度量语义距离的尺子，理解它为什么长这样。
:::

两个向量像不像，最常用的是**余弦相似度**：算它俩夹角的余弦值，范围 [-1, 1]，越接近 1 越像。

先把公式摆出来，其实就一行：

```
余弦相似度 = 点积 ÷ (a 的模长 × b 的模长)
          = (a·b) / (|a| × |b|)
```

拆开两个名词就全懂了：

- **点积**：对应位置相乘再求和，`a[0]*b[0] + a[1]*b[1] + …`；
- **模长**：向量各元素平方和再开根号，`√(a[0]² + a[1]² + …)`，几何上就是这个向量的"长度"。

对着这行公式看代码，每一项都一一对得上：

```javascript
// similarity.js
export function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) throw new Error('向量长度不一致')

  let dotProduct = 0, normA = 0, normB = 0
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i]   // 点积
    normA += vecA[i] * vecA[i]        // A 的模长平方
    normB += vecB[i] * vecB[i]        // B 的模长平方
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}
```

**为什么要除以两个模长？** 这是公式的灵魂。点积 `a·b` 其实等于 `|a| × |b| × cos(θ)`——方向信息（夹角 θ）和长度信息是**乘在一起**的。除以两个模长，正好把长度"约掉"，只剩纯粹的 `cos(θ)`。这一步等价于：先把两个向量都归一化成单位长度，再做点积。

::: details 🤔 为什么是余弦，而不是点积或欧氏距离？（面试高频追问，点开看）
一句话：**我们要比的是"语义方向"，不是"向量长度"。**

| 度量 | 在比什么 | 为什么 embedding 不爱用 |
|------|---------|----------------------|
| 点积 | 方向 + 长度 | 长度会干扰：长向量哪怕方向偏，分也可能虚高 |
| 欧氏距离 | 两点的直线距离 | 同样受长度影响，高维下还容易"距离都差不多"（维度灾难） |
| **余弦相似度** | 只比方向（夹角） | ✅ 天然剔除长度，只留语义 |

关键在于：**在 embedding 里，"长度"几乎不携带语义，"方向"才携带语义。** 举个秒懂的例子——"退货" 和 "我想申请退货，麻烦了"，后者更长、模长更大，但方向都指向同一片"售后语义区"。用余弦，两句因方向一致照样判高分；用点积或欧氏，长的那句就会被"长度"带偏。
:::

这样我们比的就是纯粹的"语义朝向"，不被文本长短干扰。

来看一组我实测的数据（基准句"退货"）：

| 候选句 | 分数 | 说明 |
|-------|------|------|
| 退款 | 0.9322 | 只共享"退"字，语义却最近，分数最高 |
| 退货流程 | 0.9165 | 共享"退货"两个字，分数反而略低 |
| 苹果手机 | 0.5152 | 完全无关 |

**"退货流程"明明比"退款"多共享一个字，分数反而更低**——字面重叠骗不了 embedding，它比的是意思、不是字。这就是第 3 章那串数字的威力，也是 RAG 比关键词搜索强的根本原因。

::: info 🧭 承上启下
现在我们有了"文本变向量"（第 3 章）和"向量算相似度"（第 4 章）两个零件。把它俩组装起来，就能做一件正经事：给一个问题，从一堆文档里捞出最相关的几段。这就是"迷你向量库"。
:::

---

## 🗄️ 第 5 章：迷你向量库——把零件组装成"可检索"

::: tip 🎯 本章目标
用一个数组 + 两个零件，搭出 RAG 里"检索"这一环。
:::

所谓向量数据库，剥开看，最小内核就是：**存的时候把每段文本连同它的向量存起来；搜的时候把问题也转成向量，逐个算相似度，排序取前几名。**

```javascript
// store.js
import { embed } from './embed.js'
import { cosineSimilarity } from './similarity.js'

export class MiniVectorStore {
  constructor() {
    this.items = []   // 每项: { text, vector }
  }

  // 批量存入文档片段
  async add(texts) {
    const vectors = await embed(texts)
    texts.forEach((text, i) => this.items.push({ text, vector: vectors[i] }))
    console.log(`已存入 ${texts.length} 段，库内共 ${this.items.length} 段`)
  }

  // 语义检索：返回最相关的 topK 段；threshold 以下的直接过滤
  async search(query, topK = 3, threshold = 0) {
    const [queryVec] = await embed(query)
    return this.items
      .map(item => ({ text: item.text, score: cosineSimilarity(queryVec, item.vector) }))
      .sort((a, b) => b.score - a.score)   // 按相似度从高到低
      .filter(item => item.score >= threshold)  // 先按阈值过滤
      .slice(0, topK)                           // 再取前 K 个
  }
}
```

就这么点代码。生产环境用 Milvus、pgvector，无非是把"内存数组 + 暴力遍历"换成"专门的索引结构"，让百万级数据也能毫秒检索——**原理和你这 30 行一模一样**。理解了这个最小版，向量数据库对你就不再是黑盒。

::: warning 🔑 注意这个 `threshold`
它是用来**拒答**的——问一个库里根本没有的问题，所有段落分数都很低，被阈值滤光，检索结果为空。这个伏笔，**第 7 章会用到**。
:::

::: info 🧭 承上启下
"检索"这一半通了。但 RAG 叫"检索增强生成"，还差"生成"——把检索到的资料塞进 prompt，让模型照着回答。把这最后一棒接上，链路就闭环了。
:::

---

## 🤖 第 6 章：拼接 prompt + 调用模型——闭合整条链路

::: tip 🎯 本章目标
把检索结果注入 prompt，跑通从"问题"到"带出处的回答"的完整流程。
:::

检索到的几段资料，要"喂"给模型。关键全在 system prompt 的两条约束：

```javascript
// chat.js
export async function chat(question, contexts) {
  const context = contexts.map((c, i) => `[资料${i + 1}] ${c.text}`).join('\n')

  const systemPrompt = `你是一个严谨的客服助手。请只根据下面提供的资料回答用户问题。
  如果资料里没有相关信息，直接说"根据现有资料无法回答"，不要编造。
  回答时如果用到了某条资料，标注它的编号。

  可用资料：
  ${context}`

  const res = await fetch('https://api.siliconflow.cn/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.SILICONFLOW_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'deepseek-ai/DeepSeek-V3',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question },
      ],
      temperature: 0.3,   // 低温度，减少自由发挥
    }),
  })
  const data = await res.json()
  return data.choices[0].message.content
}
```

两条约束是 RAG 不幻觉的命门：

1. **"只根据资料回答"** —— 把模型从"什么都敢答"框回"照着材料答"；
2. **"没有就说无法回答"** —— 给它一条体面的退路，宁可拒答，不要编。

再加上 `temperature: 0.3` 压低自由发挥。把三个文件串起来，就是完整的 `rag.js`：

```javascript
// rag.js
import { MiniVectorStore } from './store.js'
import { docs } from './knowledge.js'
import { chat } from './chat.js'

const store = new MiniVectorStore()
await store.add(docs)                              // 1. 建库

const question = process.argv.slice(2).join(' ') || '退货要多久能拿到钱'
const hits = await store.search(question, 5)       // 2. 检索

console.log('\n检索到的资料：')
hits.forEach(h => console.log(`  ${h.score.toFixed(3)}  ${h.text}`))

console.log('\nAI 回答：')
console.log(await chat(question, hits))            // 3. 生成
```

`建库 → 检索 → 生成`，三行注释就是 RAG 的全貌。跑 `node rag.js "退货要多久能拿到钱"`，你会看到它先打印命中的资料分数，再给出一段带 `[资料N]` 出处的回答。

::: info 🧭 承上启下
链路通了，但"能跑"不等于"靠谱"。RAG 真正的难点不在拼接，而在**检索质量**——搜错了，后面再好的模型也白搭。最后一章，我们用实验逼问这条链路的边界。
:::

---

## 🚧 第 7 章：边界与陷阱——RAG 不是银弹

::: tip 🎯 本章目标
通过对比实验，认清 RAG 的能力边界，这也是面试最能讲出深度的地方。
:::

我用 `compare.js` 把"直接问模型"和"走 RAG"并排打印，拿三类问题做对照：

| 问题类型 | 直接问模型 | 走 RAG | 结论 |
|---------|-----------|--------|------|
| ① 私有知识：珍珠保存多久 | 自相矛盾、编数字 | 精准答"4 小时"并标 `[资料1]` | ✅ **RAG 完胜** |
| ② 库里没有：怎么修电脑风扇 | 一本正经编一篇教程 | 检索为空，老实拒答 | ✅ **RAG 更安全** |
| ③ 公开常识：咖啡因的影响 | 全面准确 | 被文档边界限制，答得更窄甚至拒答 | ❌ **RAG 反而更差** |

三条能直接写进简历的认知：

1. **私有知识是 RAG 的主场。** 注入私有知识 + 可溯源（标出处），这两个核心价值在问①同时体现。
2. **"不懂就拒答"是 RAG 治幻觉的命门。** 问②里直接问会硬编教程，RAG 因为检索为空 + prompt 约束老实认怂——这正是第 5 章那个 `threshold` 和第 6 章那两条约束**共同促成**的。
3. **RAG 不是万能，常识题反成短板。** 问③戳破"RAG 一定更好"的错觉——能主动说出这条，比只夸优点成熟得多。

还有一个更隐蔽的陷阱，叫**"高分 ≠ 能回答"**。基准句"怎么退货"：

| 候选句 | 分数 | 是不是答案 |
|-------|------|-----------|
| 退货政策是什么 | 0.8081 | ✅ 是 |
| 怎么换货 | 0.8051 | ❌ 同领域但不同事 |

"换货"和真答案只差 **0.003**，但换货 ≠ 退货。这意味着**光靠相似度排序会把干扰项也召回，阈值还特别难一刀切**。

::: danger 🏔️ 一句话收尾
**检索质量是 RAG 的天花板。** 模型再强，喂错了料也救不回来。这也是为什么前端做 RAG，功夫不在调模型，而在"把对的资料、干净地、按合适的粒度搜出来"。
:::

---

## 🎁 结语：RAG 没有魔法，它只是一条你能看懂的数据流

::: tip 💎 只记一句话也够
**RAG 不是什么算法黑科技，它是一条前端完全 hold 得住的数据流——`把对的资料、按合适的粒度、干净地搜出来，再让模型照着答`。**
:::

回头看这一路：第 3 章把文本变成向量，第 4 章用余弦给"像不像"打分，第 5 章把这两步组装成能检索的库，第 6 章拼好 prompt 闭合链路。**没有一步是黑盒，每一步都是你早就会的 fetch、数组和数学。** 所谓的框架（LangChain、LlamaIndex），无非是把这几步包了一层——你现在拆开看过里面，它们对你就不再神秘。

而真正分高下的地方，第 7 章已经点破：**检索质量是 RAG 的天花板。** 模型再强，喂错了料也救不回来；相似度再高，也可能是"换货 ≠ 退货"那种答非所问。这就是为什么前端做 RAG，功夫不在调模型参数，而在"切块、阈值、拼接"这些把资料伺候干净的工程活——**而这些，恰恰是离用户最近的前端最该接住的一层。**

至于这些工程活具体怎么踩坑（chunk 切太碎检索就废、ECONNRESET 怎么扛、阈值高 0.1 低 0.1 差在哪），我把真刀真枪的过程写在了下一篇《前端手写 RAG 踩坑实录》。原理篇让你看懂链路，踩坑篇让你扛得住生产。

::: tip 🚀 最后的最后
代码总共一百来行，**别只读，自己敲一遍跑起来。** 当你亲眼看到"退款"和"退货"靠语义而非字面被打出 0.93 的高分时，这条链路才真正长在你脑子里。
:::
