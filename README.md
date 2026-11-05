# gobehav

中文行为/情绪打标的纯 Go 实现：输入一句话，输出带字符区间的行为/情绪标签
（`behavior_apology`、`mood_happy`、`behavior_although`…）。

用来替代原先跑在 Triton 里的 `behavior_emotion_cn` 模型。那条链路是
pyltp（LTP3）分词 + 词性标注 + 依存句法 + Excel 规则引擎，需要 Python 运行时、
模型仓库和 4 个 sheet 的规则表。gobehav 把同一套规则编译成静态资产随包内嵌，只用标准库 + `gopkg.in/yaml.v3`；
不带模型时连 CGO 都不用。

- **开箱即用**：`gobehav.New()` 不需要任何参数。规则与词表随包内嵌，
  程序旁边有 `bge_m3/` 就自动开启语义兜底，没有就纯规则。
- **纯 Go 跑规则**：不依赖 Python / LTP / 模型仓库；不带兜底时也不需要 CGO。
- **可直接替换**：`Span{start,end,tag,priority}` 与原服务 `tags[]` 字段一致，
  `Start/End` 都是字符下标、左闭右开，调用方不用做偏移换算。
- **并发安全**：配置在启动时编译成只读快照，批量调用按 CPU 核数并行。
- **配置可改**：规则放在 `conf/`，指向目录即可热加载，不重编译、不重启，
  见 [配置目录](#配置目录)。
- **可溯源**：配置头部写死导出脚本、源文件指纹、被裁字段，
  见 [基线怎么来的](#基线怎么来的)。
- **语义兜底默认开启**：规则层零命中时用 bge-m3 向量补一个标签，
  模型目录与动态库放一起就行，见 [语义兜底](#语义兜底默认开启)。

内置语种：`cn`。其余 6 个语种（`hi`/`de`/`en`/`es`/`fr`/`it`）的规则表在原仓库里
是空表（只有表头 `句子/含义/标签`），实际干活的是未随仓库提交的 ONNX 情感分类模型，
所以 gobehav 只做中文，细节见[其他语种为什么不在](#其他语种为什么不在)。

## 安装

```bash
go get github.com/xxjwxc/gobehav
```

要求 Go 1.23+。

> 下限被 `bge` 子包的依赖（`sugarme/tokenizer`）拉到了 1.23；只用规则引擎的话
> 老版本 Go 也能跑，但 `go.mod` 里的 go 指令必须是 1.23。

### 接入方要做的两个选择

代码一模一样，区别只在编译方式和运行目录：

| | 纯规则 | 带语义兜底（默认） |
| --- | --- | --- |
| 编译 | `CGO_ENABLED=0 go build`，不要 gcc | `CGO_ENABLED=1`，要 gcc（Windows 用 mingw-w64） |
| 运行目录 | 什么都不用放 | 放一个 `bge_m3/`：模型四个文件 + onnxruntime 库 |
| 规则没命中时 | 返回空标签 | 再问一次向量，阈值 0.62 |
| 二进制 | 8.4 MB | 12.8 MB，外加 618 MB 模型目录 |

`gobehav.New()` 不会因为缺模型报错：找不到 `bge_m3/` 就静默降级成纯规则，
只有「目录在、但库或模型坏了」才打一条 WARN。想显式关掉用 `WithoutModel()`。

模型目录三种给法，任选：

```go
gobehav.New()                             // 自动探测：<exe>/bge_m3、<cwd>/bge_m3
                                          //           以及两者的 assets/、models/ 子目录

gobehav.New(gobehav.WithModelDir(dir))    // 明确指定，适合模型放共享盘

gobehav.New(gobehav.WithoutModel())       // 不要兜底，纯规则
```

## 快速开始

```go
package main

import (
	"fmt"
	"log"

	"github.com/xxjwxc/gobehav"
)

func main() {
	a, err := gobehav.New()
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	res := a.Analyze("对不起，我真的很抱歉")
	fmt.Println(res.Sentence) // 对不起,我真的很抱歉
	fmt.Println(res.Tokens)   // 我/真的/很/抱歉/，/对不起
	for _, t := range res.Tags {
		fmt.Printf("[%d,%d) %s p=%d\n", t.Start, t.End, t.Tag, t.Priority)
	}
	// [0,3) behavior_apology p=1
}
```

一次调用可以用包级函数（内部复用全局默认实例）：

```go
res := gobehav.Analyze("谢谢你，辛苦了")
```

## API

```go
// 构造
a, err := gobehav.New(opts ...Option)   // 默认内嵌资产
a := gobehav.MustNew(opts ...Option)    // 失败 panic，便于包级变量初始化
a := gobehav.Default()                  // 全局默认实例，可能为 nil

// 打标
a.Analyze(text string) Result                       // 默认语种
a.AnalyzeLang(text, lang string) Result             // 指定语种
a.AnalyzeWith(text string, o CallOptions) Result    // 语种 + 调用方热规则
a.Batch(texts []string, o CallOptions) []Result     // 批量，多协程并行
a.BatchLang(texts []string, lang string) []Result   // 批量 + 指定语种

a.Langs() []string                        // 当前可用语种
a.TagTable(lang string) map[string]TagInfo // 标签表：id / 优先级 / 互斥优先级
a.Close() error                           // 停止资产热加载，可重复调用

// 单次调用选项
type CallOptions struct {
	Lang string    // cn/en/hi…；空则用 WithDefaultLang，再空则按字符集判定
	Hot  []HotRule // 先于所有内置词表与检索规则执行的热规则
}

type HotRule struct {
	Pattern  string // Go RE2 语法
	Tag      string
	Priority int32 // 0 表示取标签表里的优先级
}

type Result struct {
	Orig     string // 原始文本
	Sentence string // 去掉中文间空格后的句子
	Tokens   string // 分词结果，以 / 分隔
	Tags     []Span // 按 Start 升序
}

type Span struct {
	Start    int32  // 字符（rune）下标，左闭右开
	End      int32  // End = Start + 命中片段长度
	Tag      string
	Priority int32  // 越小越优先
}
```

`Start/End` 用 rune 下标而不是字节下标：中文在 UTF-8 下占 3 字节，
按字节算 `End` 会偏大，调用方切原文就会切错。

语种自动判定：含汉字 -> `cn`，天城文 -> `hi`，其余 -> `en`。
非 `cn` 没有内置资产，结果里 `Tags` 为空但不会报错。

## 打标流水线

```
清洗空格 -> 分词（词库正向最大匹配） -> 三层匹配 -> 合并
                                       |-- 短语层：3183 条短语过 AC 自动机
                                       |-- 规则层：232 条检索规则 + 上下文约束
                                       |-- 热规则：调用方传入的 RE2
```

- **分词**：直接用 `wordsInfo` 词库做正向最大匹配，词性取自词库，不跑 LTP。
  规则里出现但词库没有的字面词会在编译期补进词库，否则永远命中不了。
- **短语层**：`shortSentInfo` 的短语一次性过 Aho-Corasick，取全部出现位置
  （含重叠），不再逐条 `strings.Contains`。
- **规则层**：`search` 里的 `&词性=子类=属性` 在编译期展开成词集合，
  运行期只做 map 命中；`loc`（句首/句中/句尾）、`ch`（含逗号/疑问/感叹/陈述）、
  9 维上下文（主语、前方出现/紧邻、后方出现/紧邻）在命中处校验。
- **合并**：同区间的多个标签先按 `_S`（短句优先）再按最小优先级择一，
  然后套互斥优先级，最后去嵌套、解交叉。互斥与去嵌套都是**逐句**做的
  （与原实现一致）：整篇文本里某一句的 `behavior_no` 不会把别句的道歉标签吞掉。

某句话没有命中任何标签时返回空 `Tags`，调用方不会拿到半截结果。
想给这类句子补一个语义相近的标签，见下面的[语义兜底](#语义兜底默认开启)。

### 热规则

内置词表覆盖不到的业务词，按调用传进来，不重启、不改资产：

```go
res := a.AnalyzeWith("开心到飞起", gobehav.CallOptions{
	Hot: []gobehav.HotRule{{Pattern: "飞起", Tag: "mood_happy", Priority: 1}},
})
```

热规则命中先于内置词表；同一区间仍按优先级择一，`Priority` 为 0 时回落标签表。

## 配置目录

`conf/` 是配置目录，一个语种一份 `behav_<lang>.yaml`（标签表、短语、词库、
检索规则）。**这份是给人改的**，与 `bge_m3/` 那种机器产物不是一回事：

```
conf/behav_cn.yaml      中文规则，356 KB，可改
bge_m3/                 向量兜底的模型产物，机器生成，别手改（含 onnxruntime 库）
```

改完怎么生效，取决于要不要重新编译：

| 场景 | 写法 | 生效 |
| --- | --- | --- |
| 改源码树里的 conf/ | `WithAssetDir("conf")` | 1 秒内热加载，不重启 |
| 只改某一个文件 | `WithAssetFile("conf/behav_cn.yaml")` | 同上 |
| 部署包只有二进制 | 先 `gobehav.ExportAssets("conf")` 导出一份，再 `WithAssetDir("conf")` | 同上 |

```go
a, err := gobehav.New(
	gobehav.WithAssetDir("conf"),           // 目录内所有 behav_*.yaml
	gobehav.WithWatchInterval(time.Second), // 传 0 关闭热加载
)
```

不传选项时用的是**编译期快照**（也就是打包时 `conf/` 里的内容），零外部依赖、
开箱即用，但改文件不会生效。这两个目标不可能同时自动成立，所以切成显式选项：
默认吃内嵌，想可改就指磁盘。

热加载是轮询：目录模式下同时盯目录内所有 `behav_*.yaml` 的 mtime 与 size，
增删改任意一份都会重载；重载失败保留旧快照并打日志（配 `WithLogger` 才输出），
不会把服务搞挂。

`ExportAssets` 与 `export_assets.py` 都不会覆盖已存在的文件，要重置成基线
才加 `--force`——手改的规则不能被脚本悄悄冲掉。

两种选项的取舍：`WithAssetDir` 能同时管多个语种、新增文件自动生效；
`WithAssetFile` 只认一份，适合“就改这一个文件”的简单部署。

实测（356 KB 的真实中文配置，轮询间隔 50 ms）：首次加载 54 ms；往 `phrases`
里加一条短语到新标签生效 103 ms。重载是全量重建快照，不做增量合并——反正
在后台协程里跑，不阻塞请求。

### 代码里拼装资产

不想用 yaml 就直接在代码里给（`WithAssetFile` 优先于 `WithAssetDir`/`WithAssetFS`；
`WithRuleSet` 指定的语种始终覆盖其他来源）：

```go
a := gobehav.MustNew(gobehav.WithRuleSet(gobehav.RuleSet{
	Lang:    "cn",
	Tags:    map[string]gobehav.TagInfo{"behavior_apology": {ID: 1, Priority: 1}},
	Phrases: map[string]string{"对不起": "behavior_apology"},
	Words:   []string{"对不起|emo"},
	Rules:   []gobehav.Rule{{Tag: "behavior_apology", Search: "对不起"}},
}))
```

也可以只保留短语层：

```go
a := gobehav.MustNew(gobehav.WithoutRules())
```

检索规则字段速查（完整示例见 `conf/behav_cn.yaml`）：

| 字段 | 作用 | 取值 |
| --- | --- | --- |
| `search` | 命中哪些片段 | `词1,词2`，或 `&词性=子类=属性` 走词库检索 |
| `pos` | 词性约束 | `v` 动词 / `a` 形容词 / `d` 副词 / `p` 介词 / `c` 连词 / `r` 代词 / `q` 量词 / `m` 数词 / `nd` 方位词 / `wp` 标点 / `emo` 情感情绪词 |
| `loc` | 位置约束 | `B` 句首 / `I` 句中 / `E` 句尾，逗号分隔表示任一，前缀 `!` 表示排除 |
| `ch` | 语篇约束 | `F` 句中含逗号 / `Q` 疑问 / `T` 感叹 / `C` 陈述，前缀 `!` 表示排除 |
| `sub` | 主语约束 | 词性标记，逗号分隔表示任一 |
| `fa` / `fda` | 前方出现 / 不出现 | 词列表 |
| `fca` / `fdca` | 前方紧邻 / 不紧邻 | 词列表 |
| `la` / `lda` | 后方出现 / 不出现 | 词列表 |
| `lca` / `ldca` | 后方紧邻 / 不紧邻 | 词列表 |

> `search` 里逗号分隔的多项是“任一命中”，与 `pos`、`loc` 的多值口径一致。
> 原实现取多值单元格的首个元素，属于 bug，gobehav 按语义改成 any-of。

## 语义兜底（默认开启）

三层规则是字面匹配：词表没收录、又没触发检索规则的说法会**整句零命中**。
`对不起` 能打上，`这事儿整挺好` 打不上——它一个词表词都没有。

兜底（bge-m3 向量 + 标签原型检索）**默认开启、零配置**：程序旁边放一个
`bge_m3/` 目录，`gobehav.New()` 就自己加载，加载不上就静默退回纯规则。

```
程序目录/
  your-app
  conf/behav_cn.yaml            规则，改完 1 秒生效
  bge_m3/                       语义兜底
    encoder.int8.onnx           模型本体（含 .data 外部权重）
    tokenizer.json              分词器，必须与模型配套
    label_prototypes.f32        标签原型矩阵
    label_index.json            原型的行序
    libonnxruntime.so.*         Linux；Windows 换成 onnxruntime.dll
```

模型目录的探测顺序：`<exe目录>/bge_m3` → `<工作目录>/bge_m3`，
再各试一遍 `assets/bge_m3`、`models/bge_m3`。

换目录或者关掉兜底：

```go
// 模型在别处
a := gobehav.MustNew(gobehav.WithModelDir("/data/models/bge_m3"))

// 只要规则层，不需要 CGO 编译器
b := gobehav.MustNew(gobehav.WithoutModel())
```

编译要求：

- 默认 `CGO_ENABLED=1` 就带兜底（Linux 需 gcc，Windows 需 mingw-w64 的 gcc）
- `CGO_ENABLED=0` 照样能 `go build ./...`，只是兜底不可用（`bge` 换成空壳）
- **构建期不需要 onnxruntime**，只有运行时才 `dlopen` / `LoadLibrary`

不想用 bge 的话，自己实现这个接口也行：

```go
// Fallback 兜底打标器：规则层一条标签都没打出来时才会被调用。
//
// 实现必须并发安全：Batch 会多协程调用它。返回空切片或 nil 表示放弃兜底。
type Fallback interface {
	Tag(text string) []Span
}

a := gobehav.MustNew(gobehav.WithFallback(myFallback))
```

约定很窄，就三条：

1. **只在完全空命中时触发**。只要规则层出一条标签，兜底就不跑——向量推理比
   词典匹配贵三个数量级，不能每条都跑。
2. **结果不再过优先级与互斥**。兜底输出什么就是什么，`Priority` 由实现自己给。
3. **模型不进二进制**。`go:embed` 只收 `conf/behav_*.yaml`，模型按路径读，
   不带 cgo 时也能编过（推理文件带 `//go:build cgo`）。

### onnxruntime 动态库

不用装到系统里，**放模型目录就行**，程序会自己找。来源：

```bash
# Linux：pip wheel 里就带一份 28 MB 的 libonnxruntime.so.1.31.1，
# 只依赖 glibc / libstdc++ / libgcc，拷出来就能用
pip install onnxruntime
cp $(python -c "import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))" \
     )/capi/libonnxruntime.so.1.31.1 bge_m3/
```

```powershell
# Windows：同样来自 pip wheel，先问出包内路径再拷
python -c "import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))"
copy "<site-packages>\onnxruntime\capi\onnxruntime.dll" bge_m3\
```

机器上装的是老版本也没关系，下个新 wheel 解出来就行，不用动现有环境：

```powershell
python -m pip download onnxruntime==1.31.1 --no-deps -d whl
python -c "import zipfile,glob;w=glob.glob('whl/*.whl')[0];open('bge_m3/onnxruntime.dll','wb').write(zipfile.ZipFile(w).read('onnxruntime/capi/onnxruntime.dll'))"
```

版本要求 **1.31 及以上**。`onnxruntime_go` v1.38 一上来就要 ORT C API **31**，
而 ORT 每个 minor 版本只把自己那个号的 API 编进去——1.27.0 会报
`only API versions [1, 27] are supported in this build`，1.17.1 更早。
实测 1.31.1 正常；1.27.0 及以下都过不了初始化。

库的查找顺序：显式 `LibraryPath` → 模型目录 → exe 目录 → 当前目录 →
`/usr/lib`、`/usr/lib/x86_64-linux-gnu`、`/usr/local/lib` 等系统目录 → 平台默认名。

想装到系统里也可以（需要 root，注意软链名是 SONAME）：

```bash
sudo install -m 0755 bge_m3/libonnxruntime.so.1.31.1 /usr/lib/
sudo ln -sf libonnxruntime.so.1.31.1 /usr/lib/libonnxruntime.so.1   # 必须叫 .so.1
sudo ldconfig
ldconfig -p | grep onnxruntime
```

### 端到端实测

| 项 | 结果 |
| --- | --- |
| 分词 vs transformers | 12 条样例的 `input_ids` 逐位相同 |
| 向量 vs Python onnxruntime | 前 16 维最大绝对差 **0**，余弦 1.0 |
| Go 单条（含分词+推理+原型） | 14.47 ms |
| Go 批量 8 条 | 54.52 ms（6.82 ms/条） |
| 二进制体积 | 9.0 MB（不含模型） |

分词用的是 `sugarme/tokenizer` 的零件手工组装的 XLM-R pipeline（它的
`NewTokenizerFromFile` 是个空函数），推理用 `yalue/onnxruntime_go`（运行时
`dlopen` libonnxruntime，构建期不需要库）。

两个部署上的坑：

- **进程内只能初始化一次**，库路径只在第一个 `gobehav.New` 上生效，
  同一个进程里建两个指向不同库的分析器，第二个会继续用第一个的库。
- 机器上同时存在多个版本的 onnxruntime 时（别的项目自带 1.12 / 1.22 很常见），
  不要靠 `LD_LIBRARY_PATH` 碰运气——把要用的那份放进模型目录，它是第一优先级。

> 这里指的是 **CPU 版**库。GPU 版在 `onnxruntime-gpu` 的 wheel 里，
> 还要 CUDA 12 + cuDNN 9；内网没做到那一步，当前只能 CPU 推理。

### 模型产物

`assets/bge_m3/` 下的文件**不会随 `go get` 进到接入方的包里**——主包的
`go:embed` 只匹配 `conf/behav_*.yaml`，`bge` 子包是按路径读文件。所以
要用语义兜底，把它拷成运行目录下的 `bge_m3/`，再把 onnxruntime 库也放进去：

| 产物 | 体积 | 入库 | 说明 |
| --- | --- | --- | --- |
| `assets/bge_m3/encoder.int8.onnx` + `.data` | 568 MB | 否 | INT8 动态量化，fp32 的 0.98 余弦 |
| `assets/bge_m3/tokenizer.json` | 17 MB | 否 | fast tokenizer，XLM-R 词表 |
| `assets/bge_m3/label_prototypes.f32` | 292 KB | 是 | `[73, 1024]` float32，标签原型矩阵 |
| `assets/bge_m3/label_index.json` | 6 KB | 是 | 原型行序 → 标签名 |
| `assets/bge_m3/onnxruntime.dll` | 19 MB | 否 | Windows 运行时库 1.31.1 |
| `assets/bge_m3/libonnxruntime.so.1.31.1` | 29 MB | 否 | Linux 运行时库 1.31.1 |

两个平台的库可以同时躺在同一个目录里，不冲突——`findLibrary` 只按当前平台
的库名 glob（`onnxruntime*.dll` / `libonnxruntime.so.*`），另一份直接忽略。

两份小文件入库是为了跟标签表对得上，但它们同样不被内嵌，`bge.NewFromDir` 照样
要一个路径。三种拿法：

| 场景 | 做法 |
| --- | --- |
| 内网机器 | 直接拷 `~/gobehav_out/bge_m3/` 整个目录成 `bge_m3/` |
| 有 Python + 能拉 pypi | 跑 [`tools/gobehav/export_bge_m3.py`](tools/gobehav/export_bge_m3.py) 重新导出 |
| 只用规则引擎 | 什么都不用准备，`WithoutModel()` 或者不给现模型目录 |

> 别把 `assets/bge_m3/` 塞进你自己项目的 `//go:embed assets`——568 MB 的
> onnx 会被原样编进二进制。gobehav 自己就是踩到这一点，才把 embed 从整个
> `assets` 收窄成 `conf/behav_*.yaml`。

实测（Python 侧，CPU，`max_length=64`）：单条 14.74 ms，8 条批量 56.61 ms
（7.08 ms/条）。

拿 3152 条短语回查自己的标签，top1 命中 64.2%、top3 88.9%。top1 低不是模型差，
是标签表本身粒度重叠（`mood_happy` ↔ `mood_satisfied`、`behavior_no` ↔
`behavior_reject`）——余弦 ≥ 0.90 的短语对里，标签完全不重合的只有 4.7%。
所以兜底只当**弱提示**用。

阈值取 **0.62**：短语上覆盖率 76% / 精度 68.5%；再低就开乱标，抬到 0.70 精度
0.75 但覆盖率掉到 42%。宁可漏、不错标。

对比过短语 kNN（直接用 3152 条短语向量，12.3 MB）：top1 0.647、top3 0.805，
入库的还是原型——体积小 40 倍、top3 反而高 8 个点。详见
[tools/gobehav/README.md](tools/gobehav/README.md#兜底质量)。

> 这样兜底语料只有 73 个标签（3152 条短语聚合出原型），不是全部 140 个——
> 没有短语的标签凑不出原型。缺的标签靠规则层覆盖。

## HTTP 服务（可选）

不想引入 Go 依赖时，可以用 `cmd/gobehav-http` 起一个只用标准库的 JSON 服务：

```bash
go run ./cmd/gobehav-http -addr :8080
# 可选：-asset conf -lang cn -watch 1s -model <目录>|none

curl -s localhost:8080/analyze -d '{"text":["对不起","我今天特别开心"],"lang":"cn"}'
# {"results":[{"orig":"对不起","sentence":"对不起","tokens":"对不起",
#   "tags":[{"start":0,"end":3,"tag":"behavior_apology","priority":1}]}, ...]}
```

带上语义兜底：程序同级建个 `bge_m3/`，把模型四个文件和 onnxruntime 库丢进去，
零参数启动就会自己加载（需 `CGO_ENABLED=1` 构建）：

```bash
CGO_ENABLED=1 go build -o gbhttp ./cmd/gobehav-http
./gbhttp -addr :8080
# 2026/11/05 17:35:15 INFO gobehav: 语义兜底已启用 dir=/app/bge_m3
```

模型在别处就 `-model /data/models/bge_m3`；不想用兜底就 `-model none`。
模型/库有问题也只打一行日志照常启动——兜底不该成为起不来的理由。

请求体：`{"text":["..."],"lang":"cn","hot":[["pattern","tag",1]]}`。
另有 `GET /langs`、`GET /healthz`。

实测（远端 Linux，带兜底，RSS 442 MB）：

```
这事儿整挺好      -> (0,6) behavior_yes       p=100   ← 规则没命中，兜底打的
对不起，我真的很抱歉 -> (0,3)(8,10) behavior_apology p=1  ← 规则打的
我今天特别开心，哈哈哈哈 -> (8,12) behavior_laugh  p=1
又emo了，不想说话  -> (6,8) behavior_no       p=1
```

改 `conf/` 下的配置不用重启。往 `phrases` 里加一条 `"整挺好": behavior_laugh_S`，
3 秒内就生效（1s 轮询 + 1s 内改完），同一句话从兜底的 `behavior_yes p=100`
变成规则的 `behavior_laugh p=1`；改回去又退回兜底。

## 基线怎么来的

`conf/behav_cn.yaml` 是**导出的基线**：它由
[`tools/gobehav/export_assets.py`](tools/gobehav/export_assets.py) 从原 Triton
模型仓库的规则表生成，之后就是你可改的配置。想回到原表口径，重新导出即可
（加 `--force`）：

```
源表  triton_inference_server/models/behavior_emotion/behavior_emotion_cn/
      1/cn_behavior_analysis/data/wordInfoRules.xlsx
产物  conf/behav_cn.yaml
```

四个 sheet 的去向：

| sheet | 产物节点 | 处理 |
| --- | --- | --- |
| `wordsInfo` | `words` | 词 + 词性 + 子类 1..5；中文词性映射为 LTP 标记；子类 6 原实现未使用，丢弃 |
| `shortSentInfo` | `phrases` | 短语 -> 标签列表（保留 `_S` 后缀；同短语多标签按列表保留） |
| `rules` | `rules` | 检索规则；前两行是两级表头，从第 3 行开始读 |
| `tagsPriority` | `tags` | 标签 -> id / 优先级 / 互斥优先级 |

配置头部会把溯源信息写死在里面，可以直接回答"这份配置是从哪来的"：

```yaml
# 由 tools/gobehav/export_assets.py 从 wordInfoRules.xlsx 导出的基线。
# 重新导出：python tools/gobehav/export_assets.py --xlsx <wordInfoRules.xlsx> --force
# conf/ 是配置目录，允许手改；重跑导出会提示，确认后可加 --force 重置。
# 导出时间：2026-11-05 16:01:08
# 源文件：wordInfoRules.xlsx
# 源文件位置：models/behavior_emotion/behavior_emotion_cn/1/cn_behavior_analysis/data/wordInfoRules.xlsx
# 源文件 SHA256：fc3f1868cc2002033bd8ab2b3b05ea27a310586515df718c56b96571e2ad13d1
# 规模：词 7645 / 短语 3183 / 标签 140 / 规则 232
# 未导出的原始列：SRL、SDP（原表整列为空）；DP（依赖 LTP 依存句法，gobehav 无句法模型，含该约束的规则按「忽略 DP」导出）
# 词性口径：原表中文词性已映射为 LTP 标记（动词->v / 副词->d / 介词->p / ...）
```

改规则表或调整映射后重新生成：

```bash
python -m venv .venv
.venv/bin/pip install -r tools/gobehav/requirements.txt   # 只有 openpyxl
.venv/bin/python tools/gobehav/export_assets.py --xlsx <wordInfoRules.xlsx> --force
```

不带 `--xlsx` 时脚本会从当前目录向上逐级找 `../triton_inference_server/...` 下的同名表，
找不到就报错并提示路径。导出后会打印规模与**被丢弃字段清单**。

输出已存在就不写，除非加 `--force`——`conf/` 里的规则可能已被手改，不能
让一次重导悄悄冲掉。

被主动裁掉的原始列（会在导出时列出来，便于以后回填）：

| 列 | 原因 |
| --- | --- |
| `SRL`、`SDP` | 原表整列为空，原实现也只有死配置 |
| `DP` | 需要 LTP 依存句法（SBV/VOB/POB/LAD…），gobehav 不带句法模型，16/232 条规则按"忽略 DP"退化 |

### 其他语种为什么不在

原仓库有 3 个 Triton 模型，但规则表只有中文那份是能用的：

| 模型 | 规则表 | 实际内容 |
| --- | --- | --- |
| `behavior_emotion_cn` | `wordInfoRules.xlsx` | 4 个 sheet 都有数据：词 7645 / 短语 3183 / 标签 140 / 规则 232 |
| `behavior_emotion_hi` | `wordInfoRules_hi.xlsx` | 1 个 sheet，只有表头 `句子/含义/标签`，数据 0 条 |
| `behavior_emotion_multi` | `wordInfoRules_{de,en,es,fr,it}.xlsx` | 同上，数据 0 条；且 `hi`/`en`/`es`/`fr`/`it` 五份文件 MD5 完全相同 |

那几个语种的 Python 实现（`multi_lang_behavior_analysis/main.py`）走的是完全不同
的两条逻辑：短语 `in` 字面匹配（读那张空表，永远打不出标签）+ 一个 ONNX 情感分类
模型打整句情绪。模型文件（`model.onnx` + tokenizer）不在仓库里，线上是另外挂载的。

所以 gobehav 没办法“顺手把多语种也做了”：要么拿到那份 ONNX 模型和一份非空的短语表，
要么这几个语种就维持现状（结果只有空 `Tags`，不会报错）。

## 测试

```bash
go test ./...
go test -run TestAnalyzeCN -v ./   # 打印 9 条真实语句的标签，人工核对语义
```

`internal/matcher` 的单测覆盖 AC 自动机的交错命中（`ushers` 同时命中
`she`/`he`/`hers`）与空模式。

## 目录

```
conf/behav_cn.yaml      配置：内嵌 + 可改，头部有溯源信息
assets/bge_m3/          向量兜底的模型产物（onnx / tokenizer / 标签原型）
assets.go               go:embed 入口 + ExportAssets
conf_test.go            配置目录：导出、热加载、路径校验
gobehav.go              Analyzer 与对外 API
options.go              Option / CallOptions / Span / Result / Rule
analyze.go              单条打标主流程、空格清洗、语篇标记
tokenize.go             基于词库的正向最大匹配分词
rules.go                资产编译：短语索引、规则展开、词性映射、补词
ctx.go                  9 维上下文与位置/语篇校验
postproc.go             合并：优先级、互斥、去嵌套、解交叉
lang.go                 语种归一与字符集判定
internal/matcher/       Aho-Corasick 自动机
cmd/gobehav-http/       可选的 HTTP 服务（标准库）
tools/gobehav/          资产导出脚本与依赖
```

## 已知边界

- 只能做中文。不是导出脚本没写完，而是原仓库里只有 `cn` 那份表有数据：
  `hi`/`de`/`en`/`es`/`fr`/`it` 的 `wordInfoRules_*.xlsx` 只有一行表头
  `句子/含义/标签`，数据行全空（3176 只是格式化残留）；且 `hi`/`en`/`es`/`fr`/`it`
  五份文件 MD5 完全相同。导出脚本支持 `--lang`，但这类表导出来也是空资产。
- 不带 LTP，因此没有句法层面的判定：`DP` 约束被忽略（16/232 条规则），
  分词也不做未登录词识别，词库外的词按单字切分。
- 词性只区分词库给的粗粒度标记，不做上下文消歧。同一个词在不同句子里词性一致。
- 规则 `search` 里的 `&词性=子类=属性` 是在编译期展开的静态集合，
  运行期不会随分词结果动态变化；改词库必须重新导出资产。
- 短语层按字面匹配，大小写敏感，不做同义归并。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。规则与词表资产源自内部
`triton_inference_server` 仓库的 `wordInfoRules.xlsx`，授权随该仓库。
