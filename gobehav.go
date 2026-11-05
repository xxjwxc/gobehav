// Package gobehav 是行为/情绪打标引擎的纯 Go 实现：输入一段文本，输出字符级的
// 标签区间，用来替代原先跑在 Triton 里的 behavior_emotion_cn 模型
// （pyltp 分词 + Excel 规则引擎），但不依赖 Python、LTP、CGO 或任何模型文件。
//
// 引擎分三层：
//
//	短语层  3183 条短语一次性过 Aho-Corasick，取全部出现位置
//	规则层  232 条检索规则，校验词性 / 位置 / 语篇 / 上下文后打标
//	合并层  优先级、互斥、去嵌套、解交叉
//
// 规则与词表随包内嵌，开箱即用：
//
//	a, err := gobehav.New()
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer a.Close()
//
//	res := a.Analyze("对不起，我真的很抱歉")
//	fmt.Println(res.Sentence) // 对不起,我真的很抱歉
//	for _, t := range res.Tags {
//		fmt.Println(t.Start, t.End, t.Tag, t.Priority)
//	}
//
// 指定语种、批量处理、调用方热规则：
//
//	res = a.AnalyzeLang("I am so happy", "en")
//	out := a.Batch(texts, gobehav.CallOptions{Lang: "cn"})
//	res = a.AnalyzeWith("开心到飞起", gobehav.CallOptions{
//		Hot: []gobehav.HotRule{{Pattern: "飞起", Tag: "mood_happy", Priority: 1}},
//	})
//
// Span 的 Start / End 是字符（rune）下标，左闭右开，与 haitag 现有的
// tags[].start / tags[].end 语义一致，调用方不需要做偏移换算。
//
// Analyzer 并发安全，资产在启动时一次性编译成只读快照；某个语种缺失或
// 某句话没有命中任何标签时返回空结果，不会让调用方拿到半截数据。
package gobehav

import (
	"fmt"
	"log/slog"
	"runtime"
	"sync"
	"sync/atomic"
)

// Analyzer 行为情绪分析器，内部持有只读资产快照，可并发调用。
type Analyzer struct {
	store       *assetStore
	logger      *slog.Logger
	defaultLang string
	fallback    Fallback
	closer      func() error // 释放兜底模型（如果有）
}

// New 创建分析器。默认加载包内嵌资产与语义兜底模型，无需任何参数。
//
// 语义兜底按 exe 同级的 bge_m3 目录自动加载（见 WithModelDir）：
// 目录不在、动态库缺失、或编译时关了 cgo，都只是退化成纯规则，不影响启动。
func New(opts ...Option) (*Analyzer, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}

	if cfg.fallback == nil && !cfg.noModel {
		cfg.fallback = openFallback(&cfg)
	}

	store, err := newAssetStore(&cfg)
	if err != nil {
		return nil, err
	}

	a := &Analyzer{
		store:       store,
		logger:      cfg.logger,
		defaultLang: normLang(cfg.defaultLang),
		fallback:    cfg.fallback,
	}
	if c, ok := cfg.fallback.(interface{ Close() error }); ok {
		a.closer = c.Close
	}

	return a, nil
}

// Langs 返回可用的语种
func (a *Analyzer) Langs() []string {
	set := a.store.Current()
	if set == nil {
		return nil
	}
	out := make([]string, len(set.langs))
	copy(out, set.langs)
	return out
}

// TagTable 返回指定语种的标签表（标签 -> id / 优先级 / 互斥优先级）。
// 语种不存在时返回 nil。
func (a *Analyzer) TagTable(lang string) map[string]TagInfo {
	c := a.pick(a.store.Current(), lang)
	if c == nil {
		return nil
	}
	return c.tags
}

// Analyze 分析单条文本，语种按默认值处理。
func (a *Analyzer) Analyze(text string) Result {
	return a.AnalyzeWith(text, CallOptions{})
}

// AnalyzeLang 分析单条文本并指定语种。
func (a *Analyzer) AnalyzeLang(text, lang string) Result {
	return a.AnalyzeWith(text, CallOptions{Lang: lang})
}

// AnalyzeWith 分析单条文本，可带语种与调用方热规则。
func (a *Analyzer) AnalyzeWith(text string, o CallOptions) Result {
	set := a.store.Current()
	if set == nil {
		return Result{Orig: text, Sentence: cleanSpace(text)}
	}

	return a.analyzeOne(text, o, set)
}

// Batch 批量分析。文本之间互不相关，会按 CPU 核数并行处理。
func (a *Analyzer) Batch(texts []string, o CallOptions) []Result {
	out := make([]Result, len(texts))
	set := a.store.Current()
	if set == nil {
		for i, t := range texts {
			out[i] = Result{Orig: t, Sentence: cleanSpace(t)}
		}
		return out
	}

	workers := parallelWorkers(len(texts))
	if workers <= 1 {
		for i, t := range texts {
			out[i] = a.analyzeOne(t, o, set)
		}
		return out
	}

	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(texts) {
					return
				}
				out[i] = a.analyzeOne(texts[i], o, set)
			}
		}()
	}
	wg.Wait()

	return out
}

// BatchLang 批量分析并指定语种。
func (a *Analyzer) BatchLang(texts []string, lang string) []Result {
	return a.Batch(texts, CallOptions{Lang: lang})
}

// Close 停止资产文件热加载并释放兜底模型。可重复调用。
func (a *Analyzer) Close() error {
	a.store.Close()
	if a.closer != nil {
		return a.closer()
	}

	return nil
}

// logError 只在调用方显式传了 logger 时输出，库自身不打印任何东西。
func (a *Analyzer) logError(msg string, args ...any) {
	if a.logger != nil {
		a.logger.Error(msg, args...)
	}
}

// parallelWorkers 按文本条数与 CPU 核数决定并行度。
func parallelWorkers(n int) int {
	if n <= 1 {
		return n
	}

	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	if workers < 1 {
		workers = 1
	}

	return workers
}

// 包级便捷函数：走全局默认实例（内嵌资产），首次调用时初始化。
// 需要自定义资产、语种或日志时请改用 New。
var (
	defaultOnce sync.Once
	defaultA    *Analyzer
)

// Default 返回全局默认分析器，初始化失败时返回 nil。
func Default() *Analyzer {
	defaultOnce.Do(func() {
		a, err := New()
		if err != nil {
			return
		}
		defaultA = a
	})

	return defaultA
}

// Analyze 用全局默认实例分析单条文本。
func Analyze(text string) Result {
	a := Default()
	if a == nil {
		return Result{Orig: text, Sentence: cleanSpace(text)}
	}
	return a.Analyze(text)
}

// AnalyzeLang 用全局默认实例分析单条文本并指定语种。
func AnalyzeLang(text, lang string) Result {
	a := Default()
	if a == nil {
		return Result{Orig: text, Sentence: cleanSpace(text)}
	}
	return a.AnalyzeLang(text, lang)
}

// Batch 用全局默认实例批量分析。
func Batch(texts []string, o CallOptions) []Result {
	a := Default()
	if a == nil {
		out := make([]Result, len(texts))
		for i, t := range texts {
			out[i] = Result{Orig: t, Sentence: cleanSpace(t)}
		}
		return out
	}
	return a.Batch(texts, o)
}

// MustNew 创建分析器，失败时 panic，便于在 init/包级变量里使用。
func MustNew(opts ...Option) *Analyzer {
	a, err := New(opts...)
	if err != nil {
		panic(fmt.Sprintf("gobehav: %v", err))
	}
	return a
}
