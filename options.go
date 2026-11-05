package gobehav

import (
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"time"

	"github.com/xxjwxc/gobehav/internal/span"
)

// Span 一个标签命中区间，字段与 haitag 现有 tags[] 完全一致，可直接替换。
//
// Start / End 是字符（rune）下标，左闭右开：End = Start + 命中片段长度。
type Span = span.Span

// Result 单条分析结果
type Result struct {
	Orig     string `json:"orig"`     // 原始文本
	Sentence string `json:"sentence"` // 去掉中文间空格后的句子
	Tokens   string `json:"tokens"`   // 分词结果，以 / 分隔
	Tags     []Span `json:"tags"`     // 行为/情绪标签，按 Start 升序
}

// HotRule 调用方热规则：命中 Pattern 的片段直接打上 Tag，先于内置词表生效。
//
// Pattern 使用 Go RE2 语法，不支持 (?<=) (?=) 这类前后向断言。
// Priority 为 0 时取标签表里的优先级，标签表里也没有则记为 4。
type HotRule struct {
	Pattern  string
	Tag      string
	Priority int32
}

// Fallback 兜底打标器：规则层一条标签都没打出来时才会被调用。
//
// gobehav 自身不带任何向量模型，保持纯 Go 零依赖；需要语义兜底时，
// 调用方把 bge-m3 之类向量模型的实现注进来即可（见 bge 子包）。
//
// 实现必须并发安全：Batch 会多协程调用它。返回空切片或 nil 表示放弃兜底。
type Fallback interface {
	Tag(text string) []Span
}

// CallOptions 单次调用选项
type CallOptions struct {
	// Lang 语种，如 cn/en/hi。为空时使用 WithDefaultLang 的值，
	// 再为空则按字符集自动判定：汉字 -> cn，天城文 -> hi，其余 -> en。
	Lang string
	// Hot 调用方热规则，先于内置词表与检索规则执行。
	Hot []HotRule
}

// TagInfo 标签元信息
type TagInfo struct {
	ID        int32 `yaml:"id" json:"id"`
	Priority  int32 `yaml:"p" json:"p"`           // 越小优先级越高
	Exclusive int32 `yaml:"x" json:"x,omitempty"` // 互斥优先级，0 表示不参与互斥
}

// WordInfo 词库条目。Attrs 是子类 1..5，与检索规则里的
// "&词性=子类=属性..." 逐级对应。
type WordInfo struct {
	Word  string
	POS   string
	Attrs []string
}

// Rule 一条检索规则。
//
// Search 支持两种写法，与原规则表一致：
//
//	"关于"                    直接命中这些词（中英文逗号分隔）
//	"&方位词=后"              按词库的词性/子类/属性检索，取满足路径的全部词
//
// 其余字段是额外的约束，留空表示不限制：
//
//	pos     词性标记（v/a/d/p/c/r/q/m/nd/wp），逗号分隔表示任一
//	loc     位置标记：B 句首、E 句尾、I 句中，逗号分隔；带 ! 表示排除
//	ch      语篇标记：F 句中含逗号、Q 疑问、T 感叹、C 陈述；带 ! 表示排除
type Rule struct {
	ID       int    `yaml:"id"`
	Tag      string `yaml:"tag"`
	Category string `yaml:"cat,omitempty"`
	Search   string `yaml:"search,omitempty"`
	Chapter  string `yaml:"ch,omitempty"`
	POS      string `yaml:"pos,omitempty"`
	Location string `yaml:"loc,omitempty"`

	// 上下文约束，键名沿用原规则表的缩写
	Sub  string `yaml:"sub,omitempty"`  // 主语
	FA   string `yaml:"fa,omitempty"`   // 前方出现
	FDA  string `yaml:"fda,omitempty"`  // 前方不出现
	FCA  string `yaml:"fca,omitempty"`  // 前方紧邻
	FDCA string `yaml:"fdca,omitempty"` // 前方不紧邻
	LA   string `yaml:"la,omitempty"`   // 后方出现
	LDA  string `yaml:"lda,omitempty"`  // 后方不出现
	LCA  string `yaml:"lca,omitempty"`  // 后方紧邻
	LDCA string `yaml:"ldca,omitempty"` // 后方不紧邻
}

// RuleSet 一个语种的全部配置，结构与 conf/behav_cn.yaml 一致。
//
// Tags 是标签表（原 tagsPriority），Phrases 是短语到标签的映射（原 shortSentInfo），
// Words 是词库（原 wordsInfo），Rules 是检索规则（原 rules）。
type RuleSet struct {
	Version int                `yaml:"version" json:"version"`
	Lang    string             `yaml:"lang" json:"lang"`
	Tags    map[string]TagInfo `yaml:"tags" json:"tags"`
	Phrases map[string]string  `yaml:"phrases" json:"phrases"`
	Words   []string           `yaml:"words,omitempty" json:"words,omitempty"`
	Rules   []Rule             `yaml:"rules,omitempty" json:"rules,omitempty"`
}

// config 分析器配置
type config struct {
	fsys        fs.FS               // 资产文件系统
	dir         string              // 资产目录
	rulesYAML   []byte              // 包内默认资产内容
	yamlOnly    bool                // 只使用 rulesYAML，不读资产目录
	assetFile   string              // 外置配置文件（给了就以文件为准，并支持热加载）
	watchDir    string              // 外置配置目录（WithAssetDir），支持热加载
	ruleSets    map[string]*RuleSet // 代码内直接指定资产：语种 -> 资产
	noRules     bool                // 关闭检索规则，只做短语匹配
	fallback    Fallback            // 规则层空命中时的兜底
	modelDir    string              // bge-m3 模型目录，空表示自动探测
	noModel     bool                // 不加载语义兜底
	watch       time.Duration       // 资产文件轮询间隔，0 表示不热加载
	defaultLang string              // 默认语种
	logger      *slog.Logger        // 可选日志
}

func defaultConfig() config {
	bs, _ := assets.ReadFile(defaultAsset)

	return config{
		fsys:        assets,
		dir:         assetsDir,
		rulesYAML:   bs,
		defaultLang: "cn",
		watch:       time.Second,
	}
}

// Option 构造选项
type Option func(*config) error

// WithAssetDir 从磁盘目录加载配置，目录内需包含 behav_<lang>.yaml。
//
// 默认每秒检查一次目录内配置的变更并热加载，不重启就生效；
// WithWatchInterval(0) 可关闭。
func WithAssetDir(dir string) Option {
	return func(c *config) error {
		if dir == "" {
			return fmt.Errorf("gobehav: empty asset dir")
		}
		fi, err := os.Stat(dir)
		if err != nil {
			return fmt.Errorf("gobehav: asset dir: %w", err)
		}
		if !fi.IsDir() {
			return fmt.Errorf("gobehav: %s is not a directory", dir)
		}
		c.fsys = os.DirFS(dir)
		c.dir = "."
		c.watchDir = dir
		return nil
	}
}

// WithAssetFS 从任意 fs.FS（embed.FS、zip 包等）加载资产，dir 为空表示根目录。
func WithAssetFS(fsys fs.FS, dir string) Option {
	return func(c *config) error {
		if fsys == nil {
			return fmt.Errorf("gobehav: nil asset fs")
		}
		c.fsys = fsys
		c.dir = dir
		return nil
	}
}

// WithAssetFile 从外置 YAML 文件加载资产，并默认每 1 秒检查一次变更热加载。
// 传了本选项后，包内嵌的资产不再生效。
func WithAssetFile(file string) Option {
	return func(c *config) error {
		if file == "" {
			return fmt.Errorf("gobehav: empty asset file")
		}
		c.assetFile = file
		return nil
	}
}

// WithAssetYAML 直接指定资产的 YAML 内容，不做热加载。
// 传了本选项后，包内嵌的资产不再生效。
func WithAssetYAML(data []byte) Option {
	return func(c *config) error {
		if len(data) == 0 {
			return fmt.Errorf("gobehav: empty asset yaml")
		}
		c.rulesYAML = data
		c.yamlOnly = true
		return nil
	}
}

// WithRuleSet 直接指定某语种的资产对象，适合代码里动态拼装词表。
func WithRuleSet(rs RuleSet) Option {
	return func(c *config) error {
		if rs.Lang == "" {
			return fmt.Errorf("gobehav: rule set without lang")
		}
		if c.ruleSets == nil {
			c.ruleSets = make(map[string]*RuleSet)
		}
		c.ruleSets[rs.Lang] = &rs
		return nil
	}
}

// WithoutRules 关闭检索规则，只保留短语词表匹配。
func WithoutRules() Option {
	return func(c *config) error {
		c.noRules = true
		return nil
	}
}

// WithFallback 设置兜底打标器：只有规则层一条标签都没打出来时才会被调用。
//
// 兜底结果不会再过优先级与互斥，由实现自己决定输出什么。
func WithFallback(f Fallback) Option {
	return func(c *config) error {
		if f == nil {
			return fmt.Errorf("gobehav: nil fallback")
		}
		c.fallback = f
		return nil
	}
}

// WithWatchInterval 设置资产文件轮询间隔，0 表示关闭热加载。
func WithWatchInterval(d time.Duration) Option {
	return func(c *config) error {
		c.watch = d
		return nil
	}
}

// WithModelDir 指定 bge-m3 语义兜底模型目录。
//
// 目录里放四个固定文件即可，不需要再传任何路径：
//
//	encoder.int8.onnx     模型本体
//	tokenizer.json        分词器，必需与模型配套
//	label_prototypes.f32  标签原型向量
//	label_index.json      	原型的行序
//
// onnxruntime 动态库也建议放在这个目录，程序会自己找：
// Linux 认 libonnxruntime.so*，Windows 认 onnxruntime.dll，macOS 认 libonnxruntime*.dylib。
//
// 不传本选项时自动按 exe 同级依次探测 bge_m3、assets/bge_m3、models/bge_m3，
// 都没有就只跑规则层，不报错。
func WithModelDir(dir string) Option {
	return func(c *config) error {
		c.modelDir = dir
		return nil
	}
}

// WithoutModel 关闭语义兜底，只跑规则层（不需要 CGO 编译器）。
func WithoutModel() Option {
	return func(c *config) error {
		c.noModel = true
		return nil
	}
}

// WithDefaultLang 设置默认语种：调用方未传 Lang 时优先使用它。
func WithDefaultLang(lang string) Option {
	return func(c *config) error {
		if lang == "" {
			return fmt.Errorf("gobehav: empty default lang")
		}
		c.defaultLang = lang
		return nil
	}
}

// WithLogger 设置日志输出，用于资产加载/热加载失败的提示。默认不输出任何日志。
func WithLogger(l *slog.Logger) Option {
	return func(c *config) error {
		c.logger = l
		return nil
	}
}
