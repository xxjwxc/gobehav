package gobehav

import (
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/xxjwxc/gobehav/internal/matcher"
	"gopkg.in/yaml.v3"
)

// ------------------------------------------------------------------ 编译产物

// compiled 一个语种编译后的只读资产，整体替换以保证并发安全。
type compiled struct {
	version int
	lang    string
	tags    map[string]TagInfo
	dict    *tokenizer
	phrase  *phraseIndex
	rules   []compiledRule
}

// phraseIndex 短语词表。3152 条短语一次性过 Aho-Corasick，
// 取代原实现「逐条 strings.Contains」的全量扫描。
type phraseIndex struct {
	ac   *matcher.Automaton
	tags [][]string
}

// compiledRule 一条编译后的检索规则。
//
// Search 里的词表检索（&词性=子类=属性）在编译期就展开成具体词集合，
// 运行期只剩一次 map 命中判断，不涉及任何遍历。
type compiledRule struct {
	tag     string
	words   map[string]struct{} // nil 表示不限定命中词
	pos     []string
	loc     []string // 支持 ! 前缀取反
	chapter []string
	ctx     ctxSet
}

// ctxSet 一条规则的上下文约束，nil 表示该维度不限制。
type ctxSet struct {
	sub, fa, fda, fca, fdca, la, lda, lca, ldca *ctxExpr
}

func (c *ctxSet) empty() bool {
	return c.sub == nil && c.fa == nil && c.fda == nil && c.fca == nil &&
		c.fdca == nil && c.la == nil && c.lda == nil && c.lca == nil && c.ldca == nil
}

// ctxOp 上下文表达式的类型
type ctxOp uint8

const (
	ctxWords   ctxOp = iota + 1 // 字面词集合
	ctxPOS                      // 词性集合
	ctxWordPOS                  // #词+词性
	ctxRepeat                   // *重叠
)

// ctxExpr 编译后的上下文表达式
type ctxExpr struct {
	op        ctxOp
	words     map[string]struct{}
	pos       []string
	wordPart  string
	posPart   string
	wordFirst bool
}

// ------------------------------------------------------------------ 资产容器

// assetSet 一次加载得到的全部语种资产，整体替换保证并发安全。
type assetSet struct {
	byLang map[string]*compiled
	langs  []string
}

func (a *assetSet) get(lang string) *compiled {
	if a == nil {
		return nil
	}
	if c, ok := a.byLang[lang]; ok {
		return c
	}
	return nil
}

// assetStore 配置容器：持有当前快照，可选对外置配置做轮询热加载。
type assetStore struct {
	file   string // 单文件模式：直接的 yaml 路径
	dir    string // 目录模式：目录内所有 behav_*.yaml
	every  time.Duration
	logger *slog.Logger
	cur    atomic.Pointer[assetSet]

	stop      chan struct{}
	done      chan struct{}
	closeOnce sync.Once
}

func newAssetStore(c *config) (*assetStore, error) {
	s := &assetStore{logger: c.logger}

	var (
		set *assetSet
		err error
	)
	switch {
	case c.assetFile != "":
		s.file, s.every = c.assetFile, c.watch
		set, err = loadAssetFile(s.file, c)
	case c.watchDir != "":
		s.dir, s.every = c.watchDir, c.watch
		set, err = loadAssetFS(c)
	default:
		set, err = loadAssetFS(c)
	}
	if err != nil {
		return nil, err
	}
	if len(set.byLang) == 0 {
		return nil, fmt.Errorf("gobehav: no asset loaded")
	}

	s.cur.Store(set)
	if (s.file != "" || s.dir != "") && s.every > 0 {
		s.startWatch(c)
	}

	return s, nil
}

// Current 取当前生效的资产快照
func (s *assetStore) Current() *assetSet { return s.cur.Load() }

// Close 停止热加载协程
func (s *assetStore) Close() {
	if s == nil {
		return
	}
	s.closeOnce.Do(func() {
		if s.stop == nil {
			return
		}
		close(s.stop)
		<-s.done
	})
}

func (s *assetStore) startWatch(c *config) {
	s.stop = make(chan struct{})
	s.done = make(chan struct{})

	last := s.watchSig()

	go func() {
		defer close(s.done)

		t := time.NewTicker(s.every)
		defer t.Stop()

		for {
			select {
			case <-s.stop:
				return
			case <-t.C:
				sig := s.watchSig()
				if sig == "" || sig == last {
					continue
				}
				last = sig

				var (
					set *assetSet
					err error
				)
				if s.file != "" {
					set, err = loadAssetFile(s.file, c)
				} else {
					set, err = loadAssetFS(c)
				}
				if err != nil {
					s.logError("gobehav: reload asset fail, keep old", err)
					continue
				}
				s.cur.Store(set)
				s.logInfo("gobehav: asset reloaded", "langs", set.langs)
			}
		}
	}()
}

// watchSig 取配置源的时间指纹：单文件看 mtime+size，目录看内所有
// behav_*.yaml 的 mtime+size。增删改任意一份都会变，变了就重载。
func (s *assetStore) watchSig() string {
	var b strings.Builder
	add := func(name string, fi os.FileInfo) {
		fmt.Fprintf(&b, "%s:%d:%d;", name, fi.ModTime().UnixNano(), fi.Size())
	}

	if s.file != "" {
		if fi, err := os.Stat(s.file); err == nil {
			add(filepath.Base(s.file), fi)
		}
		return b.String()
	}

	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, assetPrefix) || !strings.HasSuffix(name, assetSuffix) {
			continue
		}
		if fi, err := e.Info(); err == nil {
			add(name, fi)
		}
	}
	return b.String()
}

func (s *assetStore) logInfo(msg string, args ...any) {
	if s.logger != nil {
		s.logger.Info(msg, args...)
	}
}

func (s *assetStore) logError(msg string, err error) {
	if s.logger != nil {
		s.logger.Error(msg, "err", err)
	}
}

// loadAssetFile 从单个 YAML 文件加载资产，语种由文件内容决定。
func loadAssetFile(file string, c *config) (*assetSet, error) {
	bs, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("gobehav: read asset: %w", err)
	}

	ac, err := parseAsset(bs, c)
	if err != nil {
		return nil, err
	}

	set := &assetSet{byLang: map[string]*compiled{ac.lang: ac}, langs: []string{ac.lang}}
	return set, nil
}

// loadAssetFS 从资产文件系统加载全部 behav_<lang>.yaml，再用代码内指定的资产覆盖。
func loadAssetFS(c *config) (*assetSet, error) {
	set := &assetSet{byLang: make(map[string]*compiled)}

	if c.yamlOnly {
		ac, err := parseAsset(c.rulesYAML, c)
		if err != nil {
			return nil, err
		}
		set.byLang[ac.lang] = ac
	} else {
		entries, err := fs.ReadDir(c.fsys, c.dir)
		if err != nil {
			return nil, fmt.Errorf("gobehav: read asset dir: %w", err)
		}
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() || !strings.HasPrefix(name, assetPrefix) || !strings.HasSuffix(name, assetSuffix) {
				continue
			}
			bs, err := fs.ReadFile(c.fsys, path.Join(c.dir, name))
			if err != nil {
				return nil, fmt.Errorf("gobehav: read %s: %w", name, err)
			}
			ac, err := parseAsset(bs, c)
			if err != nil {
				return nil, fmt.Errorf("gobehav: parse %s: %w", name, err)
			}
			set.byLang[ac.lang] = ac
		}
	}

	for lang, rs := range c.ruleSets {
		set.byLang[lang] = compileRuleSet(rs, c)
	}

	for lang := range set.byLang {
		set.langs = append(set.langs, lang)
	}
	sort.Strings(set.langs)
	if len(set.langs) == 0 {
		return nil, fmt.Errorf("gobehav: no asset found in %s", c.dir)
	}

	return set, nil
}

// parseAsset 解析一份资产 YAML 并编译。
func parseAsset(data []byte, c *config) (*compiled, error) {
	var rs RuleSet
	if err := yaml.Unmarshal(data, &rs); err != nil {
		return nil, fmt.Errorf("gobehav: parse asset: %w", err)
	}

	return compileRuleSet(&rs, c), nil
}

// compileRuleSet 把一份资产编译成运行期结构。
func compileRuleSet(rs *RuleSet, c *config) *compiled {
	words := parseWords(rs.Words)

	// 字面检索词里有一批词库没有收录（数字、ok、位于…）。
	// 补进词库后它们才能被分词成独立 token，否则相关规则永远命中不了。
	words = appendMissingWords(words, rs.Rules)

	idx := newWordIndex(words)

	rules := make([]compiledRule, 0, len(rs.Rules))
	if !c.noRules {
		for _, r := range rs.Rules {
			cr, err := compileRule(r, idx)
			if err != nil {
				if c.logger != nil {
					c.logger.Error("gobehav: bad rule, skipped", "id", r.ID, "tag", r.Tag, "err", err)
				}
				continue
			}
			rules = append(rules, cr)
		}
	}

	lang := rs.Lang
	if lang == "" {
		lang = "cn"
	}

	return &compiled{
		version: rs.Version,
		lang:    lang,
		tags:    rs.Tags,
		dict:    newTokenizer(words),
		phrase:  newPhraseIndex(rs.Phrases),
		rules:   rules,
	}
}

// compileRule 编译单条规则。
func compileRule(r Rule, idx *wordIndex) (compiledRule, error) {
	cr := compiledRule{
		tag:     r.Tag,
		pos:     splitList(r.POS),
		loc:     splitList(r.Location),
		chapter: splitList(r.Chapter),
	}
	if cr.tag == "" {
		return cr, fmt.Errorf("empty tag")
	}

	words, err := resolveSearch(r.Search, idx)
	if err != nil {
		return cr, err
	}
	cr.words = words

	ctx := &cr.ctx
	var e *ctxExpr
	for name, raw := range map[string]string{
		"sub": r.Sub, "fa": r.FA, "fda": r.FDA, "fca": r.FCA, "fdca": r.FDCA,
		"la": r.LA, "lda": r.LDA, "lca": r.LCA, "ldca": r.LDCA,
	} {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		e, err = compileCtx(raw, idx)
		if err != nil {
			return cr, fmt.Errorf("%s: %w", name, err)
		}
		switch name {
		case "sub":
			ctx.sub = e
		case "fa":
			ctx.fa = e
		case "fda":
			ctx.fda = e
		case "fca":
			ctx.fca = e
		case "fdca":
			ctx.fdca = e
		case "la":
			ctx.la = e
		case "lda":
			ctx.lda = e
		case "lca":
			ctx.lca = e
		case "ldca":
			ctx.ldca = e
		}
	}

	return cr, nil
}

// resolveSearch 把检索规则展开成具体词集合。
//
//	""                      -> nil，表示不限定命中词（每个 token 都是候选）
//	"关于，之后"             -> 字面词集合
//	"&方位词=后"             -> 词库里该路径下的全部词
func resolveSearch(search string, idx *wordIndex) (map[string]struct{}, error) {
	s := strings.TrimSpace(search)
	if s == "" {
		return nil, nil
	}
	if strings.HasPrefix(s, "&") {
		return lookupAttr(s[1:], idx)
	}

	set := make(map[string]struct{})
	for _, w := range splitList(s) {
		set[w] = struct{}{}
	}

	return set, nil
}

// lookupAttr 按「词性=子类1=子类2…」检索词库，逐级可以写多个候选值（逗号分隔）。
func lookupAttr(expr string, idx *wordIndex) (map[string]struct{}, error) {
	levels := strings.Split(expr, "=")
	alts := make([][]string, 0, len(levels))
	for _, lv := range levels {
		alts = append(alts, splitList(lv))
	}

	out := make(map[string]struct{})
	// 逐级展开笛卡尔积；层级最多 5 层，量级可控，且只在加载期执行一次。
	var walk func(level int, key string)
	walk = func(level int, key string) {
		if level == len(alts) {
			for w := range idx.paths[key] {
				out[w] = struct{}{}
			}
			return
		}
		for _, alt := range alts[level] {
			next := alt
			if key != "" {
				next = key + attrSep + alt
			}
			walk(level+1, next)
		}
	}
	walk(0, "")

	return out, nil
}

// compileCtx 编译上下文表达式。
//
//	"到,往,朝"              字面词
//	"$量词=个体"            词库检索
//	"&v" / "&a，v"          词性
//	"#叫+nh"                词 + 词性
//	"*重叠"                 重叠词
func compileCtx(raw string, idx *wordIndex) (*ctxExpr, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return nil, nil
	}

	switch s[0] {
	case '$':
		words, err := lookupAttr(s[1:], idx)
		if err != nil {
			return nil, err
		}
		return &ctxExpr{op: ctxWords, words: words}, nil
	case '&':
		if pos := splitList(s[1:]); len(pos) > 0 {
			return &ctxExpr{op: ctxPOS, pos: pos}, nil
		}
		return nil, fmt.Errorf("empty pos set")
	case '#':
		return compileWordPOS(s[1:])
	case '*':
		return &ctxExpr{op: ctxRepeat}, nil
	default:
		return &ctxExpr{op: ctxWords, words: setOf(splitList(s))}, nil
	}
}

// compileWordPOS 编译 "#词+词性"，顺序不敏感：哪边是汉字哪边就是词。
func compileWordPOS(expr string) (*ctxExpr, error) {
	parts := strings.SplitN(expr, "+", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("bad word+pos expr %q", expr)
	}

	e := &ctxExpr{op: ctxWordPOS}
	if isHanWord(parts[0]) {
		e.wordPart, e.posPart, e.wordFirst = parts[0], parts[1], true
	} else {
		e.wordPart, e.posPart, e.wordFirst = parts[1], parts[0], false
	}

	return e, nil
}

// ------------------------------------------------------------------ 词库索引

const attrSep = "|"

// wordIndex 词库按「词性、子类、属性」路径建索引。
// 每个词会把自己写进所有前缀路径，所以检索时一次 map 命中即可。
type wordIndex struct {
	paths map[string]map[string]struct{}
}

func newWordIndex(words []WordInfo) *wordIndex {
	idx := &wordIndex{paths: make(map[string]map[string]struct{}, len(words)*2)}

	for _, w := range words {
		key := w.POS
		idx.add(key, w.Word)
		for _, attr := range w.Attrs {
			if key == "" {
				key = attr
			} else {
				key += attrSep + attr
			}
			idx.add(key, w.Word)
		}
	}

	return idx
}

func (idx *wordIndex) add(key, word string) {
	if key == "" || word == "" {
		return
	}
	set, ok := idx.paths[key]
	if !ok {
		set = make(map[string]struct{}, 4)
		idx.paths[key] = set
	}
	set[word] = struct{}{}
}

// parseWords 解析 "词|词性|子类1|子类2…"。
func parseWords(raw []string) []WordInfo {
	out := make([]WordInfo, 0, len(raw))

	for _, line := range raw {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, attrSep)
		wi := WordInfo{Word: parts[0]}
		if len(parts) > 1 {
			wi.POS = parts[1]
		}
		for _, attr := range parts[2:] {
			if attr == "" {
				break
			}
			wi.Attrs = append(wi.Attrs, attr)
		}
		if wi.Word == "" {
			continue
		}
		out = append(out, wi)
	}

	return out
}

// appendMissingWords 把字面检索词补进词库，保证它们能被分词成独立 token。
func appendMissingWords(words []WordInfo, rules []Rule) []WordInfo {
	have := make(map[string]struct{}, len(words))
	for _, w := range words {
		have[w.Word] = struct{}{}
	}

	for _, r := range rules {
		s := strings.TrimSpace(r.Search)
		if s == "" || strings.HasPrefix(s, "&") {
			continue
		}
		for _, w := range splitList(s) {
			if _, ok := have[w]; ok {
				continue
			}
			have[w] = struct{}{}
			words = append(words, WordInfo{Word: w, POS: guessPOS(w)})
		}
	}

	return words
}

// guessPOS 给词库外的补充词猜一个词性标记，猜不出来就留空。
func guessPOS(word string) string {
	allDigit := true
	for _, r := range word {
		if !unicode.IsDigit(r) {
			allDigit = false
			break
		}
	}
	if allDigit {
		return "m"
	}
	return ""
}

// newPhraseIndex 构建短语自动机。长短语优先，保证输出顺序稳定。
func newPhraseIndex(m map[string]string) *phraseIndex {
	if len(m) == 0 {
		return nil
	}

	keys := make([]string, 0, len(m))
	for k := range m {
		if k != "" {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		if len(keys[i]) != len(keys[j]) {
			return len(keys[i]) > len(keys[j])
		}
		return keys[i] < keys[j]
	})

	p := &phraseIndex{ac: matcher.New(keys), tags: make([][]string, len(keys))}
	for i, k := range keys {
		p.tags[i] = splitList(m[k])
	}

	return p
}

// ------------------------------------------------------------------ 小工具

// splitList 按中英文逗号拆分，去空白、去空项。
func splitList(s string) []string {
	if s == "" {
		return nil
	}
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == '，' })
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

func setOf(list []string) map[string]struct{} {
	if len(list) == 0 {
		return nil
	}
	out := make(map[string]struct{}, len(list))
	for _, s := range list {
		out[s] = struct{}{}
	}
	return out
}

// isHanWord 判断首字符是否为汉字，用于区分 "#词+词性" 的两部分。
func isHanWord(s string) bool {
	for _, r := range s {
		return unicode.Is(unicode.Han, r)
	}
	return false
}
