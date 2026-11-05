package gobehav

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

// analyzeOne 分析单条文本。
func (a *Analyzer) analyzeOne(text string, o CallOptions, set *assetSet) Result {
	res := Result{Orig: text}

	cleaned := cleanSpace(text)
	res.Sentence = cleaned

	lang := normLang(o.Lang)
	if lang == "" {
		lang = a.defaultLang
	}
	if lang == "" {
		lang = detectLang(cleaned)
	}

	c := a.pick(set, lang)
	runes := []rune(cleaned)
	if c == nil || len(runes) == 0 {
		res.Tags = a.fillFallback(cleaned, res.Tags)
		return res
	}

	tokens := c.dict.cut(runes, 0, len(runes))
	res.Tokens = joinTokens(tokens)
	if len(tokens) == 0 {
		res.Tags = a.fillFallback(cleaned, res.Tags)
		return res
	}

	marks := chapterSegments(runes)
	segs := segmentIndex(runes)

	hits := make([]hit, 0, 16)
	hits = c.matchPhrases(runes, hits)
	hits = c.matchRules(tokens, marks, hits)
	hits = a.matchHot(cleaned, o.Hot, c.tags, hits)

	res.Tags = postProcess(hits, c.tags, segs)
	res.Tags = a.fillFallback(cleaned, res.Tags)

	return res
}

// fillFallback 规则层一条标签都没打出来时，交给调用方注入的兜底实现。
// 兜底是有代价的（向量模型推理），所以只在完全空命中时触发。
func (a *Analyzer) fillFallback(text string, tags []Span) []Span {
	if len(tags) > 0 || a.fallback == nil || text == "" {
		return tags
	}
	return a.fallback.Tag(text)
}

// pick 按语种取资产：优先调用方指定，其次默认语种，最后退回任意一个已加载的语种。
// 语种没有对应资产时不报错，只是返回空结果，保证服务可用性不依赖资产完整性。
func (a *Analyzer) pick(set *assetSet, lang string) *compiled {
	if set == nil {
		return nil
	}
	if lang != "" {
		if c := set.get(lang); c != nil {
			return c
		}
	}
	if c := set.get(a.defaultLang); c != nil {
		return c
	}
	if len(set.langs) > 0 {
		return set.byLang[set.langs[0]]
	}
	return nil
}

// matchPhrases 短语词表命中：3152 条短语一次 AC 扫描，取全部出现位置。
func (c *compiled) matchPhrases(runes []rune, hits []hit) []hit {
	if c.phrase == nil || c.phrase.ac.Len() == 0 {
		return hits
	}

	for _, m := range c.phrase.ac.FindAll(runes) {
		if m.Index < 0 || m.Index >= len(c.phrase.tags) {
			continue
		}
		for _, tag := range c.phrase.tags[m.Index] {
			hits = append(hits, hit{start: m.Start, end: m.End + 1, tag: tag})
		}
	}

	return hits
}

// matchRules 逐条规则在分词结果上找命中位置。
//
// 每条规则的检索词在编译期已经展开成词集合，所以这里每个 token 只做一次
// map 命中；只有词性/位置/语篇/上下文都成立时才记一条候选。
func (c *compiled) matchRules(tokens []token, marks [][]string, hits []hit) []hit {
	for _, r := range c.rules {
		for i := range tokens {
			if r.words != nil {
				if _, ok := r.words[tokens[i].Text]; !ok {
					continue
				}
			}
			if !posAllowed(tokens[i], r.pos) {
				continue
			}
			if !markAllowed(location(tokens, i), r.loc) {
				continue
			}
			if !chapterAllowed(markAt(marks, tokens[i].Start), r.chapter) {
				continue
			}
			if !r.ctx.ok(tokens, i) {
				continue
			}

			hits = append(hits, hit{start: tokens[i].Start, end: tokens[i].End + 1, tag: r.tag})
		}
	}

	return hits
}

// matchHot 执行调用方热正则。热规则不参与分词，直接在原句上扫描，
// 命中区间换算成 rune 下标后与内置结果一起参与后续合并。
func (a *Analyzer) matchHot(text string, hot []HotRule, tagTab map[string]TagInfo, hits []hit) []hit {
	if len(hot) == 0 || text == "" {
		return hits
	}

	prefix := runePrefix(text)

	for _, h := range hot {
		if h.Pattern == "" || h.Tag == "" {
			continue
		}
		re, err := regexp.Compile(h.Pattern)
		if err != nil {
			a.logError("gobehav: bad hot rule, skipped", "pattern", h.Pattern, "err", err)
			continue
		}

		prio := h.Priority
		if prio <= 0 {
			prio = priorityOf(trimShortSuffix(h.Tag), tagTab)
		}

		for _, m := range re.FindAllStringIndex(text, -1) {
			start, end := prefix[m[0]], prefix[m[1]]
			if end <= start {
				continue
			}
			hits = append(hits, hit{start: start, end: end, tag: h.Tag, prio: prio})
		}
	}

	return hits
}

// runePrefix 返回 byte 下标到 rune 下标的映射，用于把正则的字节区间换算成字符区间。
func runePrefix(s string) []int {
	prefix := make([]int, len(s)+1)

	n := 0
	for i := 0; i < len(s); {
		_, size := utf8.DecodeRuneInString(s[i:])
		for k := 0; k < size && i+k < len(prefix); k++ {
			prefix[i+k] = n
		}
		i += size
		n++
	}
	prefix[len(s)] = n

	return prefix
}

// cleanSpace 去掉汉字、标点、数字两侧的空格，保留英文单词之间的空格。
// 原实现用一条带前后断言的 PCRE 反复替换，实际效果是把中文之间的空格全去掉。
func cleanSpace(s string) string {
	runes := []rune(s)
	if len(runes) == 0 {
		return s
	}

	out := make([]rune, 0, len(runes))
	for i, r := range runes {
		if r != ' ' && r != '\u3000' {
			out = append(out, r)
			continue
		}
		if len(out) == 0 {
			continue
		}

		var next rune
		if i+1 < len(runes) {
			next = runes[i+1]
		}
		if isASCIIAlnum(out[len(out)-1]) && isASCIIAlnum(next) {
			out = append(out, ' ')
		}
	}

	return string(out)
}

func isASCIIAlnum(r rune) bool {
	return r < 128 && (r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z')
}

func joinTokens(tokens []token) string {
	if len(tokens) == 0 {
		return ""
	}

	var b strings.Builder
	for i, t := range tokens {
		if i > 0 {
			b.WriteByte('/')
		}
		b.WriteString(t.Text)
	}

	return b.String()
}

// chapterSegments 按句末标点切句，给每个字符标注所属句子的语篇标记。
//
// 标记与原实现一致：F 句中含逗号、Q 疑问、T 感叹、C 陈述，可以同时带多个。
func chapterSegments(runes []rune) [][]string {
	marks := make([][]string, len(runes))

	start := 0
	for i := 0; i < len(runes); i++ {
		if !isSentenceBreak(string(runes[i])) {
			continue
		}
		mark := segmentMark(runes, start, i+1)
		for j := start; j <= i && j < len(marks); j++ {
			marks[j] = mark
		}
		start = i + 1
	}
	if start < len(runes) {
		mark := segmentMark(runes, start, len(runes))
		for j := start; j < len(runes); j++ {
			marks[j] = mark
		}
	}

	return marks
}

// segmentIndex 把每个字符映射到所属句子的序号，用于句内的互斥与去嵌套。
// 切分口径与 chapterSegments 一致：句末标点算在前一句里。
func segmentIndex(runes []rune) []int {
	idx := make([]int, len(runes))

	seg := 0
	for i, r := range runes {
		idx[i] = seg
		if isSentenceBreak(string(r)) {
			seg++
		}
	}

	return idx
}

func segmentMark(runes []rune, from, to int) []string {
	mark := []string{"C"}

	if last := to - 1; last >= from {
		switch runes[last] {
		case '?', '？':
			mark = []string{"Q"}
		case '!', '！':
			mark = []string{"T"}
		}
	}

	for j := from; j < to; j++ {
		if runes[j] == ',' || runes[j] == '，' {
			return append([]string{"F"}, mark...)
		}
	}

	return mark
}

func markAt(marks [][]string, i int) []string {
	if i < 0 || i >= len(marks) {
		return []string{"C"}
	}
	if len(marks[i]) == 0 {
		return []string{"C"}
	}
	return marks[i]
}

// chapterAllowed 语篇标记约束：正向条件必须全部满足，取反条件必须全部不满足。
func chapterAllowed(marks, filters []string) bool {
	if len(filters) == 0 {
		return true
	}

	for _, f := range filters {
		if len(f) > 1 && f[0] == '!' {
			if hasMark(marks, f[1:]) {
				return false
			}
			continue
		}
		if !hasMark(marks, f) {
			return false
		}
	}

	return true
}

func hasMark(marks []string, want string) bool {
	for _, m := range marks {
		if m == want {
			return true
		}
	}
	return false
}
