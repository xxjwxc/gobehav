package gobehav

import "unicode/utf8"

// token 一次分词结果，Start / End 为 rune 下标，闭区间。
type token struct {
	Text  string
	POS   string
	Start int
	End   int
}

// tokenizer 基于词库的正向最大匹配分词器。
//
// 词库（原 wordsInfo 表）每一行自带词性，所以分词与词性标注都不需要额外模型：
// 命中词库的词直接取它的词性，未登录词退化为单字、词性为空。
// 规则只关心「命中词属于哪个词性」，不关心未登录词怎么归，这个精度够用。
type tokenizer struct {
	dict   map[string]string // 词 -> 词性标记
	maxLen int               // 词库最长词的 rune 长度
}

func newTokenizer(words []WordInfo) *tokenizer {
	t := &tokenizer{dict: make(map[string]string, len(words))}

	for _, w := range words {
		if w.Word == "" {
			continue
		}
		if _, ok := t.dict[w.Word]; !ok {
			t.dict[w.Word] = w.POS
		}
		if n := utf8.RuneCountInString(w.Word); n > t.maxLen {
			t.maxLen = n
		}
	}
	if t.maxLen == 0 {
		t.maxLen = 1
	}

	return t
}

// cut 对 text 的 [from, to) 区间做正向最大匹配分词。
func (t *tokenizer) cut(text []rune, from, to int) []token {
	if from < 0 {
		from = 0
	}
	if to > len(text) {
		to = len(text)
	}
	if from >= to {
		return nil
	}

	out := make([]token, 0, to-from)
	for i := from; i < to; {
		matched := false
		limit := i + t.maxLen
		if limit > to {
			limit = to
		}
		for j := limit; j > i; j-- {
			word := string(text[i:j])
			if pos, ok := t.dict[word]; ok {
				out = append(out, token{Text: word, POS: pos, Start: i, End: j - 1})
				i = j
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		out = append(out, token{Text: string(text[i : i+1]), Start: i, End: i})
		i++
	}

	return out
}

// location 计算 token 在句中的位置标记，与原实现的 B / E / I 保持一致。
func location(tokens []token, index int) string {
	if index <= 0 {
		return "B"
	}
	if index >= len(tokens)-1 {
		return "E"
	}
	if isClauseBreak(tokens[index-1].Text) {
		return "B"
	}
	if isSentenceBreak(tokens[index+1].Text) || isClauseBreak(tokens[index+1].Text) {
		return "E"
	}
	return "I"
}

// isClauseBreak 判断 token 是否为分句标点（逗号、冒号、顿号）。
func isClauseBreak(s string) bool {
	switch s {
	case "，", ",", ":", "：", "、":
		return true
	}
	return false
}

// isSentenceBreak 判断 token 是否为句末标点。
func isSentenceBreak(s string) bool {
	switch s {
	case "。", ".", "；", ";", "！", "!", "…", "?", "？":
		return true
	}
	return false
}

// isPunct 判断 token 是否任意标点，对应 LTP 的 wp / ws。
func isPunct(s string) bool {
	return isClauseBreak(s) || isSentenceBreak(s) || s == " " || s == "\u3000"
}
