package gobehav

// 上下文表达式求值。
//
// 原实现有 9 个上下文维度，这里按「前后 × 出现/紧邻 × 是否取反」归成四个动作：
//
//	window(tokens, 0, i)     前方出现
//	window(tokens, i+1, n)   后方出现
//	adjacent(tokens, i-1, -1) 前方紧邻
//	adjacent(tokens, i+1, +1) 后方紧邻
//
// 取反维度由调用方对结果取反，与规则表的 fda/fdca/lda/ldca 一一对应。

// window 判断 [from, to) 窗口内是否存在满足条件的 token。
// ref 是命中词本身，供「重叠」判断使用。
func (e *ctxExpr) window(tokens []token, from, to int, ref string) bool {
	if e == nil {
		return true
	}
	if from < 0 {
		from = 0
	}
	if to > len(tokens) {
		to = len(tokens)
	}

	for j := from; j < to; j++ {
		switch e.op {
		case ctxWords:
			if _, ok := e.words[tokens[j].Text]; ok {
				return true
			}
		case ctxPOS:
			if tokenPosIn(tokens[j], e.pos) {
				return true
			}
		case ctxRepeat:
			if tokens[j].Text == ref {
				return true
			}
		case ctxWordPOS:
			if tokens[j].Text == e.wordPart && j+1 < to && tokenPosIn(tokens[j+1], oneOf(e.posPart)) {
				return true
			}
		}
	}

	return false
}

// adjacent 判断紧邻位置 j 是否满足条件。
// dir 表示「词/词性」组合里另一半相对 j 的方向：前方紧邻取 -1，后方紧邻取 +1。
func (e *ctxExpr) adjacent(tokens []token, j, dir int, ref string) bool {
	if e == nil {
		return true
	}
	if j < 0 || j >= len(tokens) {
		return false
	}

	switch e.op {
	case ctxWords:
		_, ok := e.words[tokens[j].Text]
		return ok
	case ctxPOS:
		return tokenPosIn(tokens[j], e.pos)
	case ctxRepeat:
		return tokens[j].Text == ref
	case ctxWordPOS:
		other := j + dir
		if other < 0 || other >= len(tokens) {
			return false
		}
		if e.wordFirst {
			return tokens[j].Text == e.wordPart && tokenPosIn(tokens[other], oneOf(e.posPart))
		}
		return tokens[other].Text == e.wordPart && tokenPosIn(tokens[j], oneOf(e.posPart))
	}

	return false
}

// ok 求值整组上下文约束。
//
// sub（主语）依赖依存句法分析，gobehav 不带句法模型，按「不限制」处理，
// 与导出时丢弃 DP 列的口径一致。
//
// 存在性维度在句子首/尾没有可看的位置时按原实现跳过（视为通过）。
func (c *ctxSet) ok(tokens []token, i int) bool {
	if c.empty() || i < 0 || i >= len(tokens) {
		return true
	}

	ref := tokens[i].Text
	last := len(tokens) - 1

	if c.fa != nil && !c.fa.window(tokens, 0, i, ref) {
		return false
	}
	if c.fda != nil && i > 0 && c.fda.window(tokens, 0, i, ref) {
		return false
	}
	if c.fca != nil && i > 0 && !c.fca.adjacent(tokens, i-1, -1, ref) {
		return false
	}
	if c.fdca != nil && i > 0 && c.fdca.adjacent(tokens, i-1, -1, ref) {
		return false
	}
	if c.la != nil && !c.la.window(tokens, i+1, len(tokens), ref) {
		return false
	}
	if c.lda != nil && i < last && c.lda.window(tokens, i+1, len(tokens), ref) {
		return false
	}
	if c.lca != nil && i < last && !c.lca.adjacent(tokens, i+1, 1, ref) {
		return false
	}
	if c.ldca != nil && i < last && c.ldca.adjacent(tokens, i+1, 1, ref) {
		return false
	}

	return true
}

// tokenPosIn 判断 token 的词性是否落在集合内。
//
// 词库里词性是中文名映射过来的 LTP 标记（v/a/d/p/c/r/q/m/nd/wp），
// 未登录词没有标记，一律视为不匹配——上下文里的「不是动词」这类判断
// 宁可保守，也不要把未登录词当成动词。
func tokenPosIn(tok token, set []string) bool {
	if len(set) == 0 {
		return false
	}
	for _, p := range set {
		if tok.POS == p {
			return true
		}
		// ws 在原规则表里表示「空白/标点」，词库里对应 标点 -> wp
		if p == "ws" && (tok.POS == "wp" || isPunct(tok.Text)) {
			return true
		}
	}
	return false
}

// posAllowed 判断命中词的词性是否满足规则约束。
// 未登录词（没有词性）一律放行：我们并不知道它的词性，不该据此否掉规则。
func posAllowed(tok token, set []string) bool {
	if len(set) == 0 || tok.POS == "" {
		return true
	}
	return tokenPosIn(tok, set)
}

// markAllowed 判断标记（位置 / 语篇）是否满足约束，支持 ! 前缀取反。
func markAllowed(mark string, filters []string) bool {
	if len(filters) == 0 {
		return true
	}

	hasPositive, matched := false, false
	for _, f := range filters {
		if len(f) > 1 && f[0] == '!' {
			if mark == f[1:] {
				return false
			}
			continue
		}
		hasPositive = true
		if mark == f {
			matched = true
		}
	}

	return !hasPositive || matched
}

func oneOf(s string) []string {
	if s == "" {
		return nil
	}
	return []string{s}
}
