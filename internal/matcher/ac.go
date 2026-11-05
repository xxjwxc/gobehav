// Package matcher 提供 Aho-Corasick 自动机，用来在一段文本里一次性找出所有
// 短语命中，替掉「逐条 strings.Contains」的全量扫描。
//
// 按 rune 匹配，天然适合中文；构建完成后只读，可并发调用 FindAll。
package matcher

import "unicode/utf8"

// Match 一次命中，Start / End 均为 rune 下标，闭区间。
type Match struct {
	Start int
	End   int
	Index int // 命中的模式下标
}

type node struct {
	next map[rune]int32
	fail int32
	out  []int32
}

// Automaton Aho-Corasick 自动机，构建后只读。
type Automaton struct {
	nodes   []node
	pattern []string
	plen    []int // 每个模式的 rune 长度，避免匹配时反复转换
}

// New 用一批模式构建自动机。空模式会被忽略。
func New(patterns []string) *Automaton {
	a := &Automaton{
		pattern: patterns,
		plen:    make([]int, len(patterns)),
		nodes:   []node{{next: make(map[rune]int32, 8)}},
	}

	for i, p := range patterns {
		a.plen[i] = utf8.RuneCountInString(p)
		if p == "" {
			continue
		}
		a.insert(p, int32(i))
	}
	a.buildFail()

	return a
}

// Len 返回模式数量。
func (a *Automaton) Len() int { return len(a.pattern) }

// Pattern 返回下标对应的模式原文。
func (a *Automaton) Pattern(i int) string {
	if i < 0 || i >= len(a.pattern) {
		return ""
	}
	return a.pattern[i]
}

func (a *Automaton) newNode() int32 {
	a.nodes = append(a.nodes, node{next: make(map[rune]int32, 2)})
	return int32(len(a.nodes) - 1)
}

func (a *Automaton) insert(p string, index int32) {
	cur := int32(0)
	for _, r := range p {
		nxt, ok := a.nodes[cur].next[r]
		if !ok {
			nxt = a.newNode()
			a.nodes[cur].next[r] = nxt
		}
		cur = nxt
	}
	a.nodes[cur].out = append(a.nodes[cur].out, index)
}

// buildFail 按 BFS 建失败链，并把失败链上的输出并入当前节点，
// 这样匹配时只看当前节点的 out 就够了。
func (a *Automaton) buildFail() {
	queue := make([]int32, 0, len(a.nodes))

	for _, nxt := range a.nodes[0].next {
		a.nodes[nxt].fail = 0
		queue = append(queue, nxt)
	}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]

		for r, nxt := range a.nodes[cur].next {
			f := a.nodes[cur].fail
			for {
				if t, ok := a.nodes[f].next[r]; ok {
					a.nodes[nxt].fail = t
					break
				}
				if f == 0 {
					a.nodes[nxt].fail = 0
					break
				}
				f = a.nodes[f].fail
			}
			if out := a.nodes[a.nodes[nxt].fail].out; len(out) > 0 {
				a.nodes[nxt].out = append(a.nodes[nxt].out, out...)
			}
			queue = append(queue, nxt)
		}
	}
}

// FindAll 在 text 中找出所有模式命中。结果按结束位置升序，
// 同一结束位置按模式长度从长到短，调用方可以自行再做取舍。
func (a *Automaton) FindAll(text []rune) []Match {
	if len(text) == 0 || len(a.nodes) == 0 {
		return nil
	}

	out := make([]Match, 0, 16)
	cur := int32(0)

	for i, r := range text {
		for cur != 0 {
			if _, ok := a.nodes[cur].next[r]; ok {
				break
			}
			cur = a.nodes[cur].fail
		}
		if nxt, ok := a.nodes[cur].next[r]; ok {
			cur = nxt
		} else {
			cur = 0
		}

		for _, idx := range a.nodes[cur].out {
			n := a.plen[idx]
			if n == 0 {
				continue
			}
			start := i - n + 1
			if start < 0 {
				continue
			}
			out = append(out, Match{Start: start, End: i, Index: int(idx)})
		}
	}

	return out
}
