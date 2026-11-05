package gobehav

import "sort"

// hit 一条候选标签，off 为 rune 下标，左闭右开。
// prio 大于 0 时表示调用方热规则直接指定了优先级，优先于标签表。
type hit struct {
	start int
	end   int
	tag   string
	prio  int32
}

// cand 同一个区间上的标签合并结果
type cand struct {
	start     int
	end       int
	tag       string
	priority  int32
	exclusive int32
	conflict  int32
}

const (
	// defaultPriority 标签表里没有的标签按 4 处理，与原实现一致
	defaultPriority int32 = 4
	// noExclusive 不参与互斥的标签，互斥优先级按 10 处理，与原实现一致
	noExclusive int32 = 10
)

// postProcess 把候选标签整理成最终输出，顺序与原实现一致：
//
//	同区间择一 -> 互斥优先级 -> 去嵌套 -> 解交叉 -> 按位置排序
//
// seg 是每个字符所属的句子序号。互斥与去嵌套在原实现里是逐句做的，
// 挪到整篇上做会把别句的标签一起吞掉：一整段话里只要出现一个「不」，
// behavior_no 的互斥优先级就会把全文收敛成只剩否定标签。
func postProcess(hits []hit, tagTab map[string]TagInfo, seg []int) []Span {
	if len(hits) == 0 {
		return nil
	}

	list := pickPerSpan(hits, tagTab)
	list = perSegment(list, seg, applyExclusive)
	list = perSegment(list, seg, dropNested)
	list = dropCrossing(list)

	out := make([]Span, 0, len(list))
	for _, c := range list {
		out = append(out, Span{
			Start:    int32(c.start),
			End:      int32(c.end),
			Tag:      c.tag,
			Priority: c.priority,
		})
	}

	return out
}

// pickPerSpan 按区间聚合候选标签，每个区间只保留一个标签。
func pickPerSpan(hits []hit, tagTab map[string]TagInfo) []*cand {
	type key struct{ start, end int }
	type group struct {
		start, end int
		tags       []string
		prio       int32
	}

	groups := make(map[key]*group, len(hits))
	order := make([]*group, 0, len(hits))

	for _, h := range hits {
		if h.tag == "" || h.start < 0 || h.end <= h.start {
			continue
		}
		k := key{h.start, h.end}
		g, ok := groups[k]
		if !ok {
			g = &group{start: h.start, end: h.end}
			groups[k] = g
			order = append(order, g)
		}
		g.tags = append(g.tags, h.tag)
		if h.prio > 0 && g.prio == 0 {
			g.prio = h.prio
		}
	}

	list := make([]*cand, 0, len(order))
	for _, g := range order {
		tag, prio, conflict := pickTag(g.tags, tagTab)
		if g.prio > 0 {
			prio = g.prio
		}
		if tag == "" {
			continue
		}
		list = append(list, &cand{
			start:     g.start,
			end:       g.end,
			tag:       tag,
			priority:  prio,
			exclusive: exclusiveOf(tag, tagTab),
			conflict:  conflict,
		})
	}

	sort.SliceStable(list, func(i, j int) bool {
		if list[i].start != list[j].start {
			return list[i].start < list[j].start
		}
		return list[i].end < list[j].end
	})

	return list
}

// pickTag 在一个区间的多个标签里择一。
//
//	带 _S 后缀的短句标签优先；同一档内优先级数字小的胜出。
//	同档还有别的标签时打上冲突标记（原实现的 conflict_label）。
func pickTag(tags []string, tagTab map[string]TagInfo) (string, int32, int32) {
	seen := make(map[string]struct{}, len(tags))
	var short, normal []string

	for _, t := range tags {
		base := trimShortSuffix(t)
		if base == "" {
			continue
		}
		if _, ok := seen[base]; ok {
			continue
		}
		seen[base] = struct{}{}
		if isShortTag(t) {
			short = append(short, base)
		} else {
			normal = append(normal, base)
		}
	}

	set := normal
	if len(short) > 0 {
		set = short
	}
	if len(set) == 0 {
		return "", 0, 0
	}

	best := set[0]
	for _, t := range set[1:] {
		if priorityOf(t, tagTab) < priorityOf(best, tagTab) {
			best = t
		}
	}

	conflict := int32(0)
	if len(set) > 1 {
		conflict = 1
	}

	return best, priorityOf(best, tagTab), conflict
}

// applyExclusive 处理互斥优先级：一旦有标签声明了互斥优先级，
// 只保留互斥优先级最小的那一档，并标记为互斥冲突。
func applyExclusive(list []*cand) []*cand {
	minExc := noExclusive
	for _, c := range list {
		if c.exclusive < minExc {
			minExc = c.exclusive
		}
	}
	if minExc >= noExclusive {
		return list
	}

	out := make([]*cand, 0, len(list))
	for _, c := range list {
		if c.exclusive == minExc {
			c.conflict = 2
			out = append(out, c)
		}
	}

	return out
}

// dropNested 去掉被更长标签完全覆盖的标签，容器先排，方便一次判完。
func dropNested(list []*cand) []*cand {
	if len(list) < 2 {
		return list
	}

	sorted := make([]*cand, len(list))
	copy(sorted, list)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].start != sorted[j].start {
			return sorted[i].start < sorted[j].start
		}
		return sorted[i].end > sorted[j].end
	})

	out := make([]*cand, 0, len(sorted))
	for _, c := range sorted {
		nested := false
		for _, keep := range out {
			if keep.start <= c.start && c.end <= keep.end {
				nested = true
				break
			}
		}
		if !nested {
			out = append(out, c)
		}
	}

	return out
}

// dropCrossing 处理区间交叉：交叉时保留优先级更高的那个（数字更小）。
func dropCrossing(list []*cand) []*cand {
	if len(list) < 2 {
		return list
	}

	sorted := make([]*cand, len(list))
	copy(sorted, list)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].start < sorted[j].start })

	out := make([]*cand, 0, len(sorted))
	for _, c := range sorted {
		if len(out) == 0 {
			out = append(out, c)
			continue
		}
		last := out[len(out)-1]
		if c.start < last.end { // 与上一个保留区间交叉
			if last.priority > c.priority {
				out[len(out)-1] = c
			}
			continue
		}
		out = append(out, c)
	}

	return out
}

func priorityOf(tag string, tagTab map[string]TagInfo) int32 {
	if info, ok := tagTab[tag]; ok && info.Priority > 0 {
		return info.Priority
	}
	return defaultPriority
}

func exclusiveOf(tag string, tagTab map[string]TagInfo) int32 {
	if info, ok := tagTab[tag]; ok && info.Exclusive > 0 {
		return info.Exclusive
	}
	return noExclusive
}

// perSegment 按句子分组执行 fn，组内保持原有顺序，再按句子顺序拼回。
// 候选标签已按 start 升序，拼回后仍然升序，dropCrossing 可以接着用。
func perSegment(list []*cand, seg []int, fn func([]*cand) []*cand) []*cand {
	if len(list) < 2 {
		return fn(list)
	}

	groups := make(map[int][]*cand, 4)
	order := make([]int, 0, 4)

	for _, c := range list {
		k := segOf(seg, c.start)
		if _, ok := groups[k]; !ok {
			order = append(order, k)
		}
		groups[k] = append(groups[k], c)
	}

	if len(order) == 1 {
		return fn(list)
	}

	out := make([]*cand, 0, len(list))
	for _, k := range order {
		out = append(out, fn(groups[k])...)
	}

	return out
}

// segOf 返回下标所属的句子序号；越界（比如热正则给出异常区间）归到 -1 一组。
func segOf(seg []int, i int) int {
	if i < 0 || i >= len(seg) {
		return -1
	}

	return seg[i]
}

func isShortTag(tag string) bool { return len(tag) > 2 && tag[len(tag)-2:] == "_S" }

func trimShortSuffix(tag string) string {
	if isShortTag(tag) {
		return tag[:len(tag)-2]
	}
	return tag
}
