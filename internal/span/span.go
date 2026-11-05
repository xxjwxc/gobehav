// Package span 定义标签命中区间。单独放在 internal 下，是为了让 bge 子包
// 能引用它而不必 import 主包 —— 主包需要 import bge 做默认兜底，反向引用会成环。
package span

// Span 一个标签命中区间。
//
// Start / End 是字符（rune）下标，左闭右开：End = Start + 命中片段长度。
// 这与原 Triton 实现一致（Python 侧 end = start + len(word)），
// 调用方不需要做任何偏移换算。
type Span struct {
	Start    int32  `json:"start"`
	End      int32  `json:"end"`
	Tag      string `json:"tag"`
	Priority int32  `json:"priority"` // 标签优先级，越小越优先
}
