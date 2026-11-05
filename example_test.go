package gobehav_test

import (
	"fmt"

	gobehav "github.com/xxjwxc/gobehav"
)

// ExampleNew 最简用法：默认加载包内嵌资产，无需任何参数。
func ExampleNew() {
	a, err := gobehav.New()
	if err != nil {
		panic(err)
	}
	defer a.Close()

	fmt.Println(a.Langs())
	// Output:
	// [cn]
}

// ExampleAnalyze 输出带字符区间的标签。Start/End 是 rune 下标、左闭右开，
// 可以直接用来切 res.Sentence 取命中片段。
func ExampleAnalyze() {
	a := gobehav.MustNew()
	defer a.Close()

	res := a.Analyze("对不起")
	fmt.Println(res.Sentence)
	for _, t := range res.Tags {
		fmt.Printf("[%d,%d) %s\n", t.Start, t.End, t.Tag)
	}
	// Output:
	// 对不起
	// [0,3) behavior_apology
}

// ExampleAnalyzer_AnalyzeWith 用调用方热规则补业务词，先于内置词表生效。
func ExampleAnalyzer_AnalyzeWith() {
	a := gobehav.MustNew()
	defer a.Close()

	res := a.AnalyzeWith("开心到飞起", gobehav.CallOptions{
		Hot: []gobehav.HotRule{{Pattern: "飞起", Tag: "mood_happy", Priority: 1}},
	})
	for _, t := range res.Tags {
		if t.Tag == "mood_happy" {
			fmt.Printf("hot hit at [%d,%d)\n", t.Start, t.End)
		}
	}
	// Output:
	// hot hit at [3,5)
}

// ExampleBatch 批量调用，文本之间互不相关，内部按 CPU 核数并行。
func ExampleBatch() {
	a := gobehav.MustNew()
	defer a.Close()

	out := a.BatchLang([]string{"对不起", "谢谢你"}, "cn")
	for _, res := range out {
		fmt.Printf("%s -> %d tags\n", res.Orig, len(res.Tags))
	}
	// Output:
	// 对不起 -> 1 tags
	// 谢谢你 -> 1 tags
}
