package gobehav

import "testing"

// TestPostProcessPerSegment 互斥优先级与去嵌套只作用于句内：
// 长文本里某句的否定不该把别句的道歉一起吞掉。
func TestPostProcessPerSegment(t *testing.T) {
	tagTab := map[string]TagInfo{
		"behavior_apology": {Priority: 1},
		"behavior_no":      {Priority: 1, Exclusive: 2},
	}

	runes := []rune("抱歉。不好")
	seg := segmentIndex(runes)
	hits := []hit{
		{start: 0, end: 2, tag: "behavior_apology"},
		{start: 3, end: 5, tag: "behavior_no"},
	}

	got := postProcess(hits, tagTab, seg)
	if len(got) != 2 {
		t.Fatalf("跨句结果 = %v，期望 2 条（互斥不该跨句生效）", got)
	}
	if got[0].Tag != "behavior_apology" || got[1].Tag != "behavior_no" {
		t.Fatalf("跨句标签 = %v", got)
	}

	// 同一句里两者并存时，互斥优先级小的 behavior_no 应当把道歉压掉。
	got = postProcess(hits, tagTab, []int{0, 0, 0, 0, 0})
	if len(got) != 1 || got[0].Tag != "behavior_no" {
		t.Fatalf("句内结果 = %v，期望只留 behavior_no", got)
	}
}

// TestSegmentIndex 句末标点归前一句，与 chapterSegments 的切分一致。
func TestSegmentIndex(t *testing.T) {
	got := segmentIndex([]rune("好。不"))
	want := []int{0, 0, 1}

	if len(got) != len(want) {
		t.Fatalf("seg = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seg = %v, want %v", got, want)
		}
	}
}
