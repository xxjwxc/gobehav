package gobehav_test

import (
	"sync/atomic"
	"testing"

	"github.com/xxjwxc/gobehav"
)

// fakeFallback 记录被调用次数，固定返回一个覆盖整句的标签。
type fakeFallback struct {
	calls atomic.Int32
}

func (f *fakeFallback) Tag(text string) []gobehav.Span {
	f.calls.Add(1)
	return []gobehav.Span{{
		Start:    0,
		End:      int32(len([]rune(text))),
		Tag:      "mood_happy",
		Priority: 50,
	}}
}

func newWithFallback(t *testing.T) (*gobehav.Analyzer, *fakeFallback) {
	t.Helper()
	fb := &fakeFallback{}
	a, err := gobehav.New(gobehav.WithFallback(fb))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = a.Close() })
	return a, fb
}

// 规则层打上标签时，兜底不能被调用：向量推理比词典匹配贵得多，不能每条都跑。
func TestFallbackSkippedWhenRuleHits(t *testing.T) {
	a, fb := newWithFallback(t)

	res := a.Analyze("对不起，我真的很抱歉")
	if len(res.Tags) == 0 {
		t.Fatal("这句话规则层应该命中，结果却为空")
	}
	if n := fb.calls.Load(); n != 0 {
		t.Fatalf("规则已命中，兜底不该被调用，实际调用 %d 次", n)
	}
}

// 规则层零命中时，兜底结果原样透出，不再过优先级与互斥。
func TestFallbackFillsEmpty(t *testing.T) {
	a, fb := newWithFallback(t)

	res := a.Analyze("这事儿整挺好")
	if n := fb.calls.Load(); n != 1 {
		t.Fatalf("规则零命中应调用兜底 1 次，实际 %d 次", n)
	}
	if len(res.Tags) != 1 {
		t.Fatalf("兜底应透出 1 个标签，实际 %d 个", len(res.Tags))
	}
	got := res.Tags[0]
	if got.Tag != "mood_happy" || got.Priority != 50 {
		t.Fatalf("兜底标签被改写: %+v", got)
	}
	if want := int32(len([]rune(res.Sentence))); got.Start != 0 || got.End != want {
		t.Fatalf("区间应在清洗后的句子上: got [%d,%d) want [0,%d)", got.Start, got.End, want)
	}
}

// 空文本不值得跑向量模型。
func TestFallbackSkipsEmptyText(t *testing.T) {
	a, fb := newWithFallback(t)

	a.Analyze("")
	a.Analyze("   ")

	if n := fb.calls.Load(); n != 0 {
		t.Fatalf("空文本不该调用兜底，实际调用 %d 次", n)
	}
}

// Batch 会多协程调兜底，实现必须并发安全；这里只确认调用次数对得上。
func TestFallbackBatch(t *testing.T) {
	a, fb := newWithFallback(t)

	texts := make([]string, 16)
	for i := range texts {
		texts[i] = "这事儿整挺好"
	}

	res := a.BatchLang(texts, "cn")
	if len(res) != len(texts) {
		t.Fatalf("BatchLang 返回 %d 条，期望 %d 条", len(res), len(texts))
	}
	for i, r := range res {
		if len(r.Tags) != 1 || r.Tags[0].Tag != "mood_happy" {
			t.Fatalf("第 %d 条兜底结果异常: %+v", i, r.Tags)
		}
	}
	if n := fb.calls.Load(); n != int32(len(texts)) {
		t.Fatalf("兜底应被调用 %d 次，实际 %d 次", len(texts), n)
	}
}

func TestNilFallbackRejected(t *testing.T) {
	if _, err := gobehav.New(gobehav.WithFallback(nil)); err == nil {
		t.Fatal("WithFallback(nil) 应当报错")
	}
}
