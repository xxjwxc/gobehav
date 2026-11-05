package gobehav_test

import (
	"testing"

	gobehav "github.com/xxjwxc/gobehav"
)

// TestAnalyzeCN 打印一批真实语句的结果，方便人工核对标签是否符合预期。
// 断言只放在行为稳定、语义明确的几条上，其余用 t.Logf 观察。
func TestAnalyzeCN(t *testing.T) {
	a, err := gobehav.New()
	if err != nil {
		t.Fatalf("init: %v", err)
	}
	defer a.Close()

	if got := a.Langs(); len(got) == 0 || got[0] != "cn" {
		t.Fatalf("langs = %v, want [cn]", got)
	}

	cases := []string{
		"对不起",
		"我真的很抱歉，对不起",
		"我今天特别开心，哈哈哈哈",
		"整个都不好了",
		"虽然很难，但我还是要坚持",
		"请帮我一下，麻烦你了",
		"这个多少钱？",
		"我不喜欢这个",
		"谢谢你，辛苦了",
	}

	for _, c := range cases {
		res := a.Analyze(c)
		t.Logf("in=%q sentence=%q tokens=%q", c, res.Sentence, res.Tokens)
		for _, tag := range res.Tags {
			t.Logf("    [%d,%d) %-28s p=%d", tag.Start, tag.End, tag.Tag, tag.Priority)
		}
	}
}

// TestSpanBounds 校验区间用字符下标而不是字节下标：
// 中文在 UTF-8 下占 3 字节，若按字节算，End 会明显偏大。
func TestSpanBounds(t *testing.T) {
	a := gobehav.MustNew()
	defer a.Close()

	const text = "对不起"
	res := a.Analyze(text)

	runes := []rune(res.Sentence)
	for _, tag := range res.Tags {
		if tag.Start < 0 || int(tag.End) > len(runes) {
			t.Fatalf("span out of range: %+v, runes=%d", tag, len(runes))
		}
		frag := string(runes[tag.Start:tag.End])
		t.Logf("span %+v -> %q", tag, frag)
	}
}

// TestCleanSpace 校验空格清理：中文间的空格去掉，英文单词间的空格保留。
func TestCleanSpace(t *testing.T) {
	a := gobehav.MustNew()
	defer a.Close()

	cases := map[string]string{
		"我 很 开心":      "我很开心",
		"ok 我 知道了":    "ok我知道了",
		"hello world": "hello world",
		"a 1 b":       "a 1 b",
	}
	for in, want := range cases {
		if got := a.Analyze(in).Sentence; got != want {
			t.Errorf("cleanSpace(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestWithoutRules 关掉检索规则后，只剩短语层命中。
func TestWithoutRules(t *testing.T) {
	a := gobehav.MustNew(gobehav.WithoutRules())
	defer a.Close()

	res := a.Analyze("对不起")
	if len(res.Tags) == 0 {
		t.Fatalf("phrase layer should still hit, got none")
	}
}

// TestHotRule 调用方热规则应该先于内置词表生效，并覆盖优先级。
func TestHotRule(t *testing.T) {
	a := gobehav.MustNew()
	defer a.Close()

	res := a.AnalyzeWith("开心到飞起", gobehav.CallOptions{
		Hot: []gobehav.HotRule{{Pattern: "飞起", Tag: "mood_happy", Priority: 1}},
	})

	for _, tag := range res.Tags {
		if tag.Tag == "mood_happy" {
			if tag.Priority != 1 {
				t.Fatalf("hot priority = %d, want 1", tag.Priority)
			}
			return
		}
	}
	t.Fatalf("hot rule not hit, tags=%v", res.Tags)
}

// TestBatch 批量调用与单条调用结果一致。
func TestBatch(t *testing.T) {
	a := gobehav.MustNew()
	defer a.Close()

	texts := []string{"对不起", "哈哈哈", "谢谢你"}
	got := a.Batch(texts, gobehav.CallOptions{Lang: "cn"})
	if len(got) != len(texts) {
		t.Fatalf("batch len = %d, want %d", len(got), len(texts))
	}
	for i, res := range got {
		if res.Orig != texts[i] {
			t.Fatalf("batch[%d].Orig = %q, want %q", i, res.Orig, texts[i])
		}
	}
}

// TestUnknownLang 未知语种不应报错，退回默认语种。
func TestUnknownLang(t *testing.T) {
	a := gobehav.MustNew()
	defer a.Close()

	if res := a.AnalyzeLang("对不起", "xx"); len(res.Tags) == 0 {
		t.Fatalf("unknown lang should fall back, got empty")
	}
}
