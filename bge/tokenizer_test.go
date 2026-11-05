package bge

import (
	"os"
	"slices"
	"testing"
)

// goldenIDs 由 transformers 的 AutoTokenizer.from_pretrained("BAAI/bge-m3") 产出，
// 覆盖中文、中英混排、多余空白、空串与标点。
var goldenIDs = []struct {
	text string
	ids  []int64
}{
	{"对不起，我真的很抱歉", []int64{0, 6, 1036, 50972, 4, 631, 67664, 176922, 2}},
	{"我今天特别开心，哈哈哈哈", []int64{0, 13129, 7461, 21532, 77337, 4, 16659, 183270, 2}},
	{"这事儿整挺好", []int64{0, 63982, 168015, 13082, 31183, 1322, 2}},
	{"又emo了，不想说话", []int64{0, 6, 2100, 23800, 274, 4, 38448, 74845, 2}},
	{"这波操作我是真的服了", []int64{0, 63982, 7420, 13160, 631, 165219, 10871, 274, 2}},
	{"绝绝子，太过分了吧", []int64{0, 6, 35038, 35038, 1344, 4, 4150, 3074, 1583, 90378, 2}},
	{"我真的栓Q了", []int64{0, 6, 178084, 233726, 2737, 274, 2}},
	{"整个都不好了", []int64{0, 6, 20898, 23871, 43986, 2}},
	{"谢谢你，辛苦了", []int64{0, 6, 81611, 994, 4, 85983, 274, 2}},
	{"  有多余  空格  ", []int64{0, 6, 57285, 11152, 6, 5841, 6347, 2}},
	{"ABC abc 123", []int64{0, 47457, 1563, 238, 37638, 2}},
	{"", []int64{0, 2}},
}

// tokenizerPath 找 tokenizer.json：优先环境变量，其次仓库内的生成目录。
// 文件不在就跳过——它 17 MB，不属于仓库内容。
func tokenizerPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("BGE_TOKENIZER"); p != "" {
		return p
	}
	p := "../assets/bge_m3/tokenizer.json"
	if _, err := os.Stat(p); err != nil {
		t.Skip("没有 tokenizer.json，跳过（用 BGE_TOKENIZER 指定路径）")
	}
	return p
}

// 分词必须和 transformers 一模一样，差一个 id 向量就偏了。
func TestTokenizerMatchesTransformers(t *testing.T) {
	tok, err := newXLMRTokenizer(tokenizerPath(t), DefaultMaxLen)
	if err != nil {
		t.Fatalf("newXLMRTokenizer: %v", err)
	}

	for _, g := range goldenIDs {
		got := tok.encode(g.text)
		if !slices.Equal(got, g.ids) {
			t.Errorf("%q\n  got  %v\n  want %v", g.text, got, g.ids)
		}
	}
}

func TestTokenizerTruncates(t *testing.T) {
	tok, err := newXLMRTokenizer(tokenizerPath(t), 8)
	if err != nil {
		t.Fatalf("newXLMRTokenizer: %v", err)
	}

	got := tok.encode("我今天特别开心，哈哈哈哈，真的特别开心，哈哈哈哈")
	if len(got) != 8 {
		t.Fatalf("maxLen=8 应产出 8 个 token，实际 %d: %v", len(got), got)
	}
	if got[0] != 0 || got[7] != 2 {
		t.Fatalf("首尾必须是 [CLS]/[SEP]，实际 %v", got)
	}
}
