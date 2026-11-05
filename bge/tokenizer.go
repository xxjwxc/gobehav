package bge

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/unigram"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/spm"
	"github.com/sugarme/tokenizer/util"
)

// tokenizerConfig 只取 bge-m3（XLM-RoBERTa）用得到的字段。
//
// sugarme/tokenizer 没有实现「从 tokenizer.json 加载」——它的
// NewTokenizerFromFile 是个空函数——所以这里按 XLM-R 的结构把零件手工装起来：
// Precompiled 归一化 + WhiteSpaceSplit/Metaspace 预切分 + Unigram 模型。
// 装出来的结果与 transformers 的 input_ids 逐条一致，见 tokenizer_test.go。
type tokenizerConfig struct {
	Normalizer struct {
		PrecompiledCharsmap string `json:"precompiled_charsmap"`
	} `json:"normalizer"`
	Model struct {
		UnkID        int                 `json:"unk_id"`
		ByteFallback bool                `json:"byte_fallback"`
		Vocab        [][]json.RawMessage `json:"vocab"`
	} `json:"model"`
}

// xlmrTokenizer bge-m3 的分词器。
type xlmrTokenizer struct {
	tk     *tokenizer.Tokenizer
	maxLen int

	clsID int
	sepID int
	padID int
}

// newXLMRTokenizer 从 tokenizer.json 组装分词器。
func newXLMRTokenizer(file string, maxLen int) (*xlmrTokenizer, error) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("bge: 读 %s 失败: %w", file, err)
	}
	var c tokenizerConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("bge: 解析 %s 失败: %w", file, err)
	}
	if len(c.Model.Vocab) == 0 {
		return nil, fmt.Errorf("bge: %s 里没有 model.vocab", file)
	}

	chars, err := base64.StdEncoding.DecodeString(c.Normalizer.PrecompiledCharsmap)
	if err != nil {
		return nil, fmt.Errorf("bge: precompiled_charsmap 不是合法 base64: %w", err)
	}
	precompiled, err := spm.NewPrecompiledFrom(chars)
	if err != nil {
		return nil, fmt.Errorf("bge: 解析 precompiled_charsmap 失败: %w", err)
	}

	vocab := make([]unigram.TokenScore, 0, len(c.Model.Vocab))
	ids := make(map[string]int, len(c.Model.Vocab))
	for i, pair := range c.Model.Vocab {
		if len(pair) < 2 {
			return nil, fmt.Errorf("bge: vocab[%d] 只有 %d 个字段", i, len(pair))
		}
		var token string
		var score float64
		if err := json.Unmarshal(pair[0], &token); err != nil {
			return nil, fmt.Errorf("bge: vocab[%d] token 解析失败: %w", i, err)
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, fmt.Errorf("bge: vocab[%d] score 解析失败: %w", i, err)
		}
		vocab = append(vocab, unigram.TokenScore{Token: token, Score: score})
		if _, ok := ids[token]; !ok {
			ids[token] = i
		}
	}

	model, err := unigram.New(vocab, util.NewParams(map[string]interface{}{
		"unk_id":        float64(c.Model.UnkID),
		"byte_fallback": c.Model.ByteFallback,
	}))
	if err != nil {
		return nil, fmt.Errorf("bge: 构建 unigram 模型失败: %w", err)
	}

	t := &xlmrTokenizer{
		tk:     tokenizer.NewTokenizer(model),
		maxLen: maxLen,
		clsID:  lookup(ids, "<s>", 0),
		sepID:  lookup(ids, "</s>", 2),
		padID:  lookup(ids, "<pad>", 1),
	}
	t.tk.WithNormalizer(&normalizer.Precompiled{Precompiled: precompiled})
	t.tk.WithPreTokenizer(pretokenizer.NewSequence([]tokenizer.PreTokenizer{
		pretokenizer.NewWhitespaceSplit(),
		pretokenizer.NewMetaspaceWithScheme("▁", pretokenizer.Always),
	}))
	return t, nil
}

func lookup(ids map[string]int, token string, fallback int) int {
	if id, ok := ids[token]; ok {
		return id
	}
	return fallback
}

// encode 产出 [CLS] + tokens + [SEP]；maxLen 把两个特殊符一起算进去，
// 与 transformers 的 truncation + max_length 口径一致。
func (t *xlmrTokenizer) encode(text string) []int64 {
	enc, err := t.tk.EncodeSingle(text, false)
	if err != nil {
		// 分词失败不阻断兜底，退化成一个只有特殊符的空序列
		return []int64{int64(t.clsID), int64(t.sepID)}
	}

	src := enc.GetIds()
	if limit := t.maxLen - 2; limit > 0 && len(src) > limit {
		src = src[:limit]
	}

	out := make([]int64, 0, len(src)+2)
	out = append(out, int64(t.clsID))
	for _, id := range src {
		out = append(out, int64(id))
	}
	out = append(out, int64(t.sepID))
	return out
}
