//go:build cgo

package bge

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/xxjwxc/gobehav/internal/span"
)

// Encoder bge-m3 向量编码器 + 标签原型检索器。
//
// 它实现 gobehav.Fallback，可以安全地被 gobehav.Batch 并发调用。
type Encoder struct {
	tok   *xlmrTokenizer
	onnx  *onnxEncoder
	proto *prototypes

	threshold float32
	dim       int

	// tokenizer 内部有缓存，不能并发；推理侧在 onnxEncoder 里自己加锁。
	tokMu sync.Mutex
}

// NewFromDir 按目录加载 bge-m3：目录里放四个固定文件就行，不用一个个传路径。
//
//	encoder.int8.onnx / tokenizer.json / label_prototypes.f32 / label_index.json
//
// onnxruntime 动态库也放这个目录时，会自动从目录里加载，不需要配环境变量。
func NewFromDir(dir string) (*Encoder, error) {
	if dir == "" {
		return nil, fmt.Errorf("bge: 模型目录为空")
	}

	return New(Config{
		ModelPath:      filepath.Join(dir, ModelFile),
		TokenizerPath:  filepath.Join(dir, TokenizerFile),
		PrototypesPath: filepath.Join(dir, PrototypesFile),
		IndexPath:      filepath.Join(dir, IndexFile),
	})
}

// New 加载模型、分词器与标签原型。
func New(cfg Config) (*Encoder, error) {
	for name, path := range map[string]string{
		"ModelPath":      cfg.ModelPath,
		"TokenizerPath":  cfg.TokenizerPath,
		"PrototypesPath": cfg.PrototypesPath,
		"IndexPath":      cfg.IndexPath,
	} {
		if path == "" {
			return nil, fmt.Errorf("bge: %s 不能为空", name)
		}
		if _, err := os.Stat(path); err != nil {
			return nil, fmt.Errorf("bge: %s 指向的 %s 不可用: %w", name, path, err)
		}
	}

	threshold := cfg.Threshold
	if threshold <= 0 {
		threshold = DefaultThreshold
	}
	maxLen := cfg.MaxLen
	if maxLen <= 0 {
		maxLen = DefaultMaxLen
	}

	proto, err := loadPrototypes(cfg.PrototypesPath, cfg.IndexPath)
	if err != nil {
		return nil, err
	}
	tok, err := newXLMRTokenizer(cfg.TokenizerPath, maxLen)
	if err != nil {
		return nil, err
	}
	onnx, err := newONNXEncoder(cfg.ModelPath, cfg.LibraryPath, filepath.Dir(cfg.ModelPath), proto.dim)
	if err != nil {
		return nil, err
	}

	return &Encoder{
		tok:       tok,
		onnx:      onnx,
		proto:     proto,
		threshold: threshold,
		dim:       proto.dim,
	}, nil
}

// Close 释放 onnxruntime 会话。可重复调用。
func (e *Encoder) Close() error {
	if e == nil || e.onnx == nil {
		return nil
	}
	return e.onnx.close()
}

// Embed 把一条文本编码成 1024 维向量（已 L2 归一化）。
func (e *Encoder) Embed(text string) ([]float32, error) {
	vecs, err := e.EmbedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch 批量编码。批内按最长的一条 padding，只跑一次推理。
func (e *Encoder) EmbedBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	seqs, maxSeq := e.tokenizeAll(texts)
	if maxSeq == 0 {
		return nil, fmt.Errorf("bge: 没有可编码的 token")
	}

	ids := make([]int64, len(texts)*maxSeq)
	mask := make([]int64, len(texts)*maxSeq)
	for i, seq := range seqs {
		base := i * maxSeq
		for j := range ids[base : base+maxSeq] {
			ids[base+j] = int64(e.tok.padID)
		}
		copy(ids[base:], seq)
		for j := range seq {
			mask[base+j] = 1
		}
	}

	vecs, err := e.onnx.run(ids, mask, len(texts), maxSeq)
	if err != nil {
		return nil, err
	}
	for _, vec := range vecs {
		if len(vec) != e.dim {
			return nil, fmt.Errorf("bge: 输出维度 %d 与原型维度 %d 不一致", len(vec), e.dim)
		}
	}
	return vecs, nil
}

// TopK 返回与 text 最相近的 k 个标签及余弦分数，不做阈值过滤。
func (e *Encoder) TopK(text string, k int) ([]Match, error) {
	vec, err := e.Embed(text)
	if err != nil {
		return nil, err
	}
	return e.proto.top(vec, k), nil
}

// Tag 实现 gobehav.Fallback：整句给一个标签，分数不到阈值就放弃。
//
// 兜底只当弱提示用——标签表本身粒度重叠（mood_happy 与 mood_satisfied、
// behavior_no 与 behavior_reject），top1 命中率在短语上只有 64%，
// 所以宁可返回空，也不要硬塞一个标签。
func (e *Encoder) Tag(text string) []span.Span {
	n := len([]rune(text))
	if n == 0 {
		return nil
	}
	matches, err := e.TopK(text, 1)
	if err != nil || len(matches) == 0 {
		return nil
	}
	if matches[0].Score < e.threshold {
		return nil
	}
	return []span.Span{{
		Start:    0,
		End:      int32(n),
		Tag:      matches[0].Tag,
		Priority: SpanPriority,
	}}
}

// tokenizeAll 逐条分词，同时返回批内最长长度。
func (e *Encoder) tokenizeAll(texts []string) ([][]int64, int) {
	e.tokMu.Lock()
	defer e.tokMu.Unlock()

	seqs := make([][]int64, len(texts))
	maxSeq := 0
	for i, t := range texts {
		seq := e.tok.encode(t)
		seqs[i] = seq
		if len(seq) > maxSeq {
			maxSeq = len(seq)
		}
	}
	return seqs, maxSeq
}
