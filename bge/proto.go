package bge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Match 一个候选标签及其余弦相似度。
type Match struct {
	Tag   string  `json:"tag"`
	Score float32 `json:"score"`
}

// prototypes 标签原型矩阵：每行一个标签的类均值向量，已 L2 归一化。
//
// 行序与 label_index.json 的 tags 一致；矩阵是 numpy 的 float32 裸二进制，
// 小端、行优先。
type prototypes struct {
	dim  int
	tags []string
	mat  []float32
}

type protoIndex struct {
	Dim   int      `json:"dim"`
	Count int      `json:"count"`
	Tags  []string `json:"tags"`
}

func loadPrototypes(vecFile, indexFile string) (*prototypes, error) {
	rawIndex, err := os.ReadFile(indexFile)
	if err != nil {
		return nil, fmt.Errorf("bge: 读 %s 失败: %w", indexFile, err)
	}
	var idx protoIndex
	if err := json.Unmarshal(rawIndex, &idx); err != nil {
		return nil, fmt.Errorf("bge: 解析 %s 失败: %w", indexFile, err)
	}
	if idx.Dim <= 0 || idx.Count <= 0 || len(idx.Tags) != idx.Count {
		return nil, fmt.Errorf("bge: %s 元数据不自洽（dim=%d count=%d tags=%d）",
			indexFile, idx.Dim, idx.Count, len(idx.Tags))
	}

	rawMat, err := os.ReadFile(vecFile)
	if err != nil {
		return nil, fmt.Errorf("bge: 读 %s 失败: %w", vecFile, err)
	}
	want := idx.Count * idx.Dim * 4
	if len(rawMat) != want {
		return nil, fmt.Errorf("bge: %s 体积 %d 与预期 %d 不符（count=%d dim=%d）",
			vecFile, len(rawMat), want, idx.Count, idx.Dim)
	}

	mat := make([]float32, idx.Count*idx.Dim)
	for i := range mat {
		mat[i] = math.Float32frombits(binary.LittleEndian.Uint32(rawMat[i*4:]))
	}
	return &prototypes{dim: idx.Dim, tags: idx.Tags, mat: mat}, nil
}

// top 返回余弦相似度最高的 k 个标签，按分数降序。
func (p *prototypes) top(vec []float32, k int) []Match {
	if k <= 0 || k > len(p.tags) {
		k = len(p.tags)
	}
	all := make([]Match, len(p.tags))
	for i, tag := range p.tags {
		var dot float32
		row := p.mat[i*p.dim : (i+1)*p.dim]
		for j, v := range row {
			dot += v * vec[j]
		}
		all[i] = Match{Tag: tag, Score: dot}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].Score > all[b].Score })
	return all[:k]
}
