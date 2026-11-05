package bge

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writePrototypes(t *testing.T, tags []string, dim int, mat []float32) (string, string) {
	t.Helper()
	dir := t.TempDir()

	vecFile := filepath.Join(dir, "label_prototypes.f32")
	buf := make([]byte, len(mat)*4)
	for i, v := range mat {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	if err := os.WriteFile(vecFile, buf, 0o600); err != nil {
		t.Fatalf("写矩阵: %v", err)
	}

	indexFile := filepath.Join(dir, "label_index.json")
	raw, err := json.Marshal(map[string]interface{}{
		"dim": dim, "count": len(tags), "tags": tags,
	})
	if err != nil {
		t.Fatalf("序列化索引: %v", err)
	}
	if err := os.WriteFile(indexFile, raw, 0o600); err != nil {
		t.Fatalf("写索引: %v", err)
	}
	return vecFile, indexFile
}

func TestPrototypesTop(t *testing.T) {
	vecFile, indexFile := writePrototypes(t,
		[]string{"behavior_all", "behavior_but", "mood_happy"}, 4,
		[]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0.6, 0.8, 0, 0,
		})

	p, err := loadPrototypes(vecFile, indexFile)
	if err != nil {
		t.Fatalf("loadPrototypes: %v", err)
	}

	matches := p.top([]float32{0.7, 0.7, 0, 0}, 2)
	if len(matches) != 2 {
		t.Fatalf("期望 2 个候选，实际 %d 个", len(matches))
	}
	// 0.6*0.7 + 0.8*0.7 = 0.98，比其他两个 0.7 都高
	if matches[0].Tag != "mood_happy" {
		t.Fatalf("top1 应为 mood_happy，实际 %s", matches[0].Tag)
	}
	if diff := matches[0].Score - 0.98; diff > 1e-5 || diff < -1e-5 {
		t.Fatalf("top1 分数应约等于 0.98，实际 %f", matches[0].Score)
	}
	if matches[0].Score < matches[1].Score {
		t.Fatal("结果没有按分数降序")
	}
}

func TestPrototypesTopClampsK(t *testing.T) {
	vecFile, indexFile := writePrototypes(t, []string{"a", "b"}, 2, []float32{1, 0, 0, 1})

	p, err := loadPrototypes(vecFile, indexFile)
	if err != nil {
		t.Fatalf("loadPrototypes: %v", err)
	}
	if got := len(p.top([]float32{1, 1}, 99)); got != 2 {
		t.Fatalf("k 超出标签数时应截断到 2，实际 %d", got)
	}
}

func TestLoadPrototypesRejectsBadSize(t *testing.T) {
	vecFile, indexFile := writePrototypes(t, []string{"a"}, 4, []float32{1, 0, 0, 0})

	// 索引说 dim=8，矩阵却只有 4 维，必须当场报错而不是读到越界
	raw, err := json.Marshal(map[string]interface{}{"dim": 8, "count": 1, "tags": []string{"a"}})
	if err != nil {
		t.Fatalf("序列化索引: %v", err)
	}
	if err := os.WriteFile(indexFile, raw, 0o600); err != nil {
		t.Fatalf("写索引: %v", err)
	}

	if _, err := loadPrototypes(vecFile, indexFile); err == nil {
		t.Fatal("维度不一致应当报错")
	}
}
