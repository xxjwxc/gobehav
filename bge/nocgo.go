//go:build !cgo

package bge

import (
	"errors"

	"github.com/xxjwxc/gobehav/internal/span"
)

// errNoCGO 没有 cgo 时的统一错误。
//
// 这个文件存在的唯一理由：让 `CGO_ENABLED=0 go build ./...` 照样能编过。
// 否则调用方一旦 import bge（比如 cmd/gobehav-http 想带兜底），关掉 cgo 的
// 构建就会直接失败。能编译、能构造，只是 New 一定报错。
var errNoCGO = errors.New("bge: 编译时未启用 cgo（CGO_ENABLED=0），向量兜底不可用")

// Encoder 没有 cgo 时是个空壳，实现 gobehav.Fallback 但不产出任何标签。
type Encoder struct{}

// New 没有 cgo 时直接报错，调用方按普通错误处理即可（兜底是可选的）。
func New(cfg Config) (*Encoder, error) { return nil, errNoCGO }

// NewFromDir 没有 cgo 时同样直接报错。
func NewFromDir(dir string) (*Encoder, error) { return nil, errNoCGO }

// Close 空实现，方便 defer。
func (e *Encoder) Close() error { return nil }

// Embed 没有 cgo 时不可用。
func (e *Encoder) Embed(text string) ([]float32, error) { return nil, errNoCGO }

// EmbedBatch 没有 cgo 时不可用。
func (e *Encoder) EmbedBatch(texts []string) ([][]float32, error) { return nil, errNoCGO }

// TopK 没有 cgo 时不可用。
func (e *Encoder) TopK(text string, k int) ([]Match, error) { return nil, errNoCGO }

// Tag 没有 cgo 时放弃兜底。
func (e *Encoder) Tag(text string) []span.Span { return nil }
