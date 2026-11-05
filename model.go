package gobehav

import (
	"os"
	"path/filepath"

	"github.com/xxjwxc/gobehav/bge"
)

// defaultModelDirs 自动探测语义兜底模型的候选目录，相对 exe 目录与工作目录。
//
// 约定是「程序旁边放一个 bge_m3 目录」，目录里的文件见 bge.NewFromDir；
// onnxruntime 动态库放在同一个目录里就会被自动加载。
var defaultModelDirs = []string{
	"bge_m3",
	filepath.Join("assets", "bge_m3"),
	filepath.Join("models", "bge_m3"),
}

// modelFiles 判断目录是不是一个可用的 bge-m3 模型目录。
func modelFiles(dir string) bool {
	for _, name := range []string{
		bge.ModelFile,
		bge.TokenizerFile,
		bge.PrototypesFile,
		bge.IndexFile,
	} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			return false
		}
	}

	return true
}

// findModelDir 按 exe 目录、工作目录的顺序探测默认模型目录，找不到返回空。
func findModelDir() string {
	bases := []string{"."}
	if exe, err := os.Executable(); err == nil {
		bases = append([]string{filepath.Dir(exe)}, bases...)
	}

	for _, base := range bases {
		for _, rel := range defaultModelDirs {
			dir := filepath.Join(base, rel)
			if modelFiles(dir) {
				return dir
			}
		}
	}

	return ""
}

// openFallback 按需打开语义兜底。
//
// 没配模型目录时自动探测；探测不到、或者加载失败（缺模型、缺动态库、没开 cgo），
// 都只是不启用兜底，规则层照常工作 —— 打标链路不能因为兜底起不来而挂掉。
func openFallback(cfg *config) Fallback {
	dir := cfg.modelDir
	if dir == "" {
		// 自动探测只会返回文件齐全的目录，所以走到这里的失败都是真问题：
		// 动态库缺失、onnx 损坏、或者编译时关了 cgo。
		dir = findModelDir()
		if dir == "" {
			return nil
		}
	}

	enc, err := bge.NewFromDir(dir)
	if err != nil {
		if cfg.logger != nil {
			cfg.logger.Warn("gobehav: 语义兜底未启用，退回纯规则", "dir", dir, "err", err)
		}
		return nil
	}

	if cfg.logger != nil {
		cfg.logger.Info("gobehav: 语义兜底已启用", "dir", dir)
	}

	return enc
}
