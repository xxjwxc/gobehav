package gobehav

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

// assets 内嵌 conf/ 下的默认配置，New() 默认直接使用，调用方无需准备任何外部文件。
//
//	conf/behav_<lang>.yaml  单个语种的全部配置（标签表、短语、词库、检索规则）
//
// conf/ 是给人改的配置目录：改这里就等于改默认行为。内嵌只是编译期快照，
// 想让改完的文件不重新编译就生效，用 WithAssetDir("conf") 或
// WithAssetFile("conf/behav_cn.yaml") 指到磁盘上同一份。
//
// 资产由 tools/gobehav/export_assets.py 从原规则表 wordInfoRules.xlsx 导出，
// 文件头记录了来源、导出时间与规模。
//
// 只内嵌 behav_*.yaml。assets/bge_m3/ 是模型产物（几百 MB 的 onnx），
// 由 bge 子包按路径读取，不能跟着编译进主包——否则二进制直接涨到几百 MB。
//
//go:embed conf/behav_*.yaml
var assets embed.FS

const (
	assetsDir = "conf"
	// defaultAsset 默认语种配置，也是 go:embed 的锚点：没有任何配置文件时编译失败，
	// 可以提前发现漏提交。
	defaultAsset = assetsDir + "/behav_cn.yaml"
	assetPrefix  = "behav_"
	assetSuffix  = ".yaml"
)

// ExportAssets 把包内内嵌的配置写到 dir，返回实际写出的文件路径。
//
// 给只有二进制、没有源码的部署方用：先导出一份到磁盘，再拿
// WithAssetDir(dir) 指过去，就能不重新编译地改规则。
//
// 已存在的文件不覆盖，免得冲掉手改过的配置；需要覆盖先自行删除。
func ExportAssets(dir string) ([]string, error) {
	if dir == "" {
		return nil, fmt.Errorf("gobehav: empty export dir")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("gobehav: create %s: %w", dir, err)
	}

	names, err := fs.Glob(assets, assetsDir+"/*"+assetSuffix)
	if err != nil {
		return nil, fmt.Errorf("gobehav: list assets: %w", err)
	}
	sort.Strings(names)

	var written []string
	for _, name := range names {
		dst := filepath.Join(dir, filepath.Base(name))
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		bs, err := assets.ReadFile(name)
		if err != nil {
			return written, fmt.Errorf("gobehav: read %s: %w", name, err)
		}
		if err := os.WriteFile(dst, bs, 0o644); err != nil {
			return written, fmt.Errorf("gobehav: write %s: %w", dst, err)
		}
		written = append(written, dst)
	}
	return written, nil
}
