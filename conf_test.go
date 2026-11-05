package gobehav_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/xxjwxc/gobehav"
)

// minAsset 一份最小可用的配置，用来测目录加载与热加载。
const minAsset = `version: 1
lang: cn
tags:
  behavior_apology:
    id: 1
    p: 1
  mood_happy:
    id: 2
    p: 2
phrases:
  对不起: behavior_apology
`

// conf/ 是给人改的配置目录：只有二进制、没有源码的部署方也要能拿到同一份。
func TestExportAssets(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "conf")

	written, err := gobehav.ExportAssets(dir)
	if err != nil {
		t.Fatalf("ExportAssets: %v", err)
	}
	if len(written) == 0 {
		t.Fatal("没有导出任何文件")
	}
	for _, p := range written {
		bs, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("读导出的 %s: %v", p, err)
		}
		if len(bs) == 0 {
			t.Fatalf("导出的 %s 是空文件", p)
		}
	}

	// 导出的副本要能直接当配置用。
	a, err := gobehav.New(gobehav.WithAssetDir(dir), gobehav.WithWatchInterval(0))
	if err != nil {
		t.Fatalf("用导出目录构造: %v", err)
	}
	defer a.Close()
	if len(a.Langs()) == 0 {
		t.Fatal("导出目录里没有可用语种")
	}
	if res := a.Analyze("对不起，我真的很抱歉"); len(res.Tags) == 0 {
		t.Fatal("导出目录构造出来的分析器打不出标签")
	}

	// 已存在就不覆盖，免得冲掉手改过的配置。
	again, err := gobehav.ExportAssets(dir)
	if err != nil {
		t.Fatalf("二次 ExportAssets: %v", err)
	}
	if len(again) != 0 {
		t.Fatalf("已存在时不应再写，实际写了 %v", again)
	}
}

func TestExportAssetsRejectsEmptyDir(t *testing.T) {
	if _, err := gobehav.ExportAssets(""); err == nil {
		t.Fatal("空目录应当报错")
	}
}

// 改 conf/ 下的文件要能不重启就生效，否则「用户可以改配置」就是句空话。
func TestAssetDirHotReload(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "behav_cn.yaml")
	if err := os.WriteFile(file, []byte(minAsset), 0o644); err != nil {
		t.Fatal(err)
	}

	a, err := gobehav.New(
		gobehav.WithAssetDir(dir),
		gobehav.WithWatchInterval(20*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer a.Close()

	if got := a.Analyze("这事儿整挺好").Tags; len(got) != 0 {
		t.Fatalf("改配置之前不该命中: %+v", got)
	}

	// 追加一条短语，模拟运维直接编辑配置文件。
	updated := minAsset + "  这事儿整挺好: mood_happy\n"
	if err := os.WriteFile(file, []byte(updated), 0o644); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(5 * time.Second)
	for {
		res := a.Analyze("这事儿整挺好")
		if len(res.Tags) == 1 && res.Tags[0].Tag == "mood_happy" {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("5s 内没热加载到新短语，最后一次: %+v", res.Tags)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// 配置目录写错路径要立刻报错，不能静默退化成内嵌配置。
func TestAssetDirMissing(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "nope")
	if _, err := gobehav.New(gobehav.WithAssetDir(missing)); err == nil {
		t.Fatal("目录不存在应当报错")
	}
}
