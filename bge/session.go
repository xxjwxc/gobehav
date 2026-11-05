//go:build cgo

package bge

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// onnxEncoder 包一层 onnxruntime。
//
// 用 DynamicAdvancedSession 是因为 bge-m3 的输入形状是动态的（batch 和 seq
// 都不定），预分配固定形状的 tensor 会逼着调用方把每条都 padding 到同一长度。
type onnxEncoder struct {
	sess *ort.DynamicAdvancedSession
	dim  int

	// ORT 的 Run 允许多线程并发，但兜底是低频路径，串行化能省掉每线程一套
	// 输入输出 tensor，也避免 ORT 内部线程超订。tokenizer 同理。
	mu sync.Mutex
}

// 进程内 onnxruntime 只能初始化一次，重复调用会报错。
//
// 这也意味着 SetSharedLibraryPath 只在第一个 Encoder 上生效：同进程里先建一个
// 指向 A 库的 Encoder，再建一个指向 B 库的，第二个会继续用 A。
var (
	envOnce sync.Once
	envErr  error
)

// libPatterns 各平台的动态库文件名，按优先级排列。
//
// Linux 上发行包与 pip wheel 都带版本号（libonnxruntime.so.1.31.1），
// 只用固定名是找不到的，所以先试无后缀的软链，再匹配带版本号的。
func libPatterns() []string {
	switch runtime.GOOS {
	case "windows":
		return []string{"onnxruntime.dll"}
	case "darwin":
		return []string{"libonnxruntime.dylib", "libonnxruntime*.dylib"}
	default:
		return []string{"libonnxruntime.so", "libonnxruntime.so.*"}
	}
}

// systemLibDirs 系统库目录。Windows 由 LoadLibrary 自己搜 exe 目录与 PATH，
// 这里只列 Linux/macOS 常见的几个。
var systemLibDirs = []string{
	"/usr/lib",
	"/usr/lib/x86_64-linux-gnu",
	"/usr/lib/aarch64-linux-gnu",
	"/usr/local/lib",
	"/opt/homebrew/lib",
}

// defaultLibName 实在找不到时的名字，交给系统加载器解析。
func defaultLibName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so.1"
	}
}

// exeDir 当前可执行文件所在目录，拿不到就返回空。
func exeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}

	return filepath.Dir(exe)
}

// findLibrary 在某个目录下按平台文件名找一个动态库。
func findLibrary(dir string) string {
	if dir == "" {
		return ""
	}

	for _, pat := range libPatterns() {
		matches, err := filepath.Glob(filepath.Join(dir, pat))
		if err != nil || len(matches) == 0 {
			continue
		}
		sort.Strings(matches)

		return matches[len(matches)-1]
	}

	return ""
}

// resolveLibrary 找 onnxruntime 动态库。顺序：
//
// 显式路径 -> 模型目录 -> exe 目录 -> 当前目录 -> 系统库目录 -> 平台默认名
//
// exe 目录和当前目录都算上，是为了让「把 dll/so 跟程序放一起」这种最省事的
// 部署方式直接能跑，不用配 LD_LIBRARY_PATH。
func resolveLibrary(libPath, modelDir string) string {
	if libPath != "" {
		return libPath
	}

	for _, dir := range []string{modelDir, exeDir(), "."} {
		if p := findLibrary(dir); p != "" {
			return p
		}
	}

	for _, dir := range systemLibDirs {
		if p := findLibrary(dir); p != "" {
			return p
		}
	}

	return defaultLibName()
}

func newONNXEncoder(modelPath, libPath, modelDir string, dim int) (*onnxEncoder, error) {
	envOnce.Do(func() {
		ort.SetSharedLibraryPath(resolveLibrary(libPath, modelDir))
		envErr = ort.InitializeEnvironment()
	})
	if envErr != nil {
		return nil, fmt.Errorf("bge: 初始化 onnxruntime 失败（把动态库放到模型目录或 exe 旁边就会自动加载）: %w", envErr)
	}

	sess, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"sentence_embedding"}, nil)
	if err != nil {
		return nil, fmt.Errorf("bge: 加载 %s 失败: %w", modelPath, err)
	}
	return &onnxEncoder{sess: sess, dim: dim}, nil
}

// run 跑一个 batch。ids/mask 按 [batch, seq] 行优先展开。
func (e *onnxEncoder) run(ids, mask []int64, batch, seq int) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	shape := ort.NewShape(int64(batch), int64(seq))

	idTensor, err := ort.NewTensor(shape, ids)
	if err != nil {
		return nil, fmt.Errorf("bge: 构造 input_ids tensor 失败: %w", err)
	}
	defer idTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape, mask)
	if err != nil {
		return nil, fmt.Errorf("bge: 构造 attention_mask tensor 失败: %w", err)
	}
	defer maskTensor.Destroy()

	outTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batch), int64(e.dim)))
	if err != nil {
		return nil, fmt.Errorf("bge: 构造输出 tensor 失败: %w", err)
	}
	defer outTensor.Destroy()

	if err := e.sess.Run(
		[]ort.Value{idTensor, maskTensor},
		[]ort.Value{outTensor},
	); err != nil {
		return nil, fmt.Errorf("bge: 推理失败: %w", err)
	}

	data := outTensor.GetData()
	vecs := make([][]float32, batch)
	for i := range vecs {
		vec := make([]float32, e.dim)
		copy(vec, data[i*e.dim:(i+1)*e.dim])
		vecs[i] = vec
	}
	return vecs, nil
}

func (e *onnxEncoder) close() error {
	if e.sess == nil {
		return nil
	}
	return e.sess.Destroy()
}
