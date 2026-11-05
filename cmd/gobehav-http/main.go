// Command gobehav-http 把 gobehav 包成一个只用标准库的 JSON 服务，
// 方便非 Go 调用方（含原来的 haitag 之外的脚本）验证效果。
//
//	go run ./cmd/gobehav-http -addr :8080
//
// 默认读 ./conf 下的配置并每秒热加载，改规则不用重启；
// 目录不存在时自动退回包内嵌配置。
//
// 语义兜底（规则一条都没命中时用 bge-m3 向量补一个标签）默认开启：
// 程序同级放一个 bge_m3 目录，里面放四个模型文件 + onnxruntime 动态库即可。
//
//	encoder.int8.onnx / tokenizer.json / label_prototypes.f32 / label_index.json
//	libonnxruntime.so*（Linux）或 onnxruntime.dll（Windows）
//
// 模型目录不在默认位置时用 -model 指定，不想用兜底用 -model none。
// 带兜底需要 CGO_ENABLED=1 编译（Windows 需 mingw-gcc）。
//
// 接口：
//
//	POST /analyze   {"text":["..."],"lang":"cn","hot":[["pattern","tag",1]]}
//	GET  /langs
//	GET  /healthz
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"
	"time"
	"unicode/utf8"

	"github.com/xxjwxc/gobehav"
)

type analyzeRequest struct {
	Text []string            `json:"text"`
	Lang string              `json:"lang"`
	Hot  [][]json.RawMessage `json:"hot"` // [pattern, tag] 或 [pattern, tag, priority]
}

type analyzeResponse struct {
	Results []gobehav.Result `json:"results"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	addr := flag.String("addr", ":8080", "监听地址")
	assetDir := flag.String("asset", "conf", "配置目录（内含 behav_<lang>.yaml），留空用包内嵌配置")
	lang := flag.String("lang", "cn", "默认语种")
	watch := flag.Duration("watch", time.Second, "配置热加载间隔，0 关闭")
	modelDir := flag.String("model", "", "bge-m3 模型目录，留空自动探测（bge_m3 / assets/bge_m3 / models/bge_m3），none 表示不用兜底")
	flag.Parse()

	// 不传 logger 的话热重载成功也是静默的，改完配置看不见任何反馈，
	// 排查时很难分清是「没重载」还是「重载了但规则没命中」。
	opts := []gobehav.Option{
		gobehav.WithDefaultLang(*lang),
		gobehav.WithLogger(slog.Default()),
	}
	if *assetDir != "" {
		// 目录不在就用内嵌配置，不因为部署时忘带 conf 就起不来。
		if fi, err := os.Stat(*assetDir); err == nil && fi.IsDir() {
			opts = append(opts, gobehav.WithAssetDir(*assetDir), gobehav.WithWatchInterval(*watch))
			log.Printf("config dir: %s (hot reload %s)", *assetDir, *watch)
		} else {
			log.Printf("config dir %q 不可用，改用包内嵌配置", *assetDir)
		}
	}

	// 兜底默认开启：模型目录与动态库都在约定位置时自动加载，
	// 缺模型、缺库、没开 cgo 都只是退化成纯规则，不影响启动。
	switch *modelDir {
	case "none":
		opts = append(opts, gobehav.WithoutModel())
		log.Printf("语义兜底已关闭，只用规则层")
	case "":
		// 自动探测
	default:
		opts = append(opts, gobehav.WithModelDir(*modelDir))
	}

	a, err := gobehav.New(opts...)
	if err != nil {
		log.Fatalf("init gobehav: %v", err)
	}
	defer a.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/analyze", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "use POST")
			return
		}

		var req analyzeRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "bad json: "+err.Error())
			return
		}
		if len(req.Text) == 0 {
			writeError(w, http.StatusBadRequest, "text is empty")
			return
		}
		// 非 UTF-8 请求体会被 json 解码成 U+FFFD，看上去能跑但结果全错，
		// 这里直接拒掉，把编码问题暴露给调用方。
		for i, s := range req.Text {
			if !utf8.ValidString(s) {
				writeError(w, http.StatusBadRequest,
					fmt.Sprintf("text[%d]: 不是合法 UTF-8，请求体请用 UTF-8 编码", i))
				return
			}
		}

		o, err := req.callOptions()
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		results := a.Batch(req.Text, o)
		// 没命中时 Tags 是 nil，对外统一输出 []，避免调用方拿到 null。
		for i := range results {
			if results[i].Tags == nil {
				results[i].Tags = []gobehav.Span{}
			}
		}

		writeJSON(w, http.StatusOK, analyzeResponse{Results: results})
	})

	mux.HandleFunc("/langs", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string][]string{"langs": a.Langs()})
	})

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "ok")
	})

	log.Printf("gobehav-http listening on %s, langs=%v", *addr, a.Langs())
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

// callOptions 把 [pattern, tag(, priority)] 数组形式的热规则转成调用选项。
// 用数组而不是对象，是为了和 WeTextGo 的 hotRegex 保持一致，请求体更短。
func (r analyzeRequest) callOptions() (gobehav.CallOptions, error) {
	o := gobehav.CallOptions{Lang: r.Lang}

	for i, item := range r.Hot {
		if len(item) < 2 {
			return o, fmt.Errorf("hot[%d]: 需要 [pattern, tag] 或 [pattern, tag, priority]", i)
		}

		var pattern, tag string
		if err := json.Unmarshal(item[0], &pattern); err != nil {
			return o, fmt.Errorf("hot[%d].pattern: %w", i, err)
		}
		if err := json.Unmarshal(item[1], &tag); err != nil {
			return o, fmt.Errorf("hot[%d].tag: %w", i, err)
		}
		if pattern == "" || tag == "" {
			return o, fmt.Errorf("hot[%d]: pattern 与 tag 不能为空", i)
		}
		if !utf8.ValidString(pattern) {
			return o, fmt.Errorf("hot[%d].pattern: 不是合法 UTF-8", i)
		}

		var prio int32
		if len(item) > 2 {
			if err := json.Unmarshal(item[2], &prio); err != nil {
				return o, fmt.Errorf("hot[%d].priority: %w", i, err)
			}
		}

		o.Hot = append(o.Hot, gobehav.HotRule{Pattern: pattern, Tag: tag, Priority: prio})
	}

	return o, nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("encode response: %v", err)
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, errorResponse{Error: msg})
}
