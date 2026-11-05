// Package bge 把 bge-m3 的 ONNX 编码器和标签原型封装成一个 gobehav.Fallback。
//
// 它只解决一件事：规则层一条标签都没打出来的句子，用向量相似度补一个
// 语义最近的标签。正常路径不经过这里，所以它的性能与依赖都不会影响主链路。
//
// # 依赖
//
// 需要 cgo 和一个可用的 libonnxruntime 动态库（onnxruntime-gpu 的 wheel 里
// 就带，路径形如 .../site-packages/onnxruntime/capi/libonnxruntime.so.1.31.1）。
// 没有 cgo 时本包只编译纯 Go 部分，不会拖累 gobehav 主包。
//
// # 用法
//
//	enc, err := bge.New(bge.Config{
//		ModelPath:      "assets/bge_m3/encoder.int8.onnx",
//		TokenizerPath:  "assets/bge_m3/tokenizer.json",
//		PrototypesPath: "assets/bge_m3/label_prototypes.f32",
//		IndexPath:      "assets/bge_m3/label_index.json",
//		LibraryPath:    "/usr/lib/libonnxruntime.so.1.31.1",
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer enc.Close()
//
//	a := gobehav.MustNew(gobehav.WithFallback(enc))
//
// 模型与原型由 tools/gobehav/export_bge_m3.py 生成，见 tools/gobehav/README.md。
package bge
