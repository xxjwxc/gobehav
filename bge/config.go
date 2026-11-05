package bge

// 模型文件的固定文件名，NewFromDir 按这四个名字去找。
//
// 放在本文件（不带 build tag）里，关掉 cgo 时主包的目录探测也要用到。
const (
	ModelFile      = "encoder.int8.onnx"
	TokenizerFile  = "tokenizer.json"
	PrototypesFile = "label_prototypes.f32"
	IndexFile      = "label_index.json"
)

const (
	// DefaultThreshold 采用标签原型打分时建议的采纳阈值。
	//
	// 拿 3152 条短语回查自己的标签，0.62 上覆盖率 76%、命中精度 68.5%；
	// 往下调会开始乱标，往上调精度涨得比覆盖率掉得慢。
	DefaultThreshold = 0.62

	// DefaultMaxLen 单条文本最多的 token 数（含 [CLS]/[SEP]）。
	DefaultMaxLen = 64

	// SpanPriority 兜底标签的优先级。取值明显大于内置标签（1~10 量级），
	// 万一调用方把兜底结果和规则结果混在一起排序，兜底不会盖住规则。
	SpanPriority = 100
)

// Config 加载编码器所需的文件与参数。
//
// 一般不用自己拼这堆路径，直接 bge.NewFromDir(dir) 即可。
type Config struct {
	// ModelPath encoder.int8.onnx 的路径。
	ModelPath string
	// TokenizerPath tokenizer.json 的路径，必须与模型配套。
	TokenizerPath string
	// PrototypesPath label_prototypes.f32 的路径。
	PrototypesPath string
	// IndexPath label_index.json 的路径，给原型矩阵提供行序与维度。
	IndexPath string
	// LibraryPath onnxruntime 动态库的路径，留空就自动找：
	// 模型目录 -> exe 目录 -> 当前目录 -> 系统库目录（/usr/lib、/usr/local/lib 等）。
	// Linux 认 libonnxruntime.so*，Windows 认 onnxruntime.dll。
	LibraryPath string
	// Threshold 采纳阈值，0 表示用 DefaultThreshold。
	Threshold float32
	// MaxLen 单条最多 token 数，0 表示用 DefaultMaxLen。
	MaxLen int
}
