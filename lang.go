package gobehav

import (
	"strings"
	"unicode"
)

// normLang 归一化语种写法：zh / zh-CN / cn 都当成 cn。
// 返回空串表示调用方没有指定语种。
func normLang(lang string) string {
	s := strings.ToLower(strings.TrimSpace(lang))

	switch s {
	case "", "auto":
		return ""
	case "zh", "zh-cn", "zh_cn", "zh-hans", "chs", "cmn":
		return "cn"
	}

	return s
}

// detectLang 按字符集猜语种。
//
// 只用来兜底：汉字 -> cn，天城文 -> hi，其余拉丁字母文本按 en 处理。
// 中文里夹几个英文单词、数字都不会误判。
func detectLang(text string) string {
	for _, r := range text {
		switch {
		case unicode.Is(unicode.Han, r):
			return "cn"
		case unicode.Is(unicode.Devanagari, r):
			return "hi"
		}
	}

	return "en"
}
