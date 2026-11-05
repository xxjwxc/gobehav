package extdemo
// 演示别的项目怎么接入 gobehav：三行拿到打标能力。
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"

	gobehav "github.com/xxjwxc/gobehav"
)

func main() {
	modelDir := flag.String("model", "", "bge_m3 目录，留空自动探测")
	flag.Parse()

	opts := []gobehav.Option{
		gobehav.WithDefaultLang("cn"),
		gobehav.WithLogger(slog.Default()),
	}
	if *modelDir != "" {
		opts = append(opts, gobehav.WithModelDir(*modelDir))
	}
	if os.Getenv("NO_MODEL") != "" {
		opts = append(opts, gobehav.WithoutModel())
	}

	a, err := gobehav.New(opts...)
	if err != nil {
		panic(err)
	}
	defer a.Close()

	fmt.Println("语种:", a.Langs())

	for _, text := range []string{
		"对不起，我真的不想要这个东西",
		"这波操作我是真的服了",
	} {
		res := a.Analyze(text)
		fmt.Printf("\n%s\n  分词: %s\n", res.Orig, res.Tokens)
		runes := []rune(res.Sentence)
		for _, t := range res.Tags {
			fmt.Printf("  [%d,%d) %-20s p=%d 片段=%q\n",
				t.Start, t.End, t.Tag, t.Priority, string(runes[t.Start:t.End]))
		}
	}
}
