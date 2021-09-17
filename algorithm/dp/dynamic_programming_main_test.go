package dp

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v dynamic_programming_main_test.go dynamic_programming_main.go -run Test_wordBreakII
func Test_wordBreakII(t *testing.T) {
	s := "catsanddog"
	wordDict := []string{"cat", "cats", "and", "sand", "dog"}

	result := wordBreakII(s, wordDict)
	fmt.Println(result)
}
