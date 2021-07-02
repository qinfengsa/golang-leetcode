package back

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v string_main_test.go string_main.go -test.run Test_convert

func Test_letterCombinations(t *testing.T) {
	digits := "23"
	result := letterCombinations(digits)
	fmt.Println(result)
}

func Test_generateParenthesis(t *testing.T) {
	n := 3
	result := generateParenthesis(n)
	fmt.Println(result)
}
