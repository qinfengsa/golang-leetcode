package math

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v calculation_main_test.go calculation_main.go -run Test_evalRPN
func Test_evalRPN(t *testing.T) {
	tokens := []string{"10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"}
	result := evalRPN(tokens)
	fmt.Println(result)
}
