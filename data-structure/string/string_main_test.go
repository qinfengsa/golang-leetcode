package string

import (
	"fmt"
	"testing"
)

func Test_longestPalindrome(t *testing.T) {
	s := "babad"
	result := longestPalindrome(s)
	fmt.Println(result)
}

// test 命令
// go test -v string_main_test.go string_main.go -test.run Test_convert
func Test_convert(t *testing.T) {
	var s = "PAYPALISHIRING"
	var numRows = 3
	result := convert(s, numRows)
	fmt.Println(result)
}

func Test_findSubstring(t *testing.T) {
	s := "barfoothefoobarman"
	words := []string{"foo", "bar"}
	result := findSubstring(s, words)
	fmt.Println(result)
}

func Test_multiply(t *testing.T) {
	num1, num2 := "123", "456"
	result := multiply(num1, num2)
	fmt.Println(result)
}

func Test_simplifyPath(t *testing.T) {
	path := "/a/./b/../../c/"
	result := simplifyPath(path)
	fmt.Println(result)
}
