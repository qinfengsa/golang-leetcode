package search

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v string_main_test.go string_main.go -test.run Test_convert
func Test_searchII(t *testing.T) {
	nums := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1}
	target := 2
	fmt.Println(searchII(nums, target))
}
