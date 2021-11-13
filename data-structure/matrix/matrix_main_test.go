package matrix

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v string_main_test.go string_main.go -run Test_convert
func Test_findDiagonalOrder(t *testing.T) {
	mat := [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	result := findDiagonalOrder(mat)
	fmt.Println(result)
}
