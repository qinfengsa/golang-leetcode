package sweep

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v line_sweep_main_test.go line_sweep_main.go -run Test_getSkyline
func Test_getSkyline(t *testing.T) {
	buildings := [][]int{{2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}}
	result := getSkyline(buildings)
	fmt.Println(result)
}
