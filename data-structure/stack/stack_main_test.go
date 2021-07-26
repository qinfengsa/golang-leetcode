package stack

import (
	"fmt"
	"testing"
)

// go test -v stack_main_test.go stack_main.go -test.run Test_largestRectangleArea
func Test_largestRectangleArea(t *testing.T) {
	heights := []int{2, 1, 5, 6, 2, 3}
	fmt.Println(largestRectangleArea(heights))
}
