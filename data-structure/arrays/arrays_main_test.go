package arrays

import (
	"fmt"
	"testing"
)

func Test_findMedianSortedArrays(t *testing.T) {
	nums1, nums2 := []int{}, []int{1}
	result := findMedianSortedArrays(nums1, nums2)
	fmt.Println(result)
}

// test 命令
// go test -v string_main_test.go string_main.go -test.run Test_convert
func Test_threeSum(t *testing.T) {
	nums := []int{0}
	result := threeSum(nums)
	fmt.Println(result)
}

func Test_threeSumClosest(t *testing.T) {

	nums := []int{1, 1, 1, 0}
	target := -100
	result := threeSumClosest(nums, target)
	fmt.Println(result)
}

func Test_firstMissingPositive(t *testing.T) {
	nums := []int{3, 4, -1, 1}
	result := firstMissingPositive(nums)
	fmt.Println(result)
}

func Test_spiralOrder(t *testing.T) {
	matrix := [][]int{{3}, {2}}
	result := spiralOrder(matrix)
	fmt.Println(result)
}

func Test_minSubArrayLen(t *testing.T) {
	target := 7
	nums := []int{2, 3, 1, 2, 4, 3}
	result := minSubArrayLen(target, nums)
	fmt.Println(result)
}
