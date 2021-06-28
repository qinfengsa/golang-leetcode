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
