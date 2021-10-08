package array

type NumArray struct {
	sums []int
}

func Constructor(nums []int) NumArray {
	n := len(nums)
	sums := make([]int, n+1)
	for i := 0; i < n; i++ {
		sums[i+1] = nums[i] + sums[i]
	}
	return NumArray{sums: sums}
}

func (this *NumArray) SumRange(i int, j int) int {
	sums := this.sums
	return sums[j+1] - sums[i]
}
