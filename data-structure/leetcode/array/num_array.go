package array

type NumArray struct {
	sums []int
}

func Constructor(nums []int) NumArray {
	size := len(nums)
	sums := make([]int, size+1)
	for i := 0; i < size; i++ {
		sums[i+1] = nums[i] + sums[i]
	}
	return NumArray{sums: sums}
}

func (this *NumArray) SumRange(i int, j int) int {
	sums := this.sums
	return sums[j+1] - sums[i]
}
