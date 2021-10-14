package bit

import "sort"

type BIT struct {
	n    int
	nums []int
}

func lowBit(x int) int {
	return x & -x
}

func (this *BIT) inc(x int) {
	for ; x < len(this.nums); x += lowBit(x) {
		this.nums[x]++
	}
}

func (this *BIT) update(x, val int) {

}

func (this *BIT) query(left, right int) int {
	return this.sum(right) - this.sum(left-1)
}

func (this *BIT) sum(x int) int {
	result := 0
	for x > 0 {
		result += this.nums[x]
		x -= lowBit(x)
	}
	return result
}

// 327. 区间和的个数
// 给你一个整数数组 nums 以及两个整数 lower 和 upper 。求数组中，值位于范围 [lower, upper] （包含 lower 和 upper）之内的 区间和的个数 。
//
// 区间和 S(i, j) 表示在 nums 中，位置从 i 到 j 的元素之和，包含 i 和 j (i ≤ j)。
//
// 示例 1：
// 输入：nums = [-2,5,-1], lower = -2, upper = 2
// 输出：3
// 解释：存在三个区间：[0,0]、[2,2] 和 [0,2] ，对应的区间和分别是：-2 、-1 、2 。
//
// 示例 2：
// 输入：nums = [0], lower = 0, upper = 0 输出：1
//
// 提示：
// 1 <= nums.length <= 105
// -231 <= nums[i] <= 231 - 1
// -105 <= lower <= upper <= 105
// 题目数据保证答案是一个 32 位 的整数
func countRangeSum(nums []int, lower int, upper int) int {

	sum, n := 0, len(nums)
	sums, allNums := make([]int, n+1), make([]int, 1, 3*n+1)
	for i := 0; i < n; i++ {
		sum += nums[i]
		sums[i+1] = sum
		allNums = append(allNums, sum, sum-lower, sum-upper)
	}
	sort.Ints(allNums)

	k := 1
	kth := map[int]int{allNums[0]: k}
	for i := 1; i <= 3*n; i++ {
		if allNums[i] != allNums[i-1] {
			k++
			kth[allNums[i]] = k
		}
	}

	// 遍历 preSum，利用树状数组计算每个前缀和对应的合法区间数
	t := &BIT{nums: make([]int, k+1)}
	t.inc(kth[0])
	count := 0
	for _, num := range sums[1:] {
		left, right := kth[num-upper], kth[num-lower]
		count += t.query(left, right)
		t.inc(kth[num])
	}
	return count
}
