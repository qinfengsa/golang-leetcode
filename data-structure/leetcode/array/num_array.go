package array

// NumArray
// 307. 区域和检索 - 数组可修改
// 给你一个数组 nums ，请你完成两类查询，其中一类查询要求更新数组下标对应的值，另一类查询要求返回数组中某个范围内元素的总和。
//
// 实现 NumArray 类：
// NumArray(int[] nums) 用整数数组 nums 初始化对象
// void update(int index, int val) 将 nums[index] 的值更新为 val
// int sumRange(int left, int right) 返回子数组 nums[left, right] 的总和（即，nums[left] + nums[left + 1], ..., nums[right]）
//
// 示例：
// 输入：
// ["NumArray", "sumRange", "update", "sumRange"]
// [[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
// 输出：
// [null, 9, null, 8]
// 解释：
// NumArray numArray = new NumArray([1, 3, 5]);
// numArray.sumRange(0, 2); // 返回 9 ，sum([1,3,5]) = 9
// numArray.update(1, 2);   // nums = [1,2,5]
// numArray.sumRange(0, 2); // 返回 8 ，sum([1,2,5]) = 8
//
// 提示：
// 1 <= nums.length <= 3 * 104
// -100 <= nums[i] <= 100
// 0 <= index < nums.length
// -100 <= val <= 100
// 0 <= left <= right < nums.length
// 最多调用 3 * 104 次 update 和 sumRange 方法
type NumArray struct {
	nums []int
	sums []int
}

func Constructor(nums []int) NumArray {
	n := len(nums)
	this := NumArray{nums: make([]int, n), sums: make([]int, n+1)}

	for i := 0; i < n; i++ {
		this.Update(i, nums[i])
	}
	return this
}
func lowBit(x int) int {
	return x & -x
}
func (this *NumArray) getSum(x int) int {
	result := 0
	for x > 0 {
		result += this.sums[x]
		x -= lowBit(x)
	}
	return result
}

func (this *NumArray) Update(index int, val int) {
	div := val - this.nums[index]
	n := len(this.nums)
	x := index + 1
	for x <= n {
		this.sums[x] += div
		x += lowBit(x)
	}
	this.nums[index] = val
}

func (this *NumArray) SumRange(i int, j int) int {
	return this.getSum(j+1) - this.getSum(i)
}
