package twosum

import "math"

// TwoSum
// 170. 两数之和 III - 数据结构设计
// 设计一个接收整数流的数据结构，该数据结构支持检查是否存在两数之和等于特定值。
//
// 实现 TwoSum 类：
//
// TwoSum() 使用空数组初始化 TwoSum 对象
// void add(int number) 向数据结构添加一个数 number
// boolean find(int value) 寻找数据结构中是否存在一对整数，使得两数之和与给定的值相等。如果存在，返回 true ；否则，返回 false 。
//
// 示例：
// 输入：
// ["TwoSum", "add", "add", "add", "find", "find"]
// [[], [1], [3], [5], [4], [7]]
// 输出：
// [null, null, null, null, true, false]
//
// 解释：
// TwoSum twoSum = new TwoSum();
// twoSum.add(1);   // [] --> [1]
// twoSum.add(3);   // [1] --> [1,3]
// twoSum.add(5);   // [1,3] --> [1,3,5]
// twoSum.find(4);  // 1 + 3 = 4，返回 true
// twoSum.find(7);  // 没有两个整数加起来等于 7 ，返回 false
//
// 提示：
// -105 <= number <= 105
// -231 <= value <= 231 - 1
// 最多调用 104 次 add 和 find
type TwoSum struct {
	val            map[int]int
	maxVal, minVal int
}

func Constructor() TwoSum {
	return TwoSum{val: make(map[int]int), minVal: math.MaxInt32, maxVal: math.MinInt32}
}

func (this *TwoSum) Add(number int) {
	this.minVal = min(this.minVal, number)
	this.maxVal = max(this.maxVal, number)
	this.val[number]++
}

func (this *TwoSum) Find(value int) bool {
	if len(this.val) == 0 {
		return false
	}
	if value < this.minVal<<1 || value > this.maxVal<<1 {
		return false
	}
	for k, v := range this.val {
		if k == value-k {
			if v > 1 {
				return true
			}
		} else {
			if this.val[value-k] > 0 {
				return true
			}
		}

	}

	return false
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
