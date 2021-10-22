package iterator

import (
	"strconv"
)

// NestedIterator
//
// 341. 扁平化嵌套列表迭代器
// 给你一个嵌套的整数列表 nestedList 。每个元素要么是一个整数，要么是一个列表；该列表的元素也可能是整数或者是其他列表。请你实现一个迭代器将其扁平化，使之能够遍历这个列表中的所有整数。
//
// 实现扁平迭代器类 NestedIterator ：
//
// NestedIterator(List<NestedInteger> nestedList) 用嵌套列表 nestedList 初始化迭代器。
// int next() 返回嵌套列表的下一个整数。
// boolean hasNext() 如果仍然存在待迭代的整数，返回 true ；否则，返回 false 。
// 你的代码将会用下述伪代码检测：
//
// initialize iterator with nestedList
// res = []
// while iterator.hasNext()
//    append iterator.next() to the end of res
// return res
// 如果 res 与预期的扁平化列表匹配，那么你的代码将会被判为正确。
//
// 示例 1：
// 输入：nestedList = [[1,1],2,[1,1]]
// 输出：[1,1,2,1,1]
// 解释：通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,1,2,1,1]。
//
// 示例 2：
// 输入：nestedList = [1,[4,[6]]]
// 输出：[1,4,6]
// 解释：通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,4,6]。
//
// 提示：
// 1 <= nestedList.length <= 500
// 嵌套列表中的整数值在范围 [-106, 106] 内
type NestedIterator struct {
	nums  []int
	index int
}

func newNestedIterator(nestedList []*NestedInteger) *NestedIterator {
	nums := make([]int, 0)
	var insertNums func(nestedNums []*NestedInteger)

	insertNums = func(nestedNums []*NestedInteger) {
		for _, nestedNum := range nestedNums {
			if nestedNum.IsInteger() {
				nums = append(nums, nestedNum.GetInteger())
			} else {
				insertNums(nestedNum.GetList())
			}
		}
	}
	insertNums(nestedList)

	return &NestedIterator{nums: nums}
}

func (this *NestedIterator) Next() int {
	n := len(this.nums)
	if this.index >= n {
		return -1
	}
	val := this.nums[this.index]
	this.index++
	return val
}

func (this *NestedIterator) HasNext() bool {
	n := len(this.nums)
	return this.index < n
}

// NestedInteger This is the interface that allows for creating nested lists.
// You should not implement it, or speculate about its implementation
type NestedInteger struct {
}

// IsInteger Return true if this NestedInteger holds a single integer, rather than a nested list.
func (this NestedInteger) IsInteger() bool { return false }

// GetInteger Return the single integer that this NestedInteger holds, if it holds a single integer
// The result is undefined if this NestedInteger holds a nested list
// So before calling this method, you should have a check
func (this NestedInteger) GetInteger() int { return 0 }

// SetInteger Set this NestedInteger to hold a single integer.
func (n *NestedInteger) SetInteger(value int) {}

// Add Set this NestedInteger to hold a nested list and adds a nested integer to it.
func (this *NestedInteger) Add(elem NestedInteger) {}

// GetList Return the nested list that this NestedInteger holds, if it holds a nested list
// The list length is zero if this NestedInteger holds a single integer
// You can access NestedInteger's List element directly if you want to modify it
func (this NestedInteger) GetList() []*NestedInteger { return nil }

// 385. 迷你语法分析器
// 给定一个用字符串表示的整数的嵌套列表，实现一个解析它的语法分析器。
//
// 列表中的每个元素只可能是整数或整数嵌套列表
// 提示：你可以假定这些字符串都是格式良好的：
//
// 字符串非空
// 字符串不包含空格
// 字符串只包含数字0-9、[、-、,、]
//
// 示例 1：
// 给定 s = "324",
// 你应该返回一个 NestedInteger 对象，其中只包含整数值 324。
//
// 示例 2：
// 给定 s = "[123,[456,[789]]]",
// 返回一个 NestedInteger 对象包含一个有两个元素的嵌套列表：
// 1. 一个 integer 包含值 123
// 2. 一个包含两个元素的嵌套列表：
//    i.  一个 integer 包含值 456
//    ii. 一个包含一个元素的嵌套列表
//         a. 一个 integer 包含值 789
func deserialize(s string) *NestedInteger {

	result := &NestedInteger{}
	if s[0] != '[' {
		num, _ := strconv.Atoi(s)
		result.SetInteger(num)
		return result
	}

	var dfs func() NestedInteger
	idx, n := 0, len(s)
	dfs = func() NestedInteger {
		nest := NestedInteger{}

		negative := false
		num := 0
		for idx < n-1 {
			idx++
			if s[idx] == ',' {
				continue
			} else if s[idx] == '[' {
				nest.Add(dfs())
			} else if s[idx] == ']' {
				return nest
			} else if s[idx] == '-' {
				negative = true
			} else {
				num = num*10 + int(s[idx]-'0')
				if idx < n-1 && (s[idx+1] == ',' || s[idx+1] == ']') {
					child := NestedInteger{}
					if negative {
						num *= -1
					}
					child.SetInteger(num)
					nest.Add(child)
					negative = false
					num = 0
				}
			}

		}
		return nest
	}
	nest := dfs()
	return &nest
}

func isNum(c byte) bool {
	return c >= '0' && c <= '9'
}
