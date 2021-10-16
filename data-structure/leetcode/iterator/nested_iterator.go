package iterator

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

// This is the interface that allows for creating nested lists.
// You should not implement it, or speculate about its implementation
type NestedInteger struct {
}

// Return true if this NestedInteger holds a single integer, rather than a nested list.
func (this NestedInteger) IsInteger() bool { return false }

// Return the single integer that this NestedInteger holds, if it holds a single integer
// The result is undefined if this NestedInteger holds a nested list
// So before calling this method, you should have a check
func (this NestedInteger) GetInteger() int { return 0 }

// Set this NestedInteger to hold a single integer.
func (n *NestedInteger) SetInteger(value int) {}

// Set this NestedInteger to hold a nested list and adds a nested integer to it.
func (this *NestedInteger) Add(elem NestedInteger) {}

// Return the nested list that this NestedInteger holds, if it holds a nested list
// The list length is zero if this NestedInteger holds a single integer
// You can access NestedInteger's List element directly if you want to modify it
func (this NestedInteger) GetList() []*NestedInteger { return nil }
