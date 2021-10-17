package ranges

import (
	"github.com/emirpasic/gods/trees/redblacktree"
)

// SummaryRanges
//
// 352. 将数据流变为多个不相交区间
// 给你一个由非负整数 a1, a2, ..., an 组成的数据流输入，请你将到目前为止看到的数字总结为不相交的区间列表。
//
// 实现 SummaryRanges 类：
// SummaryRanges() 使用一个空数据流初始化对象。
// void addNum(int val) 向数据流中加入整数 val 。
// int[][] getIntervals() 以不相交区间 [starti, endi] 的列表形式返回对数据流中整数的总结。
//
// 示例：
// 输入：
// ["SummaryRanges", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals"]
// [[], [1], [], [3], [], [7], [], [2], [], [6], []]
// 输出：
// [null, null, [[1, 1]], null, [[1, 1], [3, 3]], null, [[1, 1], [3, 3], [7, 7]], null, [[1, 3], [7, 7]], null, [[1, 3], [6, 7]]]
// 解释：
// SummaryRanges summaryRanges = new SummaryRanges();
// summaryRanges.addNum(1);      // arr = [1]
// summaryRanges.getIntervals(); // 返回 [[1, 1]]
// summaryRanges.addNum(3);      // arr = [1, 3]
// summaryRanges.getIntervals(); // 返回 [[1, 1], [3, 3]]
// summaryRanges.addNum(7);      // arr = [1, 3, 7]
// summaryRanges.getIntervals(); // 返回 [[1, 1], [3, 3], [7, 7]]
// summaryRanges.addNum(2);      // arr = [1, 2, 3, 7]
// summaryRanges.getIntervals(); // 返回 [[1, 3], [7, 7]]
// summaryRanges.addNum(6);      // arr = [1, 2, 3, 6, 7]
// summaryRanges.getIntervals(); // 返回 [[1, 3], [6, 7]]
//
// 提示：
// 0 <= val <= 104
// 最多调用 addNum 和 getIntervals 方法 3 * 104 次
//
// 进阶：如果存在大量合并，并且与数据流的大小相比，不相交区间的数量很小，该怎么办?
type SummaryRanges struct {
	// 红黑树的每个结点的key为每个区间的左端点，value为每个区间的右端点
	*redblacktree.Tree
	head *Node
}

type Node struct {
	start, end int
	next       *Node
}

func Constructor() SummaryRanges {
	tree := redblacktree.NewWithIntComparator()
	node := &Node{start: -2, end: -2}
	tree.Put(-2, node)
	return SummaryRanges{tree, node}
}

func (this *SummaryRanges) AddNum(val int) {
	if this.Size() == 1 {
		node := &Node{start: val, end: val}
		this.Put(val, node)
		this.head.next = node
		return
	}
	// 找左边的节点 肯定存在
	left, _ := this.Floor(val)
	right, rightFound := this.Ceiling(val)
	leftNode := left.Value.(*Node)
	if val <= leftNode.end {
		return
	}
	if val == leftNode.end+1 {
		leftNode.end = val
	} else {
		node := &Node{start: val, end: val}
		this.Put(val, node)
		leftNode.next = node
		leftNode = node
	}
	// 判断右边的节点能不能合并
	if rightFound {
		rightNode := right.Value.(*Node)
		if leftNode.end+1 == rightNode.start {
			// 合并
			leftNode.end = rightNode.end
			leftNode.next = rightNode.next
			this.Remove(right.Key)
		} else {
			leftNode.next = rightNode
		}
	}

}

func (this *SummaryRanges) GetIntervals() [][]int {
	n := this.Size() - 1
	result := make([][]int, n)
	node := this.head.next
	for i := 0; i < n; i++ {
		result[i] = make([]int, 2)
		result[i][0] = node.start
		result[i][1] = node.end
		node = node.next
	}
	return result
}
