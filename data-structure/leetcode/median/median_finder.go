package median

import (
	"container/heap"
	"sort"
)

// MedianFinder
// 295. 数据流的中位数
// 中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。
//
// 例如，
// [2,3,4] 的中位数是 3
// [2,3] 的中位数是 (2 + 3) / 2 = 2.5
//
// 设计一个支持以下两种操作的数据结构：
//
// void addNum(int num) - 从数据流中添加一个整数到数据结构中。
// double findMedian() - 返回目前所有元素的中位数。
//
// 示例：
// addNum(1)
// addNum(2)
// findMedian() -> 1.5
// addNum(3)
// findMedian() -> 2
//
// 进阶:
// 如果数据流中所有整数都在 0 到 100 范围内，你将如何优化你的算法？
// 如果数据流中 99% 的整数都在 0 到 100 范围内，你将如何优化你的算法？
type MedianFinder struct {
	minHp, maxHp hp
}

type hp struct{ sort.IntSlice }

func (h *hp) Push(v interface{}) { h.IntSlice = append(h.IntSlice, v.(int)) }
func (h *hp) Pop() interface{} {
	a := h.IntSlice
	v := a[len(a)-1]
	h.IntSlice = a[:len(a)-1]
	return v
}

func Constructor() MedianFinder {
	return MedianFinder{}
}

func (this *MedianFinder) AddNum(num int) {
	maxH, minH := &this.maxHp, &this.minHp
	// minH 用负数表示递减
	if minH.Len() == 0 || num <= -minH.IntSlice[0] {
		heap.Push(minH, -num)
		if maxH.Len()+1 < minH.Len() {
			heap.Push(maxH, -heap.Pop(minH).(int))
		}
	} else {
		heap.Push(maxH, num)
		if maxH.Len() > minH.Len() {
			heap.Push(minH, -heap.Pop(maxH).(int))
		}
	}
}

func (this *MedianFinder) FindMedian() float64 {
	maxH, minH := this.maxHp, this.minHp
	if minH.Len() > maxH.Len() {
		return float64(-minH.IntSlice[0])
	}

	return float64(maxH.IntSlice[0]-minH.IntSlice[0]) / 2.0
}
