package sweep

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
)

// 堆
type pair struct {
	right, height int
}

type hp []pair

func (h hp) Len() int {
	return len(h)
}

func (h hp) Less(i, j int) bool {
	return h[i].height > h[j].height
}

func (h hp) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *hp) Push(x interface{}) {
	*h = append(*h, x.(pair))
}

func (h *hp) Pop() interface{} {
	tmp := *h
	v := tmp[len(tmp)-1]

	*h = tmp[:len(tmp)-1]

	return v
}

// 218. 天际线问题
// 城市的天际线是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。给你所有建筑物的位置和高度，请返回由这些建筑物形成的 天际线 。
//
// 每个建筑物的几何信息由数组 buildings 表示，其中三元组 buildings[i] = [lefti, righti, heighti] 表示：
//
// lefti 是第 i 座建筑物左边缘的 x 坐标。
// righti 是第 i 座建筑物右边缘的 x 坐标。
// heighti 是第 i 座建筑物的高度。
// 天际线 应该表示为由 “关键点” 组成的列表，格式 [[x1,y1],[x2,y2],...] ，并按 x 坐标 进行 排序 。关键点是水平线段的左端点。列表中最后一个点是最右侧建筑物的终点，y 坐标始终为 0 ，仅用于标记天际线的终点。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。
//
// 注意：输出天际线中不得有连续的相同高度的水平线。例如 [...[2 3], [4 5], [7 5], [11 5], [12 7]...] 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：[...[2 3], [4 5], [12 7], ...]
//
// 示例 1：
// 输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
// 输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
// 解释：
// 图 A 显示输入的所有建筑物的位置和高度，
// 图 B 显示由这些建筑物形成的天际线。图 B 中的红点表示输出列表中的关键点。
//
// 示例 2：
// 输入：buildings = [[0,2,3],[2,5,3]]
// 输出：[[0,3],[5,0]]
//
// 提示：
// 1 <= buildings.length <= 104
// 0 <= lefti < righti <= 231 - 1
// 1 <= heighti <= 231 - 1
// buildings 按 lefti 非递减排序
func getSkyline(buildings [][]int) [][]int {
	result := make([][]int, 0)
	n := len(buildings)
	buildX := make([]int, 2*n)
	// 坐标 x
	for i := 0; i < n; i++ {
		buildX[i<<1] = buildings[i][0]
		buildX[(i<<1)+1] = buildings[i][1]
	}
	// 坐标 x 排序
	sort.Ints(buildX)
	h := hp{}
	idx := 0
	for _, x := range buildX {
		// x 左边 的坐标 入堆
		for idx < n && buildings[idx][0] <= x {
			heap.Push(&h, pair{buildings[idx][1], buildings[idx][2]})
			idx++
		}
		// 如果 building 的 右侧 小于 x 移除堆
		for len(h) > 0 && h[0].right <= x {
			heap.Pop(&h)
		}
		// 当前的最大高度
		maxHeight := 0
		if len(h) > 0 {
			maxHeight = h[0].height
		}
		size := len(result)
		fmt.Print(x, "<>", maxHeight, "->")
		fmt.Println(result)
		// 没有 result 或者 高度发生变化
		if size == 0 || maxHeight != result[size-1][1] {
			result = append(result, []int{x, maxHeight})
		}
	}

	return result
}

// 391. 完美矩形
// 我们有 N 个与坐标轴对齐的矩形, 其中 N > 0, 判断它们是否能精确地覆盖一个矩形区域。
//
// 每个矩形用左下角的点和右上角的点的坐标来表示。例如， 一个单位正方形可以表示为 [1,1,2,2]。 ( 左下角的点的坐标为 (1, 1) 以及右上角的点的坐标为 (2, 2) )。
//
// 示例 1:
// rectangles = [
//  [1,1,3,3],
//  [3,1,4,2],
//  [3,2,4,4],
//  [1,3,2,4],
//  [2,3,3,4]
// ]
// 返回 true。5个矩形一起可以精确地覆盖一个矩形区域。
//
// 示例 2:
// rectangles = [
//  [1,1,2,3],
//  [1,3,2,4],
//  [3,1,4,2],
//  [3,2,4,4]
// ]
// 返回 false。两个矩形之间有间隔，无法覆盖成一个矩形。
//
// 示例 3:
// rectangles = [
//  [1,1,3,3],
//  [3,1,4,2],
//  [1,3,2,4],
//  [3,2,4,4]
// ]
// 返回 false。图形顶端留有间隔，无法覆盖成一个矩形。
//
// 示例 4:
// rectangles = [
//  [1,1,3,3],
//  [3,1,4,2],
//  [1,3,2,4],
//  [2,2,4,4]
// ]
// 返回 false。因为中间有相交区域，虽然形成了矩形，但不是精确覆盖。
func isRectangleCover(rectangles [][]int) bool {
	// 1,总面积等于小块的面积和
	// 2,除大矩形4个顶点外其它点都出现偶数(2或4)次
	left, right := math.MaxInt32, 0
	bottom, top := math.MaxInt32, 0
	area := 0
	countMap := make(map[string]int)

	for _, rectangle := range rectangles {
		left = min(left, rectangle[0])
		bottom = min(bottom, rectangle[1])
		right = max(right, rectangle[2])
		top = max(top, rectangle[3])
		area += (rectangle[3] - rectangle[1]) * (rectangle[2] - rectangle[0])
		leftTop, leftBottom := fmt.Sprintf("%d-%d", rectangle[0], rectangle[3]), fmt.Sprintf("%d-%d", rectangle[0], rectangle[1])
		rightTop, rightBottom := fmt.Sprintf("%d-%d", rectangle[2], rectangle[3]), fmt.Sprintf("%d-%d", rectangle[2], rectangle[1])
		countMap[leftTop]++
		countMap[leftBottom]++
		countMap[rightTop]++
		countMap[rightBottom]++
	}
	if area != (right-left)*(top-bottom) {
		return false
	}
	leftTop, leftBottom := fmt.Sprintf("%d-%d", left, top), fmt.Sprintf("%d-%d", left, bottom)
	rightTop, rightBottom := fmt.Sprintf("%d-%d", right, top), fmt.Sprintf("%d-%d", right, bottom)
	countMap[leftTop]++
	countMap[leftBottom]++
	countMap[rightTop]++
	countMap[rightBottom]++
	for _, v := range countMap {
		if v&1 == 1 {
			return false
		}
	}
	return true
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
