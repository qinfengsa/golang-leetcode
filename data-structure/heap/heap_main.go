package heap

import (
	"container/heap"
	"sort"
)

type hp [][]int

func (h hp) Len() int {
	return len(h)
}

func (h hp) Less(i, j int) bool {
	return h[i][0]+h[i][1] > h[j][0]+h[j][1]
}

func (h hp) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *hp) Push(x interface{}) {
	*h = append(*h, x.([]int))
}

func (h *hp) Pop() interface{} {
	tmp := *h
	v := tmp[len(tmp)-1]
	*h = tmp[:len(tmp)-1]
	return v
}

// 373. 查找和最小的K对数字
// 给定两个以升序排列的整数数组 nums1 和 nums2 , 以及一个整数 k 。
//
// 定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2 。
//
// 请找到和最小的 k 个数对 (u1,v1),  (u2,v2)  ...  (uk,vk) 。
//
// 示例 1:
// 输入: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
// 输出: [1,2],[1,4],[1,6]
// 解释: 返回序列中的前 3 对数：
//     [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
//
// 示例 2:
// 输入: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
// 输出: [1,1],[1,1]
// 解释: 返回序列中的前 2 对数：
//     [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
//
// 示例 3:
// 输入: nums1 = [1,2], nums2 = [3], k = 3
// 输出: [1,3],[2,3]
// 解释: 也可能序列中所有的数对都被返回:[1,3],[2,3]
//
// 提示:
// 1 <= nums1.length, nums2.length <= 104
// -109 <= nums1[i], nums2[i] <= 109
// nums1, nums2 均为升序排列
// 1 <= k <= 1000
func kSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
	h := hp{}
	for i := 0; i < min(k, len(nums1)); i++ {
		for j := 0; j < min(k, len(nums2)); j++ {
			//nums := []int{nums1[i], nums2[j]}
			//sum := nums1[i] + nums2[j]
			heap.Push(&h, []int{nums1[i], nums2[j]})
			if h.Len() > k {
				heap.Pop(&h)
			}
		}
	}
	result := make([][]int, 0)
	for i := 0; i < min(k, h.Len()); i++ {
		result = append(result, h[i])
	}
	return result
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

// 502. IPO
// 假设 力扣（LeetCode）即将开始 IPO 。为了以更高的价格将股票卖给风险投资公司，力扣 希望在 IPO 之前开展一些项目以增加其资本。 由于资源有限，它只能在 IPO 之前完成最多 k 个不同的项目。帮助 力扣 设计完成最多 k 个不同项目后得到最大总资本的方式。
//
// 给你 n 个项目。对于每个项目 i ，它都有一个纯利润 profits[i] ，和启动该项目需要的最小资本 capital[i] 。
//
// 最初，你的资本为 w 。当你完成一个项目时，你将获得纯利润，且利润将被添加到你的总资本中。
// 总而言之，从给定项目中选择 最多 k 个不同项目的列表，以 最大化最终资本 ，并输出最终可获得的最多资本。
// 答案保证在 32 位有符号整数范围内。
//
// 示例 1：
// 输入：k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
// 输出：4
// 解释：
// 由于你的初始资本为 0，你仅可以从 0 号项目开始。
// 在完成后，你将获得 1 的利润，你的总资本将变为 1。
// 此时你可以选择开始 1 号或 2 号项目。
// 由于你最多可以选择两个项目，所以你需要完成 2 号项目以获得最大的资本。
// 因此，输出最后最大化的资本，为 0 + 1 + 3 = 4。
//
// 示例 2：
// 输入：k = 3, w = 0, profits = [1,2,3], capital = [0,1,2]
// 输出：6
//
// 提示：
// 1 <= k <= 105
// 0 <= w <= 109
// n == profits.length
// n == capital.length
// 1 <= n <= 105
// 0 <= profits[i] <= 104
// 0 <= capital[i] <= 109
func findMaximizedCapital(k int, w int, profits []int, capital []int) int {
	n := len(profits)
	ipoList := make([]*IPO, n)
	for i := 0; i < n; i++ {
		ipoList[i] = &IPO{profit: profits[i], cap: capital[i]}
	}
	sort.Slice(ipoList, func(i, j int) bool {
		return ipoList[i].cap < ipoList[j].cap
	})
	index := 0
	hp := ipoHp{}
	for i := 0; i < k; i++ {
		for index < n && ipoList[index].cap <= w {
			heap.Push(&hp, ipoList[index])
			index++
		}
		if hp.Len() == 0 {
			break
		}
		ipo := heap.Pop(&hp)
		w += ipo.(*IPO).profit
	}
	return w
}

type IPO struct {
	profit int
	cap    int
}

type ipoHp []*IPO

func (h ipoHp) Len() int {
	return len(h)
}

func (h ipoHp) Less(i, j int) bool {
	return h[i].profit > h[j].profit
}

func (h ipoHp) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *ipoHp) Push(x interface{}) {
	*h = append(*h, x.(*IPO))
}

func (h *ipoHp) Pop() interface{} {
	tmp := *h
	v := tmp[len(tmp)-1]
	*h = tmp[:len(tmp)-1]
	return v
}
