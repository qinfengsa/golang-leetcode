package heap

import (
	"container/heap"
	"math"
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
//
//	[1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
//
// 示例 2:
// 输入: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
// 输出: [1,1],[1,1]
// 解释: 返回序列中的前 2 对数：
//
//	[1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
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

// 630. 课程表 III
// 这里有 n 门不同的在线课程，按从 1 到 n 编号。给你一个数组 courses ，其中 courses[i] = [durationi, lastDayi]
// 表示第 i 门课将会 持续 上 durationi 天课，并且必须在不晚于 lastDayi 的时候完成。
// 你的学期从第 1 天开始。且不能同时修读两门及两门以上的课程。
// 返回你最多可以修读的课程数目。
//
// 示例 1：
// 输入：courses = [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
// 输出：3
// 解释：
// 这里一共有 4 门课程，但是你最多可以修 3 门：
// 首先，修第 1 门课，耗费 100 天，在第 100 天完成，在第 101 天开始下门课。
// 第二，修第 3 门课，耗费 1000 天，在第 1100 天完成，在第 1101 天开始下门课程。
// 第三，修第 2 门课，耗时 200 天，在第 1300 天完成。
// 第 4 门课现在不能修，因为将会在第 3300 天完成它，这已经超出了关闭日期。
//
// 示例 2：
// 输入：courses = [[1,2]] 输出：1
//
// 示例 3：
// 输入：courses = [[3,2],[4,3]] 输出：0
//
// 提示:
// 1 <= courses.length <= 104
// 1 <= durationi, lastDayi <= 104
func scheduleCourse(courses [][]int) int {
	sort.Slice(courses, func(i, j int) bool {
		return courses[i][1] < courses[j][1]
	})
	time := 0
	h := pq{}
	for _, course := range courses {
		if time+course[0] <= course[1] {
			heap.Push(&h, course[0])
			time += course[0]
		} else if h.Len() > 0 && course[0] < h[0] {
			time += course[0] - heap.Pop(&h).(int)
			heap.Push(&h, course[0])
		}
	}
	return h.Len()
}

type pq []int

func (h pq) Len() int {
	return len(h)
}

func (h pq) Less(i, j int) bool {
	return h[i] > h[j]
}

func (h pq) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *pq) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *pq) Pop() interface{} {
	tmp := *h
	v := tmp[len(tmp)-1]
	*h = tmp[:len(tmp)-1]
	return v
}

// 2558. 从数量最多的堆取走礼物
// 给你一个整数数组 gifts ，表示各堆礼物的数量。每一秒，你需要执行以下操作：
//
// 选择礼物数量最多的那一堆。
// 如果不止一堆都符合礼物数量最多，从中选择任一堆即可。
// 选中的那一堆留下平方根数量的礼物（向下取整），取走其他的礼物。
// 返回在 k 秒后剩下的礼物数量。
//
// 示例 1：
// 输入：gifts = [25,64,9,4,100], k = 4
// 输出：29
// 解释：
// 按下述方式取走礼物：
// - 在第一秒，选中最后一堆，剩下 10 个礼物。
// - 接着第二秒选中第二堆礼物，剩下 8 个礼物。
// - 然后选中第一堆礼物，剩下 5 个礼物。
// - 最后，再次选中最后一堆礼物，剩下 3 个礼物。
// 最后剩下的礼物数量分别是 [5,8,9,4,3] ，所以，剩下礼物的总数量是 29 。
//
// 示例 2：
// 输入：gifts = [1,1,1,1], k = 4
// 输出：4
// 解释：
// 在本例中，不管选中哪一堆礼物，都必须剩下 1 个礼物。
// 也就是说，你无法获取任一堆中的礼物。
// 所以，剩下礼物的总数量是 4 。
//
// 提示：
// 1 <= gifts.length <= 103
// 1 <= gifts[i] <= 109
// 1 <= k <= 103
func pickGifts(gifts []int, k int) int64 {
	h := &pq{}
	for _, gift := range gifts {
		h.Push(gift)
	}
	heap.Init(h)
	for k > 0 {
		num := heap.Pop(h).(int)
		tmp := int(math.Floor(math.Sqrt(float64(num))))
		heap.Push(h, tmp)
		k--
	}
	var result int64 = 0
	for h.Len() > 0 {
		result += int64(h.Pop().(int))
	}
	return result
}
