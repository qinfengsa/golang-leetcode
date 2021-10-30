package interval

import "sort"

// 435. 无重叠区间
// 给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
//
// 注意:
// 可以认为区间的终点总是大于它的起点。
// 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
//
// 示例 1:
// 输入: [ [1,2], [2,3], [3,4], [1,3] ]
// 输出: 1
// 解释: 移除 [1,3] 后，剩下的区间没有重叠。
//
// 示例 2:
// 输入: [ [1,2], [1,2], [1,2] ]
// 输出: 2
// 解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
//
// 示例 3:
// 输入: [ [1,2], [2,3] ]
// 输出: 0
// 解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
func eraseOverlapIntervals(intervals [][]int) int {

	result := 0
	// 排序
	sort.Slice(intervals, func(i, j int) bool {
		a, b := intervals[i], intervals[j]
		if a[1] == b[1] {
			return a[0] < b[0]
		}
		return a[1] < b[1]
	})
	lastIndex := 0

	for i := 1; i < len(intervals); i++ {
		last, cur := intervals[lastIndex], intervals[i]
		// 符合条件 区间不重合
		if cur[0] >= last[1] {
			lastIndex = i
		} else {
			result++
		}
	}

	return result
}

// 436. 寻找右区间
// 给你一个区间数组 intervals ，其中 intervals[i] = [starti, endi] ，且每个 starti 都 不同 。
//
// 区间 i 的 右侧区间 可以记作区间 j ，并满足 startj >= endi ，且 startj 最小化 。
// 返回一个由每个区间 i 的 右侧区间 的最小起始位置组成的数组。如果某个区间 i 不存在对应的 右侧区间 ，则下标 i 处的值设为 -1 。
//
// 示例 1：
// 输入：intervals = [[1,2]]
// 输出：[-1]
// 解释：集合中只有一个区间，所以输出-1。
//
// 示例 2：
// 输入：intervals = [[3,4],[2,3],[1,2]]
// 输出：[-1, 0, 1]
// 解释：对于 [3,4] ，没有满足条件的“右侧”区间。
// 对于 [2,3] ，区间[3,4]具有最小的“右”起点;
// 对于 [1,2] ，区间[2,3]具有最小的“右”起点。
//
// 示例 3：
// 输入：intervals = [[1,4],[2,3],[3,4]]
// 输出：[-1, 2, -1]
// 解释：对于区间 [1,4] 和 [3,4] ，没有满足条件的“右侧”区间。
// 对于 [2,3] ，区间 [3,4] 有最小的“右”起点。
//
// 提示：
// 1 <= intervals.length <= 2 * 104
// intervals[i].length == 2
// -106 <= starti <= endi <= 106
// 每个间隔的起点都 不相同
func findRightInterval(intervals [][]int) []int {
	n := len(intervals)
	result := make([]int, n)
	indexMap, nums := make(map[int]int), make([]int, n)
	for i, interval := range intervals {
		indexMap[interval[0]] = i
		nums[i] = interval[0]
	}
	sort.Ints(nums)
	var getLeft func(target int) int

	getLeft = func(target int) int {
		left, right := 0, n-1
		if target > nums[right] {
			return -1
		}
		if target <= nums[left] {
			return 0
		}

		for left < right {
			mid := (left + right) >> 1
			if nums[mid] == target {
				return mid
			}
			if nums[mid] < target {
				left = mid + 1
			} else {
				right = mid
			}
		}

		return left
	}

	for i, interval := range intervals {
		right := interval[1]
		index := getLeft(right)
		if index < 0 {
			result[i] = -1
		} else {
			result[i] = indexMap[nums[index]]
		}
	}

	return result
}
