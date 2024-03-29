package search

import (
	"fmt"
	"math"
	"sort"
)

// 二分查找

// 367. 有效的完全平方数
// 给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。
//
// 说明：不要使用任何内置的库函数，如  sqrt。
//
// 示例 1：
//
// 输入：16 输出：True
// 示例 2：
//
// 输入：14 输出：False
func isPerfectSquare(num int) bool {
	if num < 0 {
		return false
	}
	if num <= 1 {
		return true
	}
	low, high := 1, num
	for low < high {
		mid := (low + high) >> 1
		tmp := mid * mid
		if tmp == num {
			return true
		}
		if tmp < num {
			low = mid + 1
		} else {
			high = mid
		}

	}

	return low*low == num
}

// 704. 二分查找
// 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
//
// 示例 1:
// 输入: nums = [-1,0,3,5,9,12], target = 9 输出: 4
// 解释: 9 出现在 nums 中并且下标为 4
//
// 示例 2:
// 输入: nums = [-1,0,3,5,9,12], target = 2 输出: -1
// 解释: 2 不存在 nums 中因此返回 -1
//
// 提示：
// 你可以假设 nums 中的所有元素是不重复的。
// n 将在 [1, 10000]之间。
// nums 的每个元素都将在 [-9999, 9999]之间。
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	if nums[left] == target {
		return left
	}
	if nums[right] == target {
		return right
	}
	if target < nums[left] || target > nums[right] {
		return -1
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
	return -1
}

// 81. 搜索旋转排序数组 II
// 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
//
// 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
//
// 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
//
// 示例 1：
//
// 输入：nums = [2,5,6,0,0,1,2], target = 0 输出：true
//
// 示例 2：
// 输入：nums = [2,5,6,0,0,1,2], target = 3 输出：false
//
// 提示：
// 1 <= nums.length <= 5000
// -104 <= nums[i] <= 104
// 题目数据保证 nums 在预先未知的某个下标上进行了旋转
// -104 <= target <= 104
//
// 进阶：
//
// 这是 搜索旋转排序数组 的延伸题目，本题中的 nums  可能包含重复元素。
// 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？
func search2(nums []int, target int) bool {
	size := len(nums)
	if size == 0 {
		return false
	}
	if size == 1 {
		return nums[0] == target
	}
	left, right := 0, size-1
	// 思路 ： 二分法找到峰值, 分成两个数组 分别二分
	for left < right {
		mid := (left + right) >> 1
		if nums[mid] == target {
			return true
		}

		if nums[mid] < nums[left] {
			right = mid
		} else if nums[mid] > nums[right] {
			left = mid + 1
		} else if nums[mid] == nums[left] {
			left++
		} else {
			right--
		}
	}
	fmt.Println("left -> ", left)
	return binSearch(nums, target, 0, left) || binSearch(nums, target, left+1, size-1)
}

func binSearch(nums []int, target int, start int, end int) bool {
	if start == end {
		return nums[start] == target
	}
	if nums[end] == target {
		return true
	}
	for start < end {
		mid := (start + end) >> 1
		if nums[mid] == target {
			return true
		} else if nums[mid] < target {
			start = mid + 1
		} else {
			end = mid
		}
	}
	return false
}

// 154. 寻找旋转排序数组中的最小值 II
// 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
// 若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
// 若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
// 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
//
// 给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
//
// 示例 1：
// 输入：nums = [1,3,5] 输出：1
//
// 示例 2：
// 输入：nums = [2,2,2,0,1] 输出：0
//
// 提示：
// n == nums.length
// 1 <= n <= 5000
// -5000 <= nums[i] <= 5000
// nums 原来是一个升序排序的数组，并进行了 1 至 n 次旋转
//
// 进阶：
// 这道题是 寻找旋转排序数组中的最小值 的延伸题目。
// 允许重复会影响算法的时间复杂度吗？会如何影响，为什么？
func findMinII(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	left, right := 0, n-1
	// 思路 ： 二分法找到峰值, 分成两个数组 分别二分
	for left < right {
		if nums[left] == nums[right] {
			left++
			continue
		}
		mid := (left + right) >> 1
		// 如果中间值比右边小,顺序排列的
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return nums[left]
}

// 33. 搜索旋转排序数组
// 整数数组 nums 按升序排列，数组中的值 互不相同 。
// 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
// 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
//
// 示例 1：
// 输入：nums = [4,5,6,7,0,1,2], target = 0 输出：4
//
// 示例 2：
// 输入：nums = [4,5,6,7,0,1,2], target = 3 输出：-1
//
// 示例 3：
// 输入：nums = [1], target = 0 输出：-1
//
// 提示：
// 1 <= nums.length <= 5000
// -10^4 <= nums[i] <= 10^4
// nums 中的每个值都 独一无二
// 题目数据保证 nums 在预先未知的某个下标上进行了旋转
// -10^4 <= target <= 10^4
//
// 进阶：你可以设计一个时间复杂度为 O(log n) 的解决方案吗？
func searchSortedNums(nums []int, target int) int {
	// 二分
	left, right := 0, len(nums)-1
	if nums[right] == target {
		return right
	}
	if nums[left] == target {
		return left
	}

	for left < right {
		mid := (left + right) >> 1
		if nums[mid] == target {
			return mid
		}
		// mid 比 右边小
		if nums[mid] < nums[right] {
			if nums[mid] < target && nums[right] >= target {
				left = mid + 1
			} else {
				right = mid - 1
			}
		} else {
			// mid 比右边大
			if nums[mid] > target && nums[left] <= target {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}

	}

	return -1
}

// 34. 在排序数组中查找元素的第一个和最后一个位置
//
// 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
// 如果数组中不存在目标值 target，返回 [-1, -1]。
//
// 进阶：
// 你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
//
// 示例 1：
// 输入：nums = [5,7,7,8,8,10], target = 8 输出：[3,4]
//
// 示例 2：
// 输入：nums = [5,7,7,8,8,10], target = 6 输出：[-1,-1]
//
// 示例 3：
// 输入：nums = [], target = 0 输出：[-1,-1]
//
// 提示：
// 0 <= nums.length <= 105
// -109 <= nums[i] <= 109
// nums 是一个非递减数组
// -109 <= target <= 109
func searchRange(nums []int, target int) []int {
	left, right := 0, len(nums)-1
	result := []int{-1, -1}
	if right < 0 {
		return result
	}
	if nums[left] > target || nums[right] < target {
		return result
	}

	for left <= right {
		if nums[left] == target && nums[right] == target {
			result[0] = left
			result[1] = right
			return result
		}
		mid := (left + right) >> 1
		if nums[mid] == target {
			for nums[left] < target {
				left++
			}
			for nums[right] > target {
				right--
			}

		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}

	}

	return result
}

// 35. 搜索插入位置
// 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
//
// 你可以假设数组中无重复元素。
//
// 示例 1: 输入: [1,3,5,6], 5 输出: 2
// 示例 2: 输入: [1,3,5,6], 2 输出: 1
// 示例 3: 输入: [1,3,5,6], 7 输出: 4
// 示例 4: 输入: [1,3,5,6], 0 输出: 0
func searchInsert(nums []int, target int) int {

	left, right := 0, len(nums)-1

	if target > nums[right] {
		return right + 1
	}
	if target == nums[right] {
		return right
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

// 74. 搜索二维矩阵
// 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
//
// 每行中的整数从左到右按升序排列。
// 每行的第一个整数大于前一行的最后一个整数。
//
// 示例 1：
// 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
// 输出：true
//
// 示例 2：
// 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
// 输出：false
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 100
// -104 <= matrix[i][j], target <= 104
func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	// 从右上角 开始
	i, j := 0, n-1

	for i < m && j >= 0 {
		num := matrix[i][j]
		if num == target {
			return true
		} else if num < target {
			i++
		} else {
			j--
		}
	}

	return false
}

// 81. 搜索旋转排序数组 II
// 已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
//
// 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
// 给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。
//
// 示例 1：
// 输入：nums = [2,5,6,0,0,1,2], target = 0 输出：true
//
// 示例 2：
// 输入：nums = [2,5,6,0,0,1,2], target = 3 输出：false
//
// 提示：
// 1 <= nums.length <= 5000
// -104 <= nums[i] <= 104
// 题目数据保证 nums 在预先未知的某个下标上进行了旋转
// -104 <= target <= 104
//
// 进阶：
// 这是 搜索旋转排序数组 的延伸题目，本题中的 nums  可能包含重复元素。
// 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？
func searchII(nums []int, target int) bool {
	// 二分
	left, right := 0, len(nums)-1
	if nums[right] == target || nums[left] == target {
		return true
	}

	// 思路 ： 二分法找到峰值, 分成两个数组 分别二分
	for left < right {
		mid := (left + right) >> 1
		if nums[mid] == target {
			return true
		}
		if nums[mid] < nums[left] {
			right = mid
		} else if nums[mid] > nums[right] {
			left = mid + 1
		} else if nums[mid] == nums[left] {
			left++
		} else {
			right--
		}
	}
	fmt.Println("left -> ", left)

	findTarget := func(start, end int) bool {
		if start == end {
			return nums[start] == target
		}
		for start < end {
			mid := (start + end) >> 1
			if nums[mid] == target {
				return true
			} else if nums[mid] < target {
				start = mid + 1
			} else {
				end = mid
			}
		}
		return false
	}

	return findTarget(0, left) || findTarget(left+1, len(nums)-1)
}

// 153. 寻找旋转排序数组中的最小值
// 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
// 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
// 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
// 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
//
// 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
//
// 示例 1：
// 输入：nums = [3,4,5,1,2] 输出：1
// 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
//
// 示例 2：
// 输入：nums = [4,5,6,7,0,1,2] 输出：0
// 解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。
//
// 示例 3：
// 输入：nums = [11,13,15,17] 输出：11
// 解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
//
// 提示：
// n == nums.length
// 1 <= n <= 5000
// -5000 <= nums[i] <= 5000
// nums 中的所有整数 互不相同
// nums 原来是一个升序排序的数组，并进行了 1 至 n 次旋转
func findMin(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	left, right := 0, n-1
	// 思路 ： 二分法找到峰值, 分成两个数组 分别二分
	for left < right {
		if nums[left] == nums[right] {
			left++
			continue
		}
		mid := (left + right) >> 1
		// 如果中间值比右边小,顺序排列的
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return nums[left]
}

// 162. 寻找峰值
// 峰值元素是指其值严格大于左右相邻值的元素。
//
// 给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
// 你可以假设 nums[-1] = nums[n] = -∞ 。
// 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
//
// 示例 1：
// 输入：nums = [1,2,3,1]
// 输出：2
// 解释：3 是峰值元素，你的函数应该返回其索引 2。
//
// 示例 2：
// 输入：nums = [1,2,1,3,5,6,4]
// 输出：1 或 5
// 解释：你的函数可以返回索引 1，其峰值元素为 2；
//
//	或者返回索引 5， 其峰值元素为 6。
//
// 提示：
// 1 <= nums.length <= 1000
// -231 <= nums[i] <= 231 - 1
// 对于所有有效的 i 都有 nums[i] != nums[i + 1]
func findPeakElement(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	left, right := 0, len(nums)-1
	// 规律一：如果nums[i] > nums[i+1]，则在i之前一定存在峰值元素
	// 规律二：如果nums[i] < nums[i+1]，则在i+1之后一定存在峰值元素
	for left < right {
		mid := (left + right) >> 1
		if nums[mid] > nums[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}

// 240. 搜索二维矩阵 II
// 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
//
// 每行的元素从左到右升序排列。
// 每列的元素从上到下升序排列。
//
// 示例 1：
// 输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
// 输出：true
//
// 示例 2：
// 输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
// 输出：false
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= n, m <= 300
// -109 <= matrix[i][j] <= 109
// 每行的所有元素从左到右升序排列
// 每列的所有元素从上到下升序排列
// -109 <= target <= 109
func searchMatrixII(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	i, j := 0, n-1
	for i >= 0 && i < m && j >= 0 && j < n {
		num := matrix[i][j]
		if num == target {
			return true
		} else if num > target {
			j--
		} else {
			i++
		}

	}

	return false
}

// 278. 第一个错误的版本
// 你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。
//
// 假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
// 你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。
//
// 示例 1：
// 输入：n = 5, bad = 4
// 输出：4
// 解释：
// 调用 isBadVersion(3) -> false
// 调用 isBadVersion(5) -> true
// 调用 isBadVersion(4) -> true
// 所以，4 是第一个错误的版本。
//
// 示例 2：
// 输入：n = 1, bad = 1
// 输出：1
//
// 提示：
// 1 <= bad <= n <= 231 - 1
func firstBadVersion(n int) int {
	var isBadVersion func(version int) bool
	left, right := 1, n
	for left < right {
		mid := (left + right) >> 1
		isBad := isBadVersion(mid)
		if isBad && !isBadVersion(mid-1) {
			return mid
		}
		if isBad {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}

// 274. H 指数
// 给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数。计算并返回该研究者的 h 指数。
//
// h 指数的定义：h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （n 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。且其余的 n - h 篇论文每篇被引用次数 不超过 h 次。
//
// 例如：某人的 h 指数是 20，这表示他已发表的论文中，每篇被引用了至少 20 次的论文总共有 20 篇。
// 提示：如果 h 有多种可能的值，h 指数 是其中最大的那个。
//
// 示例 1：
// 输入：citations = [3,0,6,1,5]
// 输出：3
// 解释：给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 3, 0, 6, 1, 5 次。
//
//	由于研究者有 3 篇论文每篇 至少 被引用了 3 次，其余两篇论文每篇被引用 不多于 3 次，所以她的 h 指数是 3。
//
// 示例 2：
// 输入：citations = [1,3,1]
// 输出：1
//
// 提示：
// n == citations.length
// 1 <= n <= 5000
// 0 <= citations[i] <= 1000
//
// 275. H 指数 II
// 给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数，citations 已经按照 升序排列 。计算并返回该研究者的 h 指数。
//
// h 指数的定义：h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （n 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。且其余的 n - h 篇论文每篇被引用次数 不超过 h 次。
// 提示：如果 h 有多种可能的值，h 指数 是其中最大的那个。
// 请你设计并实现对数时间复杂度的算法解决此问题。
//
// 示例 1：
// 输入：citations = [0,1,3,5,6]
// 输出：3
// 解释：给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 0, 1, 3, 5, 6 次。
//
//	由于研究者有 3 篇论文每篇 至少 被引用了 3 次，其余两篇论文每篇被引用 不多于 3 次，所以她的 h 指数是 3 。
//
// 示例 2：
// 输入：citations = [1,2,100] 输出：2
//
// 提示：
// n == citations.length
// 1 <= n <= 105
// 0 <= citations[i] <= 1000
// citations 按 升序排列
func hIndex(citations []int) int {
	sort.Ints(citations)
	n := len(citations)
	left, right := 0, n-1
	for left < right {
		mid := (left + right) >> 1
		if citations[mid] >= n-mid {
			right = mid
		} else {
			left = mid + 1
		}
	}
	if left < n && citations[left] == 0 {
		return citations[left]
	}

	return n - left
}

// 287. 寻找重复数
// 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。
//
// 假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。
// 你设计的解决方案必须不修改数组 nums 且只用常量级 O(1) 的额外空间。
//
// 示例 1：
// 输入：nums = [1,3,4,2,2] 输出：2
//
// 示例 2：
// 输入：nums = [3,1,3,4,2] 输出：3
//
// 示例 3：
// 输入：nums = [1,1] 输出：1
//
// 示例 4：
// 输入：nums = [1,1,2] 输出：1
//
// 提示：
// 1 <= n <= 105
// nums.length == n + 1
// 1 <= nums[i] <= n
// nums 中 只有一个整数 出现 两次或多次 ，其余整数均只出现 一次
//
// 进阶：
// 如何证明 nums 中至少存在一个重复的数字?
// 你可以设计一个线性级时间复杂度 O(n) 的解决方案吗？
func findDuplicate(nums []int) int {
	// 【笔记】这道题（据说）花费了计算机科学界的传奇人物Don Knuth 24小时才解出来。并且我只见过一个人（注：Keith Amling）用更短时间解出此题。

	// 快慢指针，一个时间复杂度为O(N)的算法。

	// 其一，对于链表问题，使用快慢指针可以判断是否有环。

	// 其二，本题可以使用数组配合下标，抽象成链表问题。但是难点是要定位环的入口位置。

	// 举个例子：nums = [2,5, 9 ,6,9,3,8, 9 ,7,1]，构造成链表就是：2->[9]->1->5->3->6->8->7->[9]，也就是在[9]处循环。

	// 其三，快慢指针问题，会在环内的[9]->1->5->3->6->8->7->[9]任何一个节点追上，不一定是在[9]处相碰，事实上会在7处碰上。

	// 其四，必须另起一个for循环定位环入口位置[9]。这里需要数学证明。

	// 快慢指针思想，把数组看成链表 数组的下标，数组的值代表next的指针
	// 题目中  n + 1 个数 ，数字都在 1 到 n 之间 就是为了这个
	// 举个例子：nums = [2,5, 9 ,6,9,3,8, 9 ,7,1]，构造成链表就是：2->[9]->1->5->3->6->8->7->[9]，
	// 也就是在[9]处循环。
	fast, slow := 0, 0
	for true {

		fast = nums[nums[fast]] // fast -> nums[fast] -> nums[nums[fast]]
		slow = nums[slow]       // slow -> nums[slow]
		if fast == slow {       // fast到slow 相撞，只能证明有重复
			// 找相遇的位置
			fast = 0
			for nums[fast] != nums[slow] {
				fast = nums[fast]
				slow = nums[slow]
			}
			return nums[slow]
		}
	}
	return -1
}

// 剑指 Offer II 069. 山峰数组的顶部
// 符合下列属性的数组 arr 称为 山峰数组（山脉数组） ：
//
// arr.length >= 3
// 存在 i（0 < i < arr.length - 1）使得：
// arr[0] < arr[1] < ... arr[i-1] < arr[i]
// arr[i] > arr[i+1] > ... > arr[arr.length - 1]
// 给定由整数组成的山峰数组 arr ，返回任何满足 arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1] 的下标 i ，即山峰顶部。
//
// 示例 1：
// 输入：arr = [0,1,0] 输出：1
//
// 示例 2：
// 输入：arr = [1,3,5,4,2] 输出：2
//
// 示例 3：
// 输入：arr = [0,10,5,2] 输出：1
//
// 示例 4：
// 输入：arr = [3,4,5,1] 输出：2
//
// 示例 5：
// 输入：arr = [24,69,100,99,79,78,67,36,26,19] 输出：2
//
// 提示：
// 3 <= arr.length <= 104
// 0 <= arr[i] <= 106
// 题目数据保证 arr 是一个山脉数组
//
// 进阶：很容易想到时间复杂度 O(n) 的解决方案，你可以设计一个 O(log(n)) 的解决方案吗？
//
// 注意：本题与主站 852 题相同：https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/
func peakIndexInMountainArray(arr []int) int {
	n := len(arr)
	left, right := 0, n-1
	for left < right {
		mid := (left + right) >> 1
		if mid > 0 && arr[mid] > arr[mid-1] && arr[mid] > arr[mid+1] {
			return mid
		}
		if arr[mid] < arr[mid-1] {
			right = mid
		} else {
			left = mid + 1
		}

	}
	return left
}

// 374. 猜数字大小
// 猜数字游戏的规则如下：
//
// 每轮游戏，我都会从 1 到 n 随机选择一个数字。 请你猜选出的是哪个数字。
// 如果你猜错了，我会告诉你，你猜测的数字比我选出的数字是大了还是小了。
// 你可以通过调用一个预先定义好的接口 int guess(int num) 来获取猜测结果，返回值一共有 3 种可能的情况（-1，1 或 0）：
//
// -1：我选出的数字比你猜的数字小 pick < num
// 1：我选出的数字比你猜的数字大 pick > num
// 0：我选出的数字和你猜的数字一样。恭喜！你猜对了！pick == num
// 返回我选出的数字。
//
// 示例 1：
// 输入：n = 10, pick = 6  输出：6
//
// 示例 2：
// 输入：n = 1, pick = 1  输出：1
//
// 示例 3：
// 输入：n = 2, pick = 1 输出：1
//
// 示例 4：
// 输入：n = 2, pick = 2 输出：2
//
// 提示：
// 1 <= n <= 231 - 1
// 1 <= pick <= n
func guessNumber(n int) int {
	var guess func(num int) int

	low, high := 1, n
	for low < high {
		mid := (low + high) >> 1
		result := guess(mid)
		if result == 0 {
			return mid
		} else if result == -1 {
			high = mid
		} else {
			low = mid + 1
		}
	}
	return low
}

// 480. 滑动窗口中位数
// 中位数是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。
//
// 例如：
// [2,3,4]，中位数是 3
// [2,3]，中位数是 (2 + 3) / 2 = 2.5
// 给你一个数组 nums，有一个长度为 k 的窗口从最左端滑动到最右端。窗口中有 k 个数，每次窗口向右移动 1 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。
//
// 示例：
// 给出 nums = [1,3,-1,-3,5,3,6,7]，以及 k = 3。
// 窗口位置                      中位数
// ---------------               -----
// [1  3  -1] -3  5  3  6  7       1
//
//	1 [3  -1  -3] 5  3  6  7      -1
//	1  3 [-1  -3  5] 3  6  7      -1
//	1  3  -1 [-3  5  3] 6  7       3
//	1  3  -1  -3 [5  3  6] 7       5
//	1  3  -1  -3  5 [3  6  7]      6
//	因此，返回该滑动窗口的中位数数组 [1,-1,-1,3,5,6]。
//
// 提示：
// 你可以假设 k 始终有效，即：k 始终小于等于输入的非空数组的元素个数。
// 与真实值误差在 10 ^ -5 以内的答案将被视作正确答案。
func medianSlidingWindow(nums []int, k int) []float64 {
	// 最大堆(存放 较小的元素) 最小堆(存放 较大的元素)

	n := len(nums)

	knums := make([]int, k)
	copy(knums, nums[:k])
	sort.Ints(knums)
	result := make([]float64, n-k+1)
	midIdx := k >> 1
	getMid := func() float64 {
		// 奇数
		if k&1 == 1 {
			return float64(knums[midIdx])
		}
		sum := knums[midIdx] + knums[midIdx-1]
		return float64(sum) / 2.0
	}
	result[0] = getMid()
	for i := k; i < n; i++ {
		delIdx := getIndex(knums, nums[i-k])
		knums = append(knums[:delIdx], knums[delIdx+1:]...)
		addIdx := getIndex(knums, nums[i])
		fmt.Println(1)
		fmt.Println(knums)
		fmt.Printf("addIdx:%d", addIdx)
		fmt.Println()
		fmt.Println(2)
		last := append([]int{nums[i]}, knums[addIdx:]...)
		knums = append(knums[:addIdx], last...)
		fmt.Println(knums)
		result[i-k+1] = getMid()
	}

	return result
}

func getIndex(nums []int, target int) int {
	left, right := 0, len(nums)-1
	// 二分查找
	for left <= right {
		mid := (left + right) >> 1
		if target == nums[mid] {
			return mid
		}
		if nums[mid] > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return left
}

// 540. 有序数组中的单一元素
// 给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。
//
// 示例 1:
// 输入: nums = [1,1,2,3,3,4,4,8,8] 输出: 2
//
// 示例 2:
// 输入: nums =  [3,3,7,7,10,11,11] 输出: 10
//
// 提示:
// 1 <= nums.length <= 105
// 0 <= nums[i] <= 105
//
// 进阶: 采用的方案可以在 O(log n) 时间复杂度和 O(1) 空间复杂度中运行吗？
func singleNonDuplicate(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	if nums[0] != nums[1] {
		return nums[0]
	}
	if nums[n-1] != nums[n-2] {
		return nums[n-1]
	}
	left, right := 0, n-1
	for left < right {
		mid := (left + right) >> 1
		if mid > 0 && nums[mid] != nums[mid-1] && nums[mid] != nums[mid+1] {
			return nums[mid]
		}
		// mid 是 奇数
		if mid&1 == 1 {
			// 判断 mid 和 mid - 1 是否相等
			if nums[mid] == nums[mid-1] {
				// target 在 后面
				left = mid + 1
			} else {
				right = mid
			}
		} else {
			// 偶数
			if nums[mid] == nums[mid-1] {
				// target 在 前面
				right = mid
			} else {
				left = mid + 1
			}
		}
	}

	return nums[left]
}

// 658. 找到 K 个最接近的元素
// 给定一个排序好的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）的 k 个数。返回的结果必须要是按升序排好的。
//
// 整数 a 比整数 b 更接近 x 需要满足：
//
// |a - x| < |b - x| 或者
// |a - x| == |b - x| 且 a < b
//
// 示例 1：
// 输入：arr = [1,2,3,4,5], k = 4, x = 3
// 输出：[1,2,3,4]
//
// 示例 2：
// 输入：arr = [1,2,3,4,5], k = 4, x = -1
// 输出：[1,2,3,4]
//
// 提示：
// 1 <= k <= arr.length
// 1 <= arr.length <= 104
// 数组里的每个元素与 x 的绝对值不超过 104
func findClosestElements(arr []int, k int, x int) []int {
	result := make([]int, k)
	n := len(arr)
	left, right := 0, n-k

	for left < right {
		mid := (left + right) >> 1

		// arr[mid], x , arr[mid+k]
		// 要保证 x 在mid 和mid + k 中间 x 距离左侧比距离右侧更近
		if x-arr[mid] > arr[mid+k]-x {
			left = mid + 1
		} else {
			right = mid
		}
	}

	for i := 0; i < k; i++ {
		result[i] = arr[left+i]
	}

	return result
}

// 668. 乘法表中第k小的数
// 几乎每一个人都用 乘法表。但是你能在乘法表中快速找到第k小的数字吗？
//
// 给定高度m 、宽度n 的一张 m * n的乘法表，以及正整数k，你需要返回表中第k 小的数字。
//
// 例 1：
// 输入: m = 3, n = 3, k = 5
// 输出: 3
// 解释:
// 乘法表:
// 1	2	3
// 2	4	6
// 3	6	9
// 第5小的数字是 3 (1, 2, 2, 3, 3).
//
// 例 2：
// 输入: m = 2, n = 3, k = 6
// 输出: 6
// 解释:
// 乘法表:
// 1	2	3
// 2	4	6
//
// 第6小的数字是 6 (1, 2, 2, 3, 4, 6).
// 注意：
//
// m 和 n 的范围在 [1, 30000] 之间。
// k 的范围在 [1, m * n] 之间。
func findKthNumber(m int, n int, k int) int {
	if k == 1 {
		return 1
	}
	if k == m*n {
		return m * n
	}
	// 寻找 比 num 小 的 元素个数
	getNumLess := func(num int) int {
		count := 0
		for i := 1; i <= m; i++ {
			count += min(num/i, n)
		}

		return count
	}

	// 二分
	low, high := 1, m*n
	for low < high {
		mid := (low + high) >> 1
		if getNumLess(mid) >= k {
			high = mid
		} else {
			low = mid + 1
		}
	}
	return low
}
func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

// 719. 找出第 k 小的距离对
// 给定一个整数数组，返回所有数对之间的第 k 个最小距离。一对 (A, B) 的距离被定义为 A 和 B 之间的绝对差值。
//
// 示例 1:
// 输入：
// nums = [1,3,1]
// k = 1
// 输出：0
// 解释：
// 所有数对如下：
// (1,3) -> 2
// (1,1) -> 0
// (3,1) -> 2
// 因此第 1 个最小距离的数对是 (1,1)，它们之间的距离为 0。
//
// 提示:
// 2 <= len(nums) <= 10000.
// 0 <= nums[i] < 1000000.
// 1 <= k <= len(nums) * (len(nums) - 1) / 2.
func smallestDistancePair(nums []int, k int) int {

	// 思路分析： 与 LeetCode 乘法表中第k小的数（二分搜索） 非常类似。
	// 最大的距离必定是最大值 - 最小值，而最小的距离可能是0（也可能比0大）。所以我们采用二分法，
	// 初始化left = 0， right = 最大距离。对于mid = (left + right) / 2，我们计算距离小于等于k个距离对的个数shortDisMid。
	//
	// 如果shortDisMid < k， 则第k小的距离对必定不会出现在[left, mid]，所以修改left = mid + 1
	// 否则 right = mid
	sort.Ints(nums)
	n := len(nums)

	countDistance := func(num int) int {
		count := 0
		right := n - 1
		for left := n - 2; left >= 0; left-- {
			for left < right && nums[right]-nums[left] > num {
				right--
			}
			count += right - left
		}
		return count
	}
	left, right := 0, nums[n-1]-nums[0]

	for left < right {
		mid := (left + right) >> 1
		// 获取距离大小不超过k的距离对的个数
		count := countDistance(mid)
		if count < k {
			left = mid + 1
		} else {
			right = mid
		}
	}

	return left
}

// 875. 爱吃香蕉的珂珂
// 珂珂喜欢吃香蕉。这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 h 小时后回来。
//
// 珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。
//
// 珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
// 返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。
//
// 示例 1：
// 输入：piles = [3,6,7,11], h = 8 输出：4
//
// 示例 2：
// 输入：piles = [30,11,23,4,20], h = 5 输出：30
//
// 示例 3：
// 输入：piles = [30,11,23,4,20], h = 6 输出：23
//
// 提示：
// 1 <= piles.length <= 104
// piles.length <= h <= 109
// 1 <= piles[i] <= 109
func minEatingSpeed(piles []int, h int) int {
	n := len(piles)
	low, high := 1, 0
	for _, pile := range piles {
		high = max(high, pile)
	}
	if n == h {
		return high
	}

	canFinish := func(speed int) bool {
		time := 0
		for _, pile := range piles {
			time += (pile-1)/speed + 1
		}
		return time <= h
	}

	for low < high {
		mid := (low + high) >> 1
		if canFinish(mid) {
			high = mid
		} else {
			low = mid + 1
		}
	}
	return low
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 744. 寻找比目标字母大的最小字母
// 给你一个排序后的字符列表 letters ，列表中只包含小写英文字母。另给出一个目标字母 target，请你寻找在这一有序列表里比目标字母大的最小字母。
//
// 在比较时，字母是依序循环出现的。举个例子：
// 如果目标字母 target = 'z' 并且字符列表为 letters = ['a', 'b']，则答案返回 'a'
//
// 示例 1：
// 输入: letters = ["c", "f", "j"]，target = "a"
// 输出: "c"
//
// 示例 2:
// 输入: letters = ["c","f","j"], target = "c"
// 输出: "f"
//
// 示例 3:
// 输入: letters = ["c","f","j"], target = "d"
// 输出: "f"
//
// 提示：
// 2 <= letters.length <= 104
// letters[i] 是一个小写字母
// letters 按非递减顺序排序
// letters 最少包含两个不同的字母
// target 是一个小写字母
func nextGreatestLetter(letters []byte, target byte) byte {
	n := len(letters)

	if target >= letters[0] && target < letters[n-1] {
		left, right := 0, n-1
		for left < right {
			mid := (left + right) >> 1
			if letters[mid] > target {
				right = mid
			} else {
				left = mid + 1
			}
		}

		return letters[right]
	}
	return letters[0]

}

// 2594. 修车的最少时间
// 给你一个整数数组 ranks ，表示一些机械工的 能力值 。ranksi 是第 i 位机械工的能力值。能力值为 r 的机械工可以在 r * n2 分钟内修好 n 辆车。
// 同时给你一个整数 cars ，表示总共需要修理的汽车数目。
// 请你返回修理所有汽车 最少 需要多少时间。
// 注意：所有机械工可以同时修理汽车。
//
// 示例 1：
// 输入：ranks = [4,2,3,1], cars = 10
// 输出：16
// 解释：
// - 第一位机械工修 2 辆车，需要 4 * 2 * 2 = 16 分钟。
// - 第二位机械工修 2 辆车，需要 2 * 2 * 2 = 8 分钟。
// - 第三位机械工修 2 辆车，需要 3 * 2 * 2 = 12 分钟。
// - 第四位机械工修 4 辆车，需要 1 * 4 * 4 = 16 分钟。
// 16 分钟是修理完所有车需要的最少时间。
//
// 示例 2：
// 输入：ranks = [5,1,8], cars = 6
// 输出：16
// 解释：
// - 第一位机械工修 1 辆车，需要 5 * 1 * 1 = 5 分钟。
// - 第二位机械工修 4 辆车，需要 1 * 4 * 4 = 16 分钟。
// - 第三位机械工修 1 辆车，需要 8 * 1 * 1 = 8 分钟。
// 16 分钟时修理完所有车需要的最少时间。
//
// 提示：
// 1 <= ranks.length <= 105
// 1 <= ranks[i] <= 100
// 1 <= cars <= 106
func repairCars(ranks []int, cars int) int64 {

	left, right := 1, ranks[0]*cars*cars

	var canRepair = func(minute int) bool {
		count := 0
		for _, rank := range ranks {
			t := float64(minute) / float64(rank)
			count += int(math.Sqrt(t))
		}
		return count >= cars
	}

	for left < right {
		mid := (left + right) >> 1
		if canRepair(mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return int64(right)
}
