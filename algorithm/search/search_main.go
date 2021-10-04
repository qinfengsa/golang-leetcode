package search

import (
	"fmt"
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

func searchTest() {
	nums := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1}
	target := 2
	result := search2(nums, target)
	fmt.Println(result)
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
//     或者返回索引 5， 其峰值元素为 6。
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
