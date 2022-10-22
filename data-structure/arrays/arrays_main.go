package arrays

import (
	"container/list"
	"fmt"
	"log"
	"math"
	"math/bits"
	"sort"
	"strconv"
)

func twoSumTest() {
	nums := []int{2, 7, 11, 15}
	target := 9
	result := twoSum(nums, target)
	fmt.Println(result)
}

// 1. 两数之和
// 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
// 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
// 示例: 给定 nums = [2, 7, 11, 15], target = 9
// 因为 nums[0] + nums[1] = 2 + 7 = 9
// 所以返回 [0, 1]
func twoSum(nums []int, target int) []int {
	numMap := make(map[int]int)
	result := []int{0, 0}
	for i, num := range nums {
		tmp := target - num
		idx, ok := numMap[tmp]
		if ok {
			result[0] = idx
			result[1] = i
			break
		}
		numMap[num] = i
	}
	return result
}

// 26. 删除排序数组中的重复项
// 给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
//
// 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
//
// 示例 1:
// 给定数组 nums = [1,1,2],
// 函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。
// 你不需要考虑数组中超出新长度后面的元素。
//
// 示例 2:
// 给定 nums = [0,0,1,1,1,2,2,3,3,4],
// 函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
// 你不需要考虑数组中超出新长度后面的元素。
//
// 说明:
// 为什么返回数值是整数，但输出的答案是数组呢?
// 请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
// 你可以想象内部操作如下:
//  nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
// int len = removeDuplicates(nums);
// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
// for (int i = 0; i < len; i++) {
//    print(nums[i]);
// }
func removeDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	idx := 0
	for i, num := range nums {
		if i == 0 {
			continue
		}
		if num != nums[idx] {
			idx++
			nums[idx] = num
		}
	}

	return idx + 1
}

// 27. 移除元素
// 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
//
// 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
//
// 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
// 示例 1:
//
// 给定 nums = [3,2,2,3], val = 3,
//
// 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
//
// 你不需要考虑数组中超出新长度后面的元素。
// 示例 2:
//
// 给定 nums = [0,1,2,2,3,0,4,2], val = 2,
//
// 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
//
// 注意这五个元素可为任意顺序。
//
// 你不需要考虑数组中超出新长度后面的元素。
func removeElement(nums []int, val int) int {
	if len(nums) == 0 {
		return 0
	}
	idx := 0
	for _, num := range nums {
		if num != val {
			nums[idx] = num
			idx++
		}
	}

	return idx
}

// 53. 最大子序和
// 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
//
// 示例: 输入: [-2,1,-3,4,-1,2,1,-5,4] 输出: 6
// 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
// 进阶: 如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。
func maxSubArray(nums []int) int {
	max, sum := nums[0], nums[0]
	// dp[i] = Math.max(dp[i - 1] + nums[i] , nums[i])
	for i := 1; i < len(nums); i++ {
		if sum > 0 {
			sum += nums[i]
		} else {
			sum = nums[i]
		}
		if sum > max {
			max = sum
		}
	}

	return max
}

// 66. 加一
// 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
//
// 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
//
// 你可以假设除了整数 0 之外，这个整数不会以零开头。
//
// 示例 1: 输入: [1,2,3] 输出: [1,2,4]
// 解释: 输入数组表示数字 123。
// 示例 2: 输入: [4,3,2,1] 输出: [4,3,2,2]
// 解释: 输入数组表示数字 4321。
func plusOne(digits []int) []int {
	length := len(digits)
	last := 1
	for i := length - 1; i >= 0; i-- {
		if last == 0 {
			break
		}
		num := digits[i] + last
		if num >= 10 {
			num -= 10
			last = 1
		} else {
			last = 0
		}
		digits[i] = num
	}
	if last > 0 {
		var result []int
		result = append(result, 1)
		result = append(result, digits...)
		return result
	}
	return digits
}

// 88. 合并两个有序数组
// 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
// 说明: 初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。
// 你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
//
// 示例: 输入: nums1 = [1,2,3,0,0,0], m = 3 nums2 = [2,5,6],       n = 3
//
// 输出: [1,2,2,3,5,6]
func merge(nums1 []int, m int, nums2 []int, n int) {
	// 从后往前
	i, j, idx := m-1, n-1, m+n-1
	for i >= 0 || j >= 0 {
		if i < 0 {
			nums1[idx] = nums2[j]
			j--
			idx--
			continue
		}
		if j < 0 {
			nums1[idx] = nums1[i]
			i--
			idx--
			continue
		}
		if nums1[i] >= nums2[j] {
			nums1[idx] = nums1[i]
			i--
		} else {
			nums1[idx] = nums2[j]
			j--
		}
		idx--
	}
}

// 118. 杨辉三角
// 给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
//
//
//
// 在杨辉三角中，每个数是它左上方和右上方的数的和。
//
// 示例: 输入: 5
// 输出:
// [
//      [1],
//     [1,1],
//    [1,2,1],
//   [1,3,3,1],
//  [1,4,6,4,1]
// ]
func generate(numRows int) [][]int {
	result := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		result[i] = make([]int, i+1)
		result[i][0] = 1
		result[i][i] = 1
		for j := 1; j < i; j++ {
			result[i][j] = result[i-1][j-1] + result[i-1][j]
		}
	}
	return result
}

// 119. 杨辉三角 II
// 给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。
//
// 在杨辉三角中，每个数是它左上方和右上方的数的和。
//
// 示例: 输入: 3 输出: [1,3,3,1]
// 进阶：
//
// 你可以优化你的算法到 O(k) 空间复杂度吗？
func getRow(rowIndex int) []int {
	result := make([]int, rowIndex+1)
	result[0] = 1
	for i := 1; i <= rowIndex; i++ {
		result[i] = 1
		for j := i - 1; j > 0; j-- {
			result[j] = result[j-1] + result[j]
		}
	}

	return result
}

// 167. 两数之和 II - 输入有序数组
// 给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
//
// 函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
//
// 说明:
//
// 返回的下标值（index1 和 index2）不是从零开始的。
// 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
// 示例:
//
// 输入: numbers = [2, 7, 11, 15], target = 9 输出: [1,2]
// 解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
func twoSum2(numbers []int, target int) []int {
	left, right := 0, len(numbers)-1
	for left < right {
		sum := numbers[left] + numbers[right]
		if sum == target {
			return []int{left + 1, right + 1}
		} else if sum < target {
			left++
		} else {
			right--
		}
	}
	return []int{-1, -1}
}

// 169. 多数元素
// 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
//
// 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
// 示例 1:
// 输入: [3,2,3] 输出: 3
// 示例 2:
// 输入: [2,2,1,1,1,2,2] 输出: 2
func majorityElement(nums []int) int {
	count := 0
	var result int
	for _, num := range nums {
		if count == 0 {
			result = num
		}
		if result == num {
			count++
		} else {
			count--
		}
	}
	return result
}

// 283. 移动零
// 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
//
// 示例: 输入: [0,1,0,3,12] 输出: [1,3,12,0,0]
// 说明:
//
// 必须在原数组上操作，不能拷贝额外的数组。
// 尽量减少操作次数。
func moveZeroes(nums []int) {
	index := 0
	for _, num := range nums {
		if num == 0 {
			continue
		}
		nums[index] = num
		index++
	}
	for i := index; i < len(nums); i++ {
		nums[i] = 0
	}
}

// 463. 岛屿的周长
// 给定一个包含 0 和 1 的二维网格地图，其中 1 表示陆地 0 表示水域。
//
// 网格中的格子水平和垂直方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
//
// 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。
// 示例 :
//
// 输入:
// [[0,1,0,0],
//  [1,1,1,0],
//  [0,1,0,0],
//  [1,1,0,0]]
//
// 输出: 16
//
// 解释: 它的周长是下面图片中的 16 个黄色的边：
func islandPerimeter(grid [][]int) int {
	rows, cols := len(grid), len(grid[0])
	result := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if grid[i][j] == 0 {
				continue
			}
			result += 4
			if i > 0 && grid[i-1][j] == 1 {
				result -= 2
			}
			if j > 0 && grid[i][j-1] == 1 {
				result -= 2
			}
		}
	}

	return result
}

// 414. 第三大的数
// 给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。
//
// 示例 1: 输入: [3, 2, 1] 输出: 1
// 解释: 第三大的数是 1.
//
// 示例 2:
// 输入: [1, 2] 输出: 2
// 解释: 第三大的数不存在, 所以返回最大的数 2 .
//
// 示例 3:
// 输入: [2, 2, 3, 1] 输出: 1
// 解释: 注意，要求返回第三大的数，是指第三大且唯一出现的数。
// 存在两个值为2的数，它们都排第二。
func thirdMax(nums []int) int {
	size := len(nums)
	min := -2147483648
	if size == 1 {
		return nums[0]
	}
	first, second, third := nums[0], min, min

	numMap := map[int]bool{}
	numMap[first] = true
	for i := 1; i < size; i++ {
		num := nums[i]
		if !numMap[num] && len(numMap) < 3 {
			numMap[num] = true
		}

		if num == first || num == second || num == third {
			continue
		}
		if num > first {
			third, second, first = second, first, num
		} else if num > second {
			third, second = second, num
		} else if nums[i] > third {
			third = nums[i]
		}
	}
	if len(numMap) < 3 {
		return first
	}
	return third
}

// 448. 找到所有数组中消失的数字
// 给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
//
// 找到所有在 [1, n] 范围之间没有出现在数组中的数字。
//
// 您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。
//
// 示例:
//
// 输入:
// [4,3,2,7,8,2,3,1]
//
// 输出:
// [5,6]
func findDisappearedNumbers(nums []int) []int {

	var result []int
	for _, num := range nums {
		if num < 0 {
			num = -num
		}
		index := num - 1
		if nums[index] > 0 {
			nums[index] *= -1
		}
	}
	for i, num := range nums {
		if num > 0 {
			result = append(result, i+1)
		}
	}
	return result
}

// 453. 最小移动次数使数组元素相等
// 给定一个长度为 n 的非空整数数组，找到让数组所有元素相等的最小移动次数。每次移动将会使 n - 1 个元素增加 1。
//
// 示例:
// 输入: [1,2,3] 输出: 3
//
// 解释:
// 只需要3次移动（注意每次移动会增加两个元素的值）：
//
// [1,2,3]  =>  [2,3,3]  =>  [3,4,3]  =>  [4,4,4]
func minMoves(nums []int) int {
	min := nums[0]
	for i := 0; i < len(nums); i++ {
		if nums[i] < min {
			min = nums[i]
		}
	}
	result := 0
	for _, num := range nums {
		result += num - min
	}

	return result
}

// 941. 有效的山脉数组
// 给定一个整数数组 A，如果它是有效的山脉数组就返回 true，否则返回 false。
//
// 让我们回顾一下，如果 A 满足下述条件，那么它是一个山脉数组：
//
// A.length >= 3
// 在 0 < i < A.length - 1 条件下，存在 i 使得：
// A[0] < A[1] < ... A[i-1] < A[i]
// A[i] > A[i+1] > ... > A[A.length - 1]
//
// 示例 1：
// 输入：[2,1] 输出：false
//
// 示例 2：
// 输入：[3,5,5] 输出：false
//
// 示例 3：
// 输入：[0,3,2,1] 输出：true
//
// 提示：
//
// 0 <= A.length <= 10000
// 0 <= A[i] <= 10000
func validMountainArray(A []int) bool {
	size := len(A)
	i, j := 0, size-1
	for i+1 < size && A[i] < A[i+1] {
		i++
	}
	if i == j {
		return false
	}
	for j-1 >= 0 && A[j-1] < A[j] {
		j--
	}

	return i == j && j != 0
}

// 485. 最大连续1的个数
// 给定一个二进制数组， 计算其中最大连续1的个数。
//
// 示例 1:
//
// 输入: [1,1,0,1,1,1] 输出: 3
// 解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.
// 注意：
//
// 输入的数组只包含 0 和1。
// 输入数组的长度是正整数，且不超过 10,000。
func findMaxConsecutiveOnes(nums []int) int {
	maxVal, count := 0, 0
	for _, num := range nums {
		if num == 0 {
			count = 0
		} else {
			count++
			maxVal = max(maxVal, count)
		}
	}
	return maxVal
}

// 506. 相对名次
// 给出 N 名运动员的成绩，找出他们的相对名次并授予前三名对应的奖牌。前三名运动员将会被分别授予 “金牌”，“银牌” 和“ 铜牌”（"Gold Medal", "Silver Medal", "Bronze Medal"）。
//
// (注：分数越高的选手，排名越靠前。)
//
// 示例 1:
// 输入: [5, 4, 3, 2, 1]
// 输出: ["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
// 解释: 前三名运动员的成绩为前三高的，因此将会分别被授予 “金牌”，“银牌”和“铜牌” ("Gold Medal", "Silver Medal" and "Bronze Medal").
// 余下的两名运动员，我们只需要通过他们的成绩计算将其相对名次即可。
// 提示:
//
// N 是一个正整数并且不会超过 10000。
// 所有运动员的成绩都不相同。
func findRelativeRanks(score []int) []string {
	// 所有运动员的成绩都不相同
	indexMap := make(map[int]int)
	for i, num := range score {
		indexMap[num] = i
	}
	sort.Ints(score)
	n := len(score)
	result := make([]string, n)
	var getRank = func(rank int) string {
		switch rank {
		case 1:
			return "Gold Medal"
		case 2:
			return "Silver Medal"
		case 3:
			return "Bronze Medal"
		default:
			return strconv.Itoa(rank)
		}
	}
	for i, num := range score {
		rank := n - i
		index := indexMap[num]
		result[index] = getRank(rank)
	}

	return result
}

// 1356. 根据数字二进制下 1 的数目排序
// 给你一个整数数组 arr 。请你将数组中的元素按照其二进制表示中数字 1 的数目升序排序。
//
// 如果存在多个数字二进制中 1 的数目相同，则必须将它们按照数值大小升序排列。
//
// 请你返回排序后的数组。
// 示例 1：
//
// 输入：arr = [0,1,2,3,4,5,6,7,8]
// 输出：[0,1,2,4,8,3,5,6,7]
// 解释：[0] 是唯一一个有 0 个 1 的数。
// [1,2,4,8] 都有 1 个 1 。
// [3,5,6] 有 2 个 1 。
// [7] 有 3 个 1 。
// 按照 1 的个数排序得到的结果数组为 [0,1,2,4,8,3,5,6,7]
//
// 示例 2：
// 输入：arr = [1024,512,256,128,64,32,16,8,4,2,1]
// 输出：[1,2,4,8,16,32,64,128,256,512,1024]
// 解释：数组中所有整数二进制下都只有 1 个 1 ，所以你需要按照数值大小将它们排序。
//
// 示例 3：
// 输入：arr = [10000,10000] 输出：[10000,10000]
//
// 示例 4：
// 输入：arr = [2,3,5,7,11,13,17,19] 输出：[2,3,5,17,7,11,13,19]
//
// 示例 5：
// 输入：arr = [10,100,1000,10000] 输出：[10,100,10000,1000]
//
// 提示：
// 1 <= arr.length <= 500
// 0 <= arr[i] <= 10^4
func sortByBits(arr []int) []int {
	sort.Slice(arr, func(i, j int) bool {
		a, b := bits.OnesCount(uint(arr[i])), bits.OnesCount(uint(arr[j]))
		return a < b || a == b && arr[i] < arr[j]
	})
	return arr
}

// 561. 数组拆分 I
// 给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，
// 使得从 1 到 n 的 min(ai, bi) 总和最大。
// 返回该 最大总和 。
//
// 示例 1：
// 输入：nums = [1,4,3,2] 输出：4
// 解释：所有可能的分法（忽略元素顺序）为：
// 1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
// 2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
// 3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
// 所以最大总和为 4
//
// 示例 2：
// 输入：nums = [6,2,6,5,1,2] 输出：9
// 解释：最优的分法为 (2, 1), (2, 5), (6, 6). min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9
//
// 提示：
// 1 <= n <= 104
// nums.length == 2 * n
// -104 <= nums[i] <= 104
func arrayPairSum(nums []int) int {
	sort.Ints(nums)
	result := 0
	for i := 0; i < len(nums); i += 2 {
		result += nums[i]
	}
	return result
}

// 566. 重塑矩阵
// 在MATLAB中，有一个非常有用的函数 reshape，它可以将一个矩阵重塑为另一个大小不同的新矩阵，但保留其原始数据。
// 给出一个由二维数组表示的矩阵，以及两个正整数r和c，分别表示想要的重构的矩阵的行数和列数。
// 重构后的矩阵需要将原始矩阵的所有元素以相同的行遍历顺序填充。
// 如果具有给定参数的reshape操作是可行且合理的，则输出新的重塑矩阵；否则，输出原始矩阵。
//
// 示例 1:
// 输入:  nums =  [[1,2], [3,4]] r = 1, c = 4
// 输出:  [[1,2,3,4]]
// 解释:
// 行遍历nums的结果是 [1,2,3,4]。新的矩阵是 1 * 4 矩阵, 用之前的元素值一行一行填充新矩阵。
//
// 示例 2:
// 输入:  nums = [[1,2], [3,4]]  r = 2, c = 4
// 输出: [[1,2], [3,4]]
// 解释:
// 没有办法将 2 * 2 矩阵转化为 2 * 4 矩阵。 所以输出原矩阵。
// 注意：
// 给定矩阵的宽和高范围在 [1, 100]。
// 给定的 r 和 c 都是正数。
func matrixReshape(mat [][]int, r int, c int) [][]int {
	rows, cols := len(mat), len(mat[0])
	if rows*cols != r*c {
		return mat
	}
	var result = make([][]int, r)
	for i := 0; i < r; i++ {
		result[i] = make([]int, c)
	}
	rowIndex, colIndex := 0, 0
	for _, rowNum := range mat {
		for _, num := range rowNum {
			result[rowIndex][colIndex] = num
			colIndex++
			if colIndex == c {
				rowIndex++
				colIndex = 0
			}
		}
	}
	return result
}

// 922. 按奇偶排序数组 II
// 给定一个非负整数数组 A， A 中一半整数是奇数，一半整数是偶数。
//
// 对数组进行排序，以便当 A[i] 为奇数时，i 也是奇数；当 A[i] 为偶数时， i 也是偶数。
//
// 你可以返回任何满足上述条件的数组作为答案。
// 示例：
// 输入：[4,2,5,7] 输出：[4,5,2,7]
// 解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
func sortArrayByParityII(A []int) []int {
	size := len(A)
	even, odd := 0, 1
	for {
		for even < size && A[even]&1 == 0 {
			even += 2
		}
		for odd < size && A[odd]&1 == 1 {
			odd += 2
		}
		if even >= size || odd >= size {
			break
		}
		A[even], A[odd] = A[odd], A[even]
	}
	return A
	/*result := make([]int, size)
	idx1, idx2 := 0, 1
	for _, num := range A {
		if num&1 == 0 {
			result[idx1] = num
			idx1 += 2
		} else {
			result[idx2] = num
			idx2 += 2
		}
	}
	return result*/
}

// 594. 最长和谐子序列
// 和谐数组是指一个数组里元素的最大值和最小值之间的差别正好是1。
//
// 现在，给定一个整数数组，你需要在所有可能的子序列中找到最长的和谐子序列的长度。
//
// 示例 1:
// 输入: [1,3,2,2,5,2,3,7] 输出: 5
// 原因: 最长的和谐数组是：[3,2,2,2,3].
// 说明: 输入的数组长度最大不超过20,000.
func findLHS(nums []int) int {

	countMap := map[int]int{}
	max := 0
	for _, num := range nums {
		count := countMap[num]
		countMap[num] = count + 1
	}
	for k, v := range countMap {
		if count, ok := countMap[k+1]; ok {
			count += v
			if count > max {
				max = count
			}
		}
	}
	return max
}

// 605. 种花问题
// 假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
//
// 给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。

// 示例 1:
// 输入: flowerbed = [1,0,0,0,1], n = 1 输出: True
//
// 示例 2:
// 输入: flowerbed = [1,0,0,0,1], n = 2 输出: False
//
// 注意:
// 数组内已种好的花不会违反种植规则。
// 输入的数组长度范围为 [1, 20000]。
// n 是非负整数，且不会超过输入数组的大小。
func canPlaceFlowers(flowerbed []int, n int) bool {
	size := len(flowerbed)
	if n == 0 {
		return true
	}
	if size == 1 {
		return flowerbed[0] == 0
	}
	for i := 0; i < size; i++ {
		if n == 0 {
			return true
		}
		if flowerbed[i] == 1 {
			continue
		}
		if i == 0 {
			if flowerbed[i+1] == 0 {
				flowerbed[i] = 1
				n--
			}

			continue
		}
		if i == size-1 {
			if flowerbed[i-1] == 0 {
				flowerbed[i] = 1
				n--
			}

			continue
		}
		if flowerbed[i-1] == 0 && flowerbed[i+1] == 0 {
			flowerbed[i] = 1
			n--
		}

	}
	return n == 0
}

// 643. 子数组最大平均数 I
// 给定 n 个整数，找出平均数最大且长度为 k 的连续子数组，并输出该最大平均数。
//
// 示例 1:
// 输入: [1,12,-5,-6,50,3], k = 4 输出: 12.75
// 解释: 最大平均数 (12-5-6+50)/4 = 51/4 = 12.75
//
// 注意: 1 <= k <= n <= 30,000。
// 所给数据范围 [-10,000，10,000]。
func findMaxAverage(nums []int, k int) float64 {
	sum, size := 0, len(nums)
	for i := 0; i < k; i++ {
		sum += nums[i]
	}
	maxVal := sum
	for i := k; i < size; i++ {
		sum += nums[i] - nums[i-k]
		if sum > maxVal {
			maxVal = sum
		}
	}
	return float64(maxVal) / float64(k)
}

// 645. 错误的集合
// 集合 S 包含从1到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个元素复制了成了集合里面的另外一个元素的值，导致集合丢失了一个整数并且有一个元素重复。
//
// 给定一个数组 nums 代表了集合 S 发生错误后的结果。你的任务是首先寻找到重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。
//
// 示例 1:
// 输入: nums = [1,2,2,4] 输出: [2,3]
//
// 注意: 给定数组的长度范围是 [2, 10000]。
// 给定的数组是无序的。
func findErrorNums(nums []int) []int {
	n := len(nums)
	arr := make([]bool, n+1)
	repeat, defect := 0, 0
	for i, num := range nums {
		defect ^= i + 1
		if arr[num] {
			repeat = num
		} else {
			arr[num] = true
			defect ^= num
		}
	}
	result := make([]int, 2)
	result[0] = repeat
	result[1] = defect
	return result
}

// 661. 图片平滑器
// 包含整数的二维矩阵 M 表示一个图片的灰度。你需要设计一个平滑器来让每一个单元的灰度成为平均灰度 (向下舍入) ，平均灰度的计算是周围的8个单元和它本身的值求平均，如果周围的单元格不足八个，则尽可能多的利用它们。
//
// 示例 1:
// 输入:
// [[1,1,1],
// [1,0,1],
// [1,1,1]]
// 输出:
// [[0, 0, 0],
// [0, 0, 0],
// [0, 0, 0]]
// 解释:
// 对于点 (0,0), (0,2), (2,0), (2,2): 平均(3/4) = 平均(0.75) = 0
// 对于点 (0,1), (1,0), (1,2), (2,1): 平均(5/6) = 平均(0.83333333) = 0
// 对于点 (1,1): 平均(8/9) = 平均(0.88888889) = 0
// 注意:
// 给定矩阵中的整数范围为 [0, 255]。
// 矩阵的长和宽的范围均为 [1, 150]。
func imageSmoother(img [][]int) [][]int {
	rows, cols := len(img), len(img[0])

	result := make([][]int, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]int, cols)
		for j := 0; j < cols; j++ {
			var count, sum = 1, img[i][j]
			left, right, up, down := false, false, false, false
			if i-1 >= 0 {
				count++
				up = true
				sum += img[i-1][j]
			}
			if j-1 >= 0 {
				count++
				left = true
				sum += img[i][j-1]
			}
			if i+1 < rows {
				count++
				down = true
				sum += img[i+1][j]
			}
			if j+1 < cols {
				count++
				right = true
				sum += img[i][j+1]
			}

			if left && up {
				count++
				sum += img[i-1][j-1]
			}
			if left && down {
				count++
				sum += img[i+1][j-1]
			}
			if right && up {
				count++
				sum += img[i-1][j+1]
			}
			if right && down {
				count++
				sum += img[i+1][j+1]
			}
			result[i][j] = sum / count
		}
	}

	return result
}

// 665. 非递减数列
// 给你一个长度为 n 的整数数组，请你判断在 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。
//
// 我们是这样定义一个非递减数列的： 对于数组中所有的 i (0 <= i <= n-2)，总满足 nums[i] <= nums[i + 1]。
//
// 示例 1:
// 输入: nums = [4,2,3] 输出: true
// 解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。
//
// 示例 2:
// 输入: nums = [4,2,1] 输出: false
// 解释: 你不能在只改变一个元素的情况下将其变为非递减数列。
//
// 说明：
// 1 <= n <= 10 ^ 4
// - 10 ^ 5 <= nums[i] <= 10 ^ 5
func checkPossibility(nums []int) bool {
	n := len(nums)
	if n <= 2 {
		return true
	}

	count := 0
	for i := 1; i < n; i++ {
		if nums[i-1] > nums[i] {
			count++
			if count > 1 {
				return false
			}
			if i-2 >= 0 && nums[i-2] > nums[i] {
				nums[i] = nums[i-1]
			} else {
				nums[i-1] = nums[i]
			}
		}
	}

	return true
}

// 674. 最长连续递增序列
// 给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。
//
// 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
//
// 示例 1：
// 输入：nums = [1,3,5,4,7] 输出：3
// 解释：最长连续递增序列是 [1,3,5], 长度为3。
// 尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。
//
// 示例 2：
// 输入：nums = [2,2,2,2,2] 输出：1
// 解释：最长连续递增序列是 [2], 长度为1。
//
// 提示：
// 0 <= nums.length <= 104
// -109 <= nums[i] <= 109
func findLengthOfLCIS(nums []int) int {
	size := len(nums)
	if size == 0 {
		return 0
	}
	maxCnt, count := 1, 1
	for i := 1; i < size; i++ {
		if nums[i] > nums[i-1] {
			count++
		} else {
			count = 1
		}
		if count > maxCnt {
			maxCnt = count
		}
	}
	return maxCnt
}

// 697. 数组的度
// 给定一个非空且只包含非负数的整数数组 nums, 数组的度的定义是指数组里任一元素出现频数的最大值。
//
// 你的任务是找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。
//
// 示例 1:
// 输入: [1, 2, 2, 3, 1] 输出: 2
// 解释:
// 输入数组的度是2，因为元素1和2的出现频数最大，均为2.
// 连续子数组里面拥有相同度的有如下所示:
// [1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
// 最短连续子数组[2, 2]的长度为2，所以返回2.
//
// 示例 2:
// 输入: [1,2,2,3,1,4,2]
// 输出: 6
func findShortestSubArray(nums []int) int {
	maxVal, result := 0, len(nums)
	countMap := map[int]int{}
	indexMap := map[int]int{}
	for i, num := range nums {
		count := countMap[num]
		if count == 0 {
			indexMap[num] = i
		}
		count++
		countMap[num] = count
		size := i - indexMap[num] + 1
		if count > maxVal {
			maxVal = count
			result = size
		} else if count == maxVal && size < result {
			result = size
		}

	}
	return result
}

// 717. 1比特与2比特字符
// 有两种特殊字符。第一种字符可以用一比特0来表示。第二种字符可以用两比特(10 或 11)来表示。
//
// 现给一个由若干比特组成的字符串。问最后一个字符是否必定为一个一比特字符。给定的字符串总是由0结束。
//
// 示例 1:
// 输入: bits = [1, 0, 0] 输出: True
// 解释:  唯一的编码方式是一个两比特字符和一个一比特字符。所以最后一个字符是一比特字符。
//
// 示例 2:
// 输入:  bits = [1, 1, 1, 0] 输出: False
// 解释:  唯一的编码方式是两比特字符和两比特字符。所以最后一个字符不是一比特字符。
// 注意:
//
// 1 <= len(bits) <= 1000.
// bits[i] 总是0 或 1.
func isOneBitCharacter(bits []int) bool {
	size := len(bits)
	if bits[size-1] != 0 {
		return false
	}
	if size == 1 {
		return true
	}
	if bits[size-2] == 0 {
		return true
	}
	i := 0
	for i < size-1 {
		i += bits[i] + 1
	}

	return i == size-1
}

// 给定由若干0和1组成的数组 A。我们定义N_i：从A[0] 到A[i]的第 i个子数组被解释为一个二进制数（从最高有效位到最低有效位）。
//
// 返回布尔值列表answer，只有当N_i可以被 5整除时，答案answer[i] 为true，否则为 false。
// 示例 1：
// 输入：[0,1,1] 输出：[true,false,false]
// 解释：
// 输入数字为 0, 01, 011；也就是十进制中的 0, 1, 3 。只有第一个数可以被 5 整除，因此 answer[0] 为真。
//
// 示例 2：
// 输入：[1,1,1] 输出：[false,false,false]
//
// 示例 3：
// 输入：[0,1,1,1,1,1] 输出：[true,false,false,false,true,false]
//
// 示例4：
// 输入：[1,1,1,0,1] 输出：[false,false,false,false,false]
//
// 提示： 1 <= A.length <= 30000 A[i] 为0或1
//

func prefixesDivBy5(A []int) []bool {
	size := len(A)
	var result = make([]bool, size)
	num := 0
	for i, a := range A {
		num = (num << 1) + a
		if num >= 10 {
			num %= 10
		}
		if num == 0 || num == 5 {
			result[i] = true
		}

	}
	return result
}

func pivotIndexTest() {
	nums := []int{-1, -1, -1, 0, 1, 1}
	pivotIndex(nums)
}

// 724. 寻找数组的中心索引
// 给定一个整数类型的数组 nums，请编写一个能够返回数组 “中心索引” 的方法。
//
// 我们是这样定义数组 中心索引 的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
//
// 如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。
//
// 示例 1：
// 输入： nums = [1, 7, 3, 6, 5, 6] 输出：3
// 解释： 索引 3 (nums[3] = 6) 的左侧数之和 (1 + 7 + 3 = 11)，与右侧数之和 (5 + 6 = 11) 相等。
// 同时, 3 也是第一个符合要求的中心索引。
//
// 示例 2：
// 输入： nums = [1, 2, 3] 输出：-1
// 解释： 数组中不存在满足此条件的中心索引。
//
// 说明： nums 的长度范围为 [0, 10000]。
// 任何一个 nums[i] 将会是一个范围在 [-1000, 1000]的整数。
func pivotIndex(nums []int) int {
	size := len(nums)
	if size <= 1 {
		return -1
	}
	sums, sum := make([]int, size+1), 0
	for i := 0; i < size; i++ {
		sum += nums[i]
		sums[i+1] = sum
	}
	log.Println(sum)
	log.Println(sums)
	for i := 0; i < size; i++ {
		if sums[i] == sum-sums[i+1] {
			return i
		}
	}
	return -1
}

// 888. 公平的糖果棒交换
// 爱丽丝和鲍勃有不同大小的糖果棒：A[i] 是爱丽丝拥有的第 i 根糖果棒的大小，B[j] 是鲍勃拥有的第 j 根糖果棒的大小。
//
// 因为他们是朋友，所以他们想交换一根糖果棒，这样交换后，他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有的糖果棒大小的总和。）
//
// 返回一个整数数组 ans，其中 ans[0] 是爱丽丝必须交换的糖果棒的大小，ans[1] 是 Bob 必须交换的糖果棒的大小。
//
// 如果有多个答案，你可以返回其中任何一个。保证答案存在。
//
// 示例 1： 输入：A = [1,1], B = [2,2] 输出：[1,2]
//
// 示例 2：输入：A = [1,2], B = [2,3] 输出：[1,2]
//
// 示例 3：输入：A = [2], B = [1,3] 输出：[2,3]
//
// 示例 4：输入：A = [1,2,5], B = [2,4] 输出：[5,4]
//
// 提示：
// 1 <= A.length <= 10000
// 1 <= B.length <= 10000
// 1 <= A[i] <= 100000
// 1 <= B[i] <= 100000
// 保证爱丽丝与鲍勃的糖果总量不同。
// 答案肯定存在。
func fairCandySwap(A []int, B []int) []int {
	var result = make([]int, 2)
	sum1, sum2 := 0, 0

	for _, a := range A {
		sum1 += a

	}
	var bMap = make(map[int]bool)
	for _, b := range B {
		sum2 += b
		bMap[b] = true
	}
	swapVal := (sum2 - sum1) >> 1
	for _, a := range A {
		if bMap[a+swapVal] {
			result[0] = a
			result[1] = a + swapVal
			break
		}

	}
	return result
}

// 766. 托普利茨矩阵
// 给你一个 m x n 的矩阵 matrix 。如果这个矩阵是托普利茨矩阵，返回 true ；否则，返回 false 。
//
// 如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是 托普利茨矩阵 。
//
// 示例 1：
// 输入：matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]] 输出：true
// 解释：
// 在上述矩阵中, 其对角线为:
// "[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。
// 各条对角线上的所有元素均相同, 因此答案是 True 。
//
// 示例 2：
// 输入：matrix = [[1,2],[2,2]] 输出：false
// 解释：
// 对角线 "[1, 2]" 上的元素不同。
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 20
// 0 <= matrix[i][j] <= 99
//
// 进阶：
// 如果矩阵存储在磁盘上，并且内存有限，以至于一次最多只能将矩阵的一行加载到内存中，该怎么办？
// 如果矩阵太大，以至于一次只能将不完整的一行加载到内存中，该怎么办？
func isToeplitzMatrix(matrix [][]int) bool {
	m, n := len(matrix), len(matrix[0])
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][j] != matrix[i-1][j-1] {
				return false
			}
		}
	}

	return true
}

// 832. 翻转图像
// 给定一个二进制矩阵 A，我们想先水平翻转图像，然后反转图像并返回结果。
// 水平翻转图片就是将图片的每一行都进行翻转，即逆序。例如，水平翻转 [1, 1, 0] 的结果是 [0, 1, 1]。
// 反转图片的意思是图片中的 0 全部被 1 替换， 1 全部被 0 替换。例如，反转 [0, 1, 1] 的结果是 [1, 0, 0]。
//
// 示例 1:
// 输入: [[1,1,0],[1,0,1],[0,0,0]] 输出: [[1,0,0],[0,1,0],[1,1,1]]
// 解释: 首先翻转每一行: [[0,1,1],[1,0,1],[0,0,0]]；
//     然后反转图片: [[1,0,0],[0,1,0],[1,1,1]]
//
// 示例 2:
// 输入: [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]] 输出: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
// 解释: 首先翻转每一行: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]]；
//     然后反转图片: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
//
// 说明:
// 1 <= A.length = A[0].length <= 20
// 0 <= A[i][j] <= 1
func flipAndInvertImage(A [][]int) [][]int {

	for i, _ := range A {
		nums, size := A[i], len(A[i])
		left, right := 0, size-1
		for left < right {
			nums[left], nums[right] = 1-nums[right], 1-nums[left]
			left++
			right--
		}
		if size&1 == 1 {
			nums[left] = 1 - nums[left]
		}
	}
	return A
}

// 867. 转置矩阵
// 给你一个二维整数数组 matrix， 返回 matrix 的 转置矩阵 。
//
// 矩阵的 转置 是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。
//
// 示例 1：
// 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]] 输出：[[1,4,7],[2,5,8],[3,6,9]]
//
// 示例 2：
// 输入：matrix = [[1,2,3],[4,5,6]] 输出：[[1,4],[2,5],[3,6]]
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 1000
// 1 <= m * n <= 105
// -109 <= matrix[i][j] <= 109
func transpose(matrix [][]int) [][]int {
	m, n := len(matrix), len(matrix[0])
	result := make([][]int, n)
	for i := 0; i < n; i++ {
		result[i] = make([]int, m)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[j][i] = matrix[i][j]
		}
	}

	return result
}

// 4. 寻找两个正序数组的中位数
// 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
//
// 示例 1：
// 输入：nums1 = [1,3], nums2 = [2] 输出：2.00000
// 解释：合并数组 = [1,2,3] ，中位数 2
//
// 示例 2：
// 输入：nums1 = [1,2], nums2 = [3,4] 输出：2.50000
// 解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
//
// 示例 3：
// 输入：nums1 = [0,0], nums2 = [0,0] 输出：0.00000
//
// 示例 4：
// 输入：nums1 = [], nums2 = [1] 输出：1.00000
//
// 示例 5：
// 输入：nums1 = [2], nums2 = [] 输出：2.00000
//
// 提示：
// nums1.length == m
// nums2.length == n
// 0 <= m <= 1000
// 0 <= n <= 1000
// 1 <= m + n <= 2000
// -106 <= nums1[i], nums2[i] <= 106
//
// 进阶：你能设计一个时间复杂度为 O(log (m+n)) 的算法解决此问题吗？
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	if m > n {
		return findMedianSortedArrays(nums2, nums1)
	}
	left, right := 0, m
	midIdx := (m + n) >> 1
	mid1, mid2 := 0, 0
	// 把数组划分为 左右两部分
	for left <= right {
		// 左边 nums1 [0, i - 1] nums2[0,j - 1]
		// 右边 nums1 [i, m - 1] nums2 [j, n - 1]
		i := (left + right) / 2
		j := midIdx - i
		num10, num11 := math.MinInt32, math.MaxInt32
		if i > 0 {
			num10 = nums1[i-1]
		}
		if i < m {
			num11 = nums1[i]
		}
		num20, num21 := math.MinInt32, math.MaxInt32
		if j > 0 {
			num20 = nums2[j-1]
		}
		if j < n {
			num21 = nums2[j]
		}
		if num10 <= num21 {
			mid1 = max(num10, num20)
			mid2 = min(num11, num21)
			left = i + 1
		} else {
			right = i - 1
		}

	}
	fmt.Printf("mid1 :%d mid2:%d\n", mid1, mid2)
	if (m+n)&1 == 0 {
		return float64(mid1+mid2) / 2.0
	}

	return float64(mid2)
}

func max(nums ...int) int {
	val := nums[0]
	for _, num := range nums {
		if num > val {
			val = num
		}
	}
	return val
}
func min(nums ...int) int {
	val := nums[0]
	for _, num := range nums {
		if num < val {
			val = num
		}
	}
	return val
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

// 11. 盛最多水的容器
// 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
//
// 说明：你不能倾斜容器。
//
// 示例 1：
// 输入：[1,8,6,2,5,4,8,3,7] 输出：49
// 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
//
// 示例 2：
// 输入：height = [1,1] 输出：1
//
// 示例 3：
// 输入：height = [4,3,2,1,4] 输出：16
//
// 示例 4：
// 输入：height = [1,2,1] 输出：2
//
// 提示：
// n = height.length
// 2 <= n <= 3 * 104
// 0 <= height[i] <= 3 * 104
func maxArea(height []int) int {
	// 双指针
	left, right := 0, len(height)-1
	result := 0
	for left < right {
		l := right - left
		var h int
		if height[left] <= height[right] {
			h = height[left]
			left++
		} else {
			h = height[right]
			right--
		}
		result = max(result, h*l)
	}
	return result
}

// 15. 三数之和
// 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
//
// 注意：答案中不可以包含重复的三元组。
//
// 示例 1：
// 输入：nums = [-1,0,1,2,-1,-4] 输出：[[-1,-1,2],[-1,0,1]]
//
// 示例 2：
// 输入：nums = [] 输出：[]
//
// 示例 3：
// 输入：nums = [0] 输出：[]
//
// 提示：
// 0 <= nums.length <= 3000
// -105 <= nums[i] <= 105
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	size := len(nums)
	var result = make([][]int, 0)
	// 选定一个主元
	for i := 0; i < size-2; i++ {
		if nums[i] > 0 {
			break
		}
		// 最小的元素 > 0 sum肯定大于0
		if nums[i] > 0 {
			break
		}
		// 如果 num 和前一位相等，跳过
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		left, right := i+1, size-1
		for left < right {
			// 最大元素  < 0 sum肯定小于0
			if nums[right] < 0 {
				break
			}
			sum := nums[i] + nums[left] + nums[right]
			if sum == 0 {
				result = append(result, []int{nums[i], nums[left], nums[right]})
				left++
				right--
				for left < right && nums[left] == nums[left-1] {
					left++
				}
				for left < right && nums[right] == nums[right+1] {
					right--
				}
			} else if sum < 0 {
				left++
			} else {
				right--
			}

		}
	}

	return result
}

// LCP 07. 传递信息
// 小朋友 A 在和 ta 的小伙伴们玩传信息游戏，游戏规则如下：
//
// 有 n 名玩家，所有玩家编号分别为 0 ～ n-1，其中小朋友 A 的编号为 0
// 每个玩家都有固定的若干个可传信息的其他玩家（也可能没有）。传信息的关系是单向的（比如 A 可以向 B 传信息，但 B 不能向 A 传信息）。
// 每轮信息必须需要传递给另一个人，且信息可重复经过同一个人
// 给定总玩家数 n，以及按 [玩家编号,对应可传递玩家编号] 关系组成的二维数组 relation。返回信息从小 A (编号 0 ) 经过 k 轮传递到编号为 n-1 的小伙伴处的方案数；若不能到达，返回 0。
//
// 示例 1：
// 输入：n = 5, relation = [[0,2],[2,1],[3,4],[2,3],[1,4],[2,0],[0,4]], k = 3
// 输出：3
// 解释：信息从小 A 编号 0 处开始，经 3 轮传递，到达编号 4。共有 3 种方案，分别是 0->2->0->4， 0->2->1->4， 0->2->3->4。
//
// 示例 2：
// 输入：n = 3, relation = [[0,2],[2,1]], k = 2 输出：0
// 解释：信息不能从小 A 处经过 2 轮传递到编号 2
//
// 限制：
// 2 <= n <= 10
// 1 <= k <= 5
// 1 <= relation.length <= 90, 且 relation[i].length == 2
// 0 <= relation[i][0],relation[i][1] < n 且 relation[i][0] != relation[i][1]
func numWays(n int, relation [][]int, k int) int {

	// 深度优先遍历
	graph := make([][]bool, n)
	for i := 0; i < n; i++ {
		graph[i] = make([]bool, n)
	}
	for _, rel := range relation {
		graph[rel[0]][rel[1]] = true
	}
	result := 0

	var dfs func(num, count int)

	dfs = func(num, count int) {
		if count == 0 {
			if num == n-1 {
				result++
			}
			return
		}

		for i := 0; i < n; i++ {
			if i == num {
				continue
			}
			if graph[num][i] {
				dfs(i, count-1)
			}
		}
	}

	dfs(0, k)
	return result
}

// 16. 最接近的三数之和
// 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。
// 返回这三个数的和。假定每组输入只存在唯一答案。
//
// 示例：
// 输入：nums = [-1,2,1,-4], target = 1 输出：2
// 解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
//
// 提示：
// 3 <= nums.length <= 10^3
// -10^3 <= nums[i] <= 10^3
// -10^4 <= target <= 10^4
func threeSumClosest(nums []int, target int) int {
	sort.Ints(nums)
	size := len(nums)

	sub, result := math.MaxInt32, 0

	// 选定一个主元
	for i := 0; i < size-2; i++ {

		left, right := i+1, size-1
		for left < right {
			sum := nums[i] + nums[left] + nums[right]
			if sum == target {
				return target
			} else if sum < target {
				if target-sum < sub {
					result = sum
					sub = target - sum
				}
				left++
			} else {
				if sum-target < sub {
					result = sum
					sub = sum - target
				}
				right--
			}

		}
	}
	return result
}

// 18. 四数之和
// 给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
//
// 注意：答案中不可以包含重复的四元组。
//
// 示例 1：
// 输入：nums = [1,0,-1,0,-2,2], target = 0 输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
//
// 示例 2：
// 输入：nums = [], target = 0 输出：[]
//
// 提示：
// 0 <= nums.length <= 200
// -109 <= nums[i] <= 109
// -109 <= target <= 109
func fourSum(nums []int, target int) [][]int {

	result := make([][]int, 0)
	sort.Ints(nums)
	n := len(nums)

	// 选定两个个主元
	for i := 0; i < n-3; i++ {
		// 如果 num 和前一位相等，跳过
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		// 最小值 大于 target
		if nums[i]+nums[i+1]+nums[i+2]+nums[i+3] > target {
			break
		}
		if nums[i]+nums[n-1]+nums[n-2]+nums[n-3] < target {
			continue
		}
		for j := i + 1; j < n-2; j++ {

			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			if nums[i]+nums[j]+nums[j+1]+nums[j+2] > target {
				break
			}
			if nums[i]+nums[j]+nums[n-1]+nums[n-2] < target {
				continue
			}

			left, right := j+1, n-1
			for left < right {
				sum := nums[i] + nums[j] + nums[left] + nums[right]
				if sum == target {
					result = append(result, []int{nums[i], nums[j], nums[left], nums[right]})
					left++
					right--
					for left < right && nums[left] == nums[left-1] {
						left++
					}
					for left < right && nums[right] == nums[right+1] {
						right--
					}
				} else if sum < target {
					left++
				} else {
					right--
				}

			}
		}
	}

	return result
}

// 31. 下一个排列
// 实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
// 如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
// 必须 原地 修改，只允许使用额外常数空间。
//
// 示例 1：
// 输入：nums = [1,2,3] 输出：[1,3,2]
//
// 示例 2：
// 输入：nums = [3,2,1] 输出：[1,2,3]
//
// 示例 3：
// 输入：nums = [1,1,5] 输出：[1,5,1]
//
// 示例 4：
// 输入：nums = [1] 输出：[1]
//
// 提示：
// 1 <= nums.length <= 100
// 0 <= nums[i] <= 100
func nextPermutation(nums []int) {
	size := len(nums)
	if size == 1 {
		return
	}
	if size == 2 {
		nums[0], nums[1] = nums[1], nums[0]
		return
	}
	// 最后两个元素升序 直接交换
	if nums[size-1] > nums[size-2] {
		nums[size-1], nums[size-2] = nums[size-2], nums[size-1]
		return
	}

	// 从后往前 找到第一个逆序排列
	idx := size - 1
	for idx > 0 {
		if nums[idx-1] < nums[idx] {
			break
		}
		idx--
	}
	// 把逆序改正序
	left, right := idx, size-1
	for left < right {
		nums[left], nums[right] = nums[right], nums[left]
		left++
		right--
	}

	if idx == 0 {
		return
	}
	//  12453 idx = 3 -> 12435 -> 12534
	//  12354 idx = 3 -> 12345 -> 12435
	// 从 idx 找到第一个大于nums[idx - 1] 的元素 交换
	lIdx, num := idx-1, nums[idx-1]
	for idx < size {
		if nums[idx] > num {
			break
		}
		idx++
	}
	// 交换
	nums[lIdx], nums[idx] = nums[idx], nums[lIdx]
}

// 41. 缺失的第一个正数
// 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
//
// 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
// 示例 1：
// 输入：nums = [1,2,0] 输出：3
//
// 示例 2：
// 输入：nums = [3,4,-1,1] 输出：2
//
// 示例 3：
// 输入：nums = [7,8,9,11,12] 输出：1
//
// 提示：
// 1 <= nums.length <= 5 * 105
// -231 <= nums[i] <= 231 - 1
func firstMissingPositive(nums []int) int {
	// 把 1 放到 nums[0] 的 位置上  2 放到 nums[1]的位置
	size := len(nums)

	var modify func(idx int)
	modify = func(num int) {
		if num <= 0 || num > size {
			return
		}
		if nums[num-1] == num {
			return
		}
		// 记录 num - 1 位置的 元素
		tmp := nums[num-1]
		nums[num-1] = num
		// 把 tmp 放到值得位置
		modify(tmp)
	}

	for i := 0; i < size; i++ {
		if i+1 == nums[i] {
			continue
		}
		modify(nums[i])
	}

	for i := 0; i < size; i++ {
		if i+1 != nums[i] {
			return i + 1
		}
	}

	return size + 1
}

// 42. 接雨水
// 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
//
// 示例 1：
// 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1] 输出：6
// 解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
//
// 示例 2：
// 输入：height = [4,2,0,3,2,5] 输出：9
//
// 提示：
// n == height.length
// 0 <= n <= 3 * 104
// 0 <= height[i] <= 105
func trap(height []int) int {
	n := len(height)
	if n <= 2 {
		return 0
	}
	leftHeight, rightHeight := make([]int, n), make([]int, n)
	leftHeight[0] = height[0]

	for i := 1; i < n; i++ {
		leftHeight[i] = max(leftHeight[i-1], height[i])
	}

	rightHeight[n-1] = height[n-1]
	for i := n - 2; i >= 0; i-- {
		rightHeight[i] = max(rightHeight[i+1], height[i])
	}
	result := 0
	for i := 1; i < n-1; i++ {
		result += min(leftHeight[i], rightHeight[i]) - height[i]
	}
	return result
}

// 48. 旋转图像
// 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
//
// 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
//
// 示例 1：
// 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]] 输出：[[7,4,1],[8,5,2],[9,6,3]]
//
// 示例 2：
// 输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
// 输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
//
// 示例 3：
// 输入：matrix = [[1]] 输出：[[1]]
//
// 示例 4：
// 输入：matrix = [[1,2],[3,4]] 输出：[[3,1],[4,2]]
//
// 提示：
// matrix.length == n
// matrix[i].length == n
// 1 <= n <= 20
// -1000 <= matrix[i][j] <= 1000
func rotate(matrix [][]int) {
	n := len(matrix)
	if n == 1 {
		return
	}

	for i := 0; i < n>>1; i++ {
		for j := i; j < n-i-1; j++ {
			// 旋转4次
			tmp := matrix[i][j]

			matrix[i][j] = matrix[n-j-1][i]
			matrix[n-j-1][i] = matrix[n-i-1][n-j-1]
			matrix[n-i-1][n-j-1] = matrix[j][n-i-1]
			matrix[j][n-i-1] = tmp
		}
	}
}

// 面试题 17.10. 主要元素
// 数组中占比超过一半的元素称之为主要元素。给你一个 整数 数组，找出其中的主要元素。若没有，返回 -1 。请设计时间复杂度为 O(N) 、空间复杂度为 O(1) 的解决方案。
//
// 示例 1：
// 输入：[1,2,5,9,5,9,5,5,5] 输出：5
//
// 示例 2：
// 输入：[3,2] 输出：-1
//
// 示例 3：
// 输入：[2,2,1,1,1,2,2] 输出：2
func majorityElement2(nums []int) int {
	//摩尔投票算法
	count, n := 0, len(nums)
	var result int
	for _, num := range nums {
		if count == 0 {
			result = num
		}
		if result == num {
			count++
		} else {
			count--
		}
	}
	if count <= 0 {
		return -1
	}
	count = 0
	for _, num := range nums {
		if num == result {
			count++
		}
	}
	half := (n >> 1) + 1
	if count < half {
		return -1
	}
	return result
}

// 54. 螺旋矩阵
// 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
//
// 示例 1：
// 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]] 输出：[1,2,3,6,9,8,7,4,5]
//
// 示例 2：
// 输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] 输出：[1,2,3,4,8,12,11,10,9,5,6,7]
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 10
// -100 <= matrix[i][j] <= 100
func spiralOrder(matrix [][]int) []int {
	m, n := len(matrix), len(matrix[0])
	result := make([]int, m*n)
	// move 方向 0 右 1 下  2 左 3 上
	move, i, j := 0, 0, 0
	startRow, endRow, startCol, endCol := 1, m-1, 0, n-1
	if n == 1 {
		move = 1
	}
	for idx := 0; idx < m*n; idx++ {
		move = move % 4
		result[idx] = matrix[i][j]
		switch move {
		case 0:
			{

				j++
				if j == endCol {
					endCol--
					move++
				}
			}
		case 1:
			{
				i++
				if i == endRow {
					endRow--
					move++
				}
			}
		case 2:
			{

				j--
				if j == startCol {
					startCol++
					move++
				}
			}
		case 3:
			{
				i--
				if i == startRow {
					startRow++
					move++
				}
			}
		}

	}

	return result
}

// 56. 合并区间
// 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
// 请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
//
// 示例 1：
// 输入：intervals = [[1,3],[2,6],[8,10],[15,18]] 输出：[[1,6],[8,10],[15,18]]
// 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
//
// 示例 2：
// 输入：intervals = [[1,4],[4,5]] 输出：[[1,5]]
// 解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
//
// 提示：
// 1 <= intervals.length <= 104
// intervals[i].length == 2
// 0 <= starti <= endi <= 104
func mergeII(intervals [][]int) [][]int {
	result := make([][]int, 0)
	sort.Slice(intervals, func(i, j int) bool {
		a, b := intervals[i][0], intervals[j][0]
		return a < b
	})
	fmt.Println(intervals)

	for i, _ := range intervals {
		nums := intervals[i]
		n := len(result)
		if n == 0 {
			result = append(result, nums)
		} else {
			lastNums := result[n-1]
			// 判断是否包含
			if nums[0] > lastNums[1] {
				// 不包含
				result = append(result, nums)
			} else if lastNums[1] < nums[1] {
				// 部分包含
				lastNums[1] = nums[1]
			}
			// 整个包含直接跳过

		}
	}

	return result
}

// 57. 插入区间
// 给你一个 无重叠的 ，按照区间起始端点排序的区间列表。
//
// 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
//
// 示例 1：
// 输入：intervals = [[1,3],[6,9]], newInterval = [2,5] 输出：[[1,5],[6,9]]
//
// 示例 2：
// 输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
// 输出：[[1,2],[3,10],[12,16]]
// 解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
//
// 示例 3：
// 输入：intervals = [], newInterval = [5,7] 输出：[[5,7]]
//
// 示例 4：
// 输入：intervals = [[1,5]], newInterval = [2,3] 输出：[[1,5]]
//
// 示例 5：
// 输入：intervals = [[1,5]], newInterval = [2,7]
// 输出：[[1,7]]
//
// 提示：
// 0 <= intervals.length <= 104
// intervals[i].length == 2
// 0 <= intervals[i][0] <= intervals[i][1] <= 105
// intervals 根据 intervals[i][0] 按 升序 排列
// newInterval.length == 2
// 0 <= newInterval[0] <= newInterval[1] <= 105
func insert(intervals [][]int, newInterval []int) [][]int {
	result := make([][]int, 0)
	i, n := 0, len(intervals)

	// 先插入 所有比 newInterval[0] 小 的 interval
	for i < n && intervals[i][1] < newInterval[0] {
		result = append(result, intervals[i])
		i++
	}
	// 所有 和 newInterval 重合的 interval
	for i < n && intervals[i][0] <= newInterval[1] {
		newInterval[0] = min(intervals[i][0], newInterval[0])
		newInterval[1] = max(intervals[i][1], newInterval[1])
		i++
	}
	result = append(result, newInterval)

	for i < n {
		result = append(result, intervals[i])
		i++
	}
	return result
}

// 59. 螺旋矩阵 II
// 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
// 示例 1：
// 输入：n = 3 输出：[[1,2,3],[8,9,4],[7,6,5]]
//
// 示例 2：
// 输入：n = 1 输出：[[1]]
//
// 提示：
// 1 <= n <= 20
func generateMatrix(n int) [][]int {
	matrix := make([][]int, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n)
	}
	if n == 1 {
		matrix[0][0] = 1
		return matrix
	}
	// move 方向 0 右 1 下  2 左 3 上
	move, i, j := 0, 0, 0
	startRow, endRow, startCol, endCol := 1, n-1, 0, n-1

	for num := 1; num <= n*n; num++ {
		move = move % 4
		matrix[i][j] = num
		switch move {
		case 0:
			{

				j++
				if j == endCol {
					endCol--
					move++
				}
			}
		case 1:
			{
				i++
				if i == endRow {
					endRow--
					move++
				}
			}
		case 2:
			{

				j--
				if j == startCol {
					startCol++
					move++
				}
			}
		case 3:
			{
				i--
				if i == startRow {
					startRow++
					move++
				}
			}
		}
	}

	return matrix
}

// 73. 矩阵置零
// 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
//
// 进阶：
// 一个直观的解决方案是使用  O(mn) 的额外空间，但这并不是一个好的解决方案。
// 一个简单的改进方案是使用 O(m + n) 的额外空间，但这仍然不是最好的解决方案。
// 你能想出一个仅使用常量空间的解决方案吗？
//
// 示例 1：
// 输入：matrix = [[1,1,1],[1,0,1],[1,1,1]] 输出：[[1,0,1],[0,0,0],[1,0,1]]
//
// 示例 2：
// 输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]] 输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
//
// 提示：
// m == matrix.length
// n == matrix[0].length
// 1 <= m, n <= 200
// -231 <= matrix[i][j] <= 231 - 1
func setZeroes(matrix [][]int) {
	m, n := len(matrix), len(matrix[0])
	zeroRow, zeroCol := make([]bool, m), make([]bool, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if matrix[i][j] == 0 {
				zeroRow[i] = true
				zeroCol[j] = true
			}
		}
	}
	for i := 0; i < m; i++ {
		if zeroRow[i] {
			for j := 0; j < n; j++ {
				matrix[i][j] = 0
			}
		}
	}
	for j := 0; j < n; j++ {
		if zeroCol[j] {
			for i := 0; i < m; i++ {
				matrix[i][j] = 0
			}
		}
	}

}

// 75. 颜色分类
// 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
//
// 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
//
// 示例 1：
// 输入：nums = [2,0,2,1,1,0] 输出：[0,0,1,1,2,2]
//
// 示例 2：
// 输入：nums = [2,0,1] 输出：[0,1,2]
//
// 示例 3：
// 输入：nums = [0] 输出：[0]
//
// 示例 4：
// 输入：nums = [1] 输出：[1]
//
// 提示：
// n == nums.length
// 1 <= n <= 300
// nums[i] 为 0、1 或 2
func sortColors(nums []int) {
	red, white, blue := 0, 0, 0
	for _, num := range nums {
		switch num {
		case 0:
			red++
		case 1:
			white++
		case 2:
			blue++
		}
	}
	idx := 0
	for red > 0 {
		nums[idx] = 0
		red--
		idx++
	}
	for white > 0 {
		nums[idx] = 1
		white--
		idx++
	}
	for blue > 0 {
		nums[idx] = 2
		blue--
		idx++
	}

}

// 80. 删除有序数组中的重复项 II
// 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。
//
// 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
//
// 说明：
// 为什么返回数值是整数，但输出的答案是数组呢？
// 请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
// 你可以想象内部操作如下:
// // nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
// int len = removeDuplicates(nums);
// // 在函数里修改输入数组对于调用者是可见的。
// // 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
// for (int i = 0; i < len; i++) {
//    print(nums[i]);
// }
//
// 示例 1：
// 输入：nums = [1,1,1,2,2,3]
// 输出：5, nums = [1,1,2,2,3]
// 解释：函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。 不需要考虑数组中超出新长度后面的元素。
//
// 示例 2：
// 输入：nums = [0,0,1,1,1,1,2,3,3]
// 输出：7, nums = [0,0,1,1,2,3,3]
// 解释：函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。 不需要考虑数组中超出新长度后面的元素。
//
// 提示：
// 1 <= nums.length <= 3 * 104
// -104 <= nums[i] <= 104
// nums 已按升序排列
func removeDuplicatesII(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	idx := 0
	for i, num := range nums {
		if i == 0 {
			continue
		}
		if i == 1 {
			idx++
			continue
		}
		if num != nums[idx] || num != nums[idx-1] {
			idx++
			nums[idx] = num
		} else if num != nums[idx-1] {
			idx++
			nums[idx] = num
		}
	}

	return idx + 1
}

// 85. 最大矩形
// 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
//
// 示例 1：
// 输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
// 输出：6
// 解释：最大矩形如上图所示。
//
// 示例 2：
// 输入：matrix = [] 输出：0
//
// 示例 3：
// 输入：matrix = [["0"]] 输出：0
//
// 示例 4：
// 输入：matrix = [["1"]] 输出：1
//
// 示例 5：
// 输入：matrix = [["0","0"]] 输出：0
//
// 提示：
// rows == matrix.length
// cols == matrix[0].length
// 0 <= row, cols <= 200
// matrix[i][j] 为 '0' 或 '1'
func maximalRectangle(matrix [][]byte) int {
	rows := len(matrix)
	if rows == 0 {
		return 0
	}
	cols := len(matrix[0])
	if cols == 0 {
		return 0
	}

	side := make([][]int, rows)
	for i := 0; i < rows; i++ {
		side[i] = make([]int, cols)
		size := 0
		for j := 0; j < cols; j++ {
			if matrix[i][j] == '1' {
				size++
			} else {
				size = 0
			}
			side[i][j] = size
		}
	}

	result := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			minSide := math.MaxInt32
			for k := i; k >= 0; k-- {
				if side[k][j] == 0 {
					break
				}
				minSide = min(minSide, side[k][j])
				result = max(result, minSide*(i-k+1))
			}
		}
	}
	return result
}

// 128. 最长连续序列
// 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
//
// 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
//
// 示例 1：
// 输入：nums = [100,4,200,1,3,2] 输出：4
// 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
//
// 示例 2：
// 输入：nums = [0,3,7,2,5,8,4,6,0,1] 输出：9
//
// 提示：
// 0 <= nums.length <= 105
// -109 <= nums[i] <= 109
func longestConsecutive(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	result := 1
	numMap := make(map[int]bool)
	for _, num := range nums {
		numMap[num] = true
	}
	for num := range numMap {
		if numMap[num-1] {
			continue
		}
		size := 1
		for numMap[num+1] {
			num++
			size++
		}
		result = max(result, size)
	}
	return result
}

// 130. 被围绕的区域
// 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
//
// 示例 1：
// 输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
// 输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
// 解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
//
// 示例 2：
// 输入：board = [["X"]] 输出：[["X"]]
//
// 提示：
// m == board.length
// n == board[i].length
// 1 <= m, n <= 200
// board[i][j] 为 'X' 或 'O'
func solve(board [][]byte) {
	m, n := len(board), len(board[0])

	var update func(i, j int)
	update = func(i, j int) {
		if !inArea(i, j, m, n) {
			return
		}
		if board[i][j] != 'O' {
			return
		}
		board[i][j] = '*'
		for k := 0; k < 4; k++ {
			update(i+DirRow[k], j+DirCol[k])
		}

	}
	// 从边界的 'O' 进行深度优先遍历

	// 第一行和最后一行
	for j := 0; j < n; j++ {
		if board[0][j] == 'O' {
			update(0, j)
		}
		if board[m-1][j] == 'O' {
			update(m-1, j)
		}
	}
	// 第一列和最后一列
	for i := 0; i < m; i++ {
		if board[i][0] == 'O' {
			update(i, 0)
		}
		if board[i][n-1] == 'O' {
			update(i, n-1)
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if board[i][j] == '*' {
				board[i][j] = 'O'
			} else {
				board[i][j] = 'X'
			}
		}
	}

}

var (
	DirCol = []int{1, -1, 0, 0}
	DirRow = []int{0, 0, 1, -1}
)

func inArea(row, col, rows, cols int) bool {
	return row >= 0 && row < rows && col >= 0 && col < cols
}

// 137. 只出现一次的数字 II
// 给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。
//
// 示例 1：
// 输入：nums = [2,2,3,2] 输出：3
//
// 示例 2：
// 输入：nums = [0,1,0,1,0,1,99] 输出：99
//
// 提示：
// 1 <= nums.length <= 3 * 104
// -231 <= nums[i] <= 231 - 1
// nums 中，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次
//
// 进阶：你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
func singleNumberII(nums []int) int {
	result := int32(0)

	for i := 0; i < 32; i++ {
		count := int32(0)
		for _, num := range nums {
			count += (int32(num) >> i) & 1
		}
		// 第i位的1出现3的整数倍 + 1
		if count%3 > 0 {
			result |= 1 << i
		}
	}
	return int(result)
}

func singleNumber2(nums []int) int {
	numMap := make(map[int]int)
	for _, num := range nums {
		numMap[num]++
	}
	for k, v := range numMap {
		if v == 1 {
			return k
		}
	}
	return -1
}

// 164. 最大间距
// 给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。
//
// 如果数组元素个数小于 2，则返回 0。
//
// 示例 1:
// 输入: [3,6,9,1] 输出: 3
// 解释: 排序后的数组是 [1,3,6,9], 其中相邻元素 (3,6) 和 (6,9) 之间都存在最大差值 3。
//
// 示例 2:
// 输入: [10] 输出: 0
// 解释: 数组元素个数小于 2，因此返回 0。
//
// 说明:
// 你可以假设数组中所有元素都是非负整数，且数值在 32 位有符号整数范围内。
// 请尝试在线性时间复杂度和空间复杂度的条件下解决此问题。
func maximumGap(nums []int) int {
	n := len(nums)
	if n <= 1 {
		return 0
	}
	sort.Ints(nums)
	result := 0
	for i := 1; i < n; i++ {
		result = max(result, nums[i]-nums[i-1])
	}

	return result
}

// 189. 旋转数组
// 给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
//
// 进阶：
// 尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
// 你可以使用空间复杂度为 O(1) 的 原地 算法解决这个问题吗？
//
// 示例 1:
// 输入: nums = [1,2,3,4,5,6,7], k = 3 输出: [5,6,7,1,2,3,4]
// 解释:
// 向右旋转 1 步: [7,1,2,3,4,5,6]
// 向右旋转 2 步: [6,7,1,2,3,4,5]
// 向右旋转 3 步: [5,6,7,1,2,3,4]
//
// 示例 2:
// 输入：nums = [-1,-100,3,99], k = 2
// 输出：[3,99,-1,-100]
// 解释:
// 向右旋转 1 步: [99,-1,-100,3]
// 向右旋转 2 步: [3,99,-1,-100]
//
// 提示：
// 1 <= nums.length <= 2 * 104
// -231 <= nums[i] <= 231 - 1
// 0 <= k <= 105
func rotateArray(nums []int, k int) {
	n := len(nums)
	k %= n
	if k == 0 {
		return
	}
	reverse := func(left, right int) {
		for left < right {
			nums[left], nums[right] = nums[right], nums[left]
			left++
			right--
		}
	}
	reverse(0, n-1)
	reverse(0, k-1)
	reverse(k, n-1)
}

// 200. 岛屿数量
// 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
//
// 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
//
// 此外，你可以假设该网格的四条边均被水包围。
//
// 示例 1：
// 输入：grid = [
//  ["1","1","1","1","0"],
//  ["1","1","0","1","0"],
//  ["1","1","0","0","0"],
//  ["0","0","0","0","0"]
// ]
// 输出：1
//
// 示例 2：
// 输入：grid = [
//  ["1","1","0","0","0"],
//  ["1","1","0","0","0"],
//  ["0","0","1","0","0"],
//  ["0","0","0","1","1"]
// ]
// 输出：3
//
// 提示：
// m == grid.length
// n == grid[i].length
// 1 <= m, n <= 300
// grid[i][j] 的值为 '0' 或 '1'
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	result := 0

	var dfs func(row, col int)

	dfs = func(row, col int) {
		if grid[row][col] != '1' {
			return
		}
		grid[row][col] = '*'
		for k := 0; k < 4; k++ {
			nextRow, nextCol := row+DirRow[k], col+DirCol[k]
			if inArea(nextRow, nextCol, m, n) {
				dfs(nextRow, nextCol)
			}
		}
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == '1' {
				result++
				dfs(i, j)
			}
		}
	}

	return result
}

// 209. 长度最小的子数组
// 给定一个含有 n 个正整数的数组和一个正整数 target 。
//
// 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
//
// 示例 1：
// 输入：target = 7, nums = [2,3,1,2,4,3]
// 输出：2
// 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
//
// 示例 2：
// 输入：target = 4, nums = [1,4,4]
// 输出：1
//
// 示例 3：
// 输入：target = 11, nums = [1,1,1,1,1,1,1,1]
// 输出：0
//
// 提示：
// 1 <= target <= 109
// 1 <= nums.length <= 105
// 1 <= nums[i] <= 105
//
// 进阶：
// 如果你已经实现 O(n) 时间复杂度的解法, 请尝试设计一个 O(n log(n)) 时间复杂度的解法。
func minSubArrayLen(target int, nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	result := n + 1
	sums := make([]int, n+1)
	for i := 0; i < n; i++ {
		sums[i+1] = sums[i] + nums[i]
	}

	for i := 1; i <= n; i++ {
		if sums[i] < target {
			continue
		}
		// 二分查找
		idx := getIndex(sums, 0, i, sums[i]-target)
		fmt.Printf("left:%d, right:%d => sum:%d", idx, i, sums[i]-sums[idx])
		fmt.Println()
		result = min(result, i-idx+1)
	}
	if result > n {
		return 0
	}
	return result
}
func getIndex(nums []int, left, right, target int) int {
	// 二分查找
	for left < right {
		mid := (left + right) >> 1
		if target >= nums[mid] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

// 215. 数组中的第K个最大元素
// 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
//
// 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
//
// 示例 1:
// 输入: [3,2,1,5,6,4] 和 k = 2
// 输出: 5
//
// 示例 2:
// 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
// 输出: 4
//
// 提示：
// 1 <= k <= nums.length <= 104
// -104 <= nums[i] <= 104
func findKthLargest(nums []int, k int) int {
	// 快速排序 选择一个主元x 小于x放到左边 大于x放到右边 比

	var findKth func(start, end int, idx int)

	findKth = func(start, end int, idx int) {
		if start >= end {
			return
		}
		tmp := nums[start]
		left, right := start, end
		// 比tmp大的放到
		for left < right {
			// 从右边找 第一个比 tmp小的元素
			for left < right && nums[right] >= tmp {
				right--
			}
			nums[left] = nums[right]
			// 从左边找 第一个比 tmp大的元素
			for left < right && nums[left] <= tmp {
				left++
			}
			nums[right] = nums[left]
		}
		nums[left] = tmp
		if idx == left {
			return
		}
		if idx < left {
			findKth(start, left-1, idx)
		} else {
			findKth(left+1, end, idx)
		}
	}
	n := len(nums)
	findKth(0, n-1, n-k)

	return nums[n-k]
}

// 228. 汇总区间
// 给定一个无重复元素的有序整数数组 nums 。
//
// 返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。
// 列表中的每个区间范围 [a,b] 应该按如下格式输出：
// "a->b" ，如果 a != b
// "a" ，如果 a == b
//
// 示例 1：
// 输入：nums = [0,1,2,4,5,7]
// 输出：["0->2","4->5","7"]
// 解释：区间范围是：
// [0,2] --> "0->2"
// [4,5] --> "4->5"
// [7,7] --> "7"
//
// 示例 2：
// 输入：nums = [0,2,3,4,6,8,9]
// 输出：["0","2->4","6","8->9"]
// 解释：区间范围是：
// [0,0] --> "0"
// [2,4] --> "2->4"
// [6,6] --> "6"
// [8,9] --> "8->9"
//
// 示例 3：
// 输入：nums = []
// 输出：[]
//
// 示例 4：
// 输入：nums = [-1]
// 输出：["-1"]
//
// 示例 5：
// 输入：nums = [0]
// 输出：["0"]
//
// 提示：
// 0 <= nums.length <= 20
// -231 <= nums[i] <= 231 - 1
// nums 中的所有值都 互不相同
// nums 按升序排列
func summaryRanges(nums []int) []string {
	result := make([]string, 0)
	n := len(nums)
	for i := 0; i < n; i++ {
		start, end := nums[i], nums[i]
		for i+1 < n && nums[i+1] == nums[i]+1 {
			i++
			end = nums[i]
		}
		str := ""
		if start == end {
			str += strconv.Itoa(start)
		} else {
			str += strconv.Itoa(start) + "->" + strconv.Itoa(end)
		}
		result = append(result, str)
	}

	return result
}

// 229. 求众数 II
// 给定一个大小为 n 的整数数组，找出其中所有出现超过 ⌊ n/3 ⌋ 次的元素。
//
// 进阶：尝试设计时间复杂度为 O(n)、空间复杂度为 O(1)的算法解决此问题。
//
// 示例 1：
// 输入：[3,2,3]
// 输出：[3]
//
// 示例 2：
// 输入：nums = [1]
// 输出：[1]
//
// 示例 3：
// 输入：[1,1,1,3,3,2,2,2]
// 输出：[1,2]
//
// 提示：
// 1 <= nums.length <= 5 * 104
// -109 <= nums[i] <= 109
func majorityElementII(nums []int) []int {
	n := len(nums)
	result := make([]int, 0)
	// 摩尔投票法
	count1, count2 := 0, 0
	num1, num2 := nums[0], nums[0]
	for _, num := range nums {
		if num == num1 {
			count1++
			continue
		}
		if num == num2 {
			count2++
			continue
		}
		if count1 == 0 {
			num1 = num
			count1++
			continue
		}
		if count2 == 0 {
			num2 = num
			count2++
			continue
		}
		count1--
		count2--
	}
	count1, count2 = 0, 0
	for _, num := range nums {
		if num == num1 {
			count1++
		} else if num == num2 {
			count2++
		}
	}
	if count1 > n/3 {
		result = append(result, num1)
	}
	if count2 > n/3 {
		result = append(result, num2)
	}

	return result
}

// 238. 除自身以外数组的乘积
// 给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
//
// 示例:
// 输入: [1,2,3,4]
// 输出: [24,12,8,6]
//
// 提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。
// 说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
// 进阶：
// 你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
func productExceptSelf(nums []int) []int {
	n := len(nums)
	result := make([]int, n)

	// 左边元素的乘积
	num := 1

	for i := 0; i < n; i++ {
		result[i] = num
		num *= nums[i]
	}
	num = 1
	for i := n - 1; i >= 0; i-- {
		result[i] *= num
		num *= nums[i]
	}

	return result
}

// 239. 滑动窗口最大值
// 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
//
// 返回滑动窗口中的最大值。
//
// 示例 1：
// 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
// 输出：[3,3,5,5,6,7]
// 解释：
// 滑动窗口的位置                最大值
// ---------------               -----
// [1  3  -1] -3  5  3  6  7       3
//  1 [3  -1  -3] 5  3  6  7       3
//  1  3 [-1  -3  5] 3  6  7       5
//  1  3  -1 [-3  5  3] 6  7       5
//  1  3  -1  -3 [5  3  6] 7       6
//  1  3  -1  -3  5 [3  6  7]      7
//
// 示例 2：
// 输入：nums = [1], k = 1
// 输出：[1]
//
// 示例 3：
// 输入：nums = [1,-1], k = 1
// 输出：[1,-1]
//
// 示例 4：
// 输入：nums = [9,11], k = 2
// 输出：[11]
//
// 示例 5：
// 输入：nums = [4,-2], k = 2
// 输出：[4]
//
// 提示：
// 1 <= nums.length <= 105
// -104 <= nums[i] <= 104
// 1 <= k <= nums.length
func maxSlidingWindow(nums []int, k int) []int {
	n := len(nums)
	if k == 1 {
		return nums
	}
	m := n - k + 1
	result := make([]int, m)
	// 双端队列
	deque := list.New()
	for i := 0; i < k; i++ {
		for deque.Len() > 0 {
			back := deque.Back()
			// 前面的 元素小 移除  当前最大 nums[i]
			if back.Value.(int) < nums[i] {
				deque.Remove(back)
			} else {
				break
			}
		}

		deque.PushBack(nums[i])
	}
	front := deque.Front()
	result[0] = front.Value.(int)

	for i := k; i < n; i++ {
		if deque.Len() > 0 {
			front = deque.Front()
			if nums[i-k] == front.Value.(int) {
				deque.Remove(front)
			}
		}
		for deque.Len() > 0 {
			back := deque.Back()
			// 前面的 元素小 移除  当前最大 nums[i]
			if back.Value.(int) < nums[i] {
				deque.Remove(back)
			} else {
				break
			}
		}
		deque.PushBack(nums[i])
		front = deque.Front()
		result[i-k+1] = front.Value.(int)
	}

	return result
}

// 260. 只出现一次的数字 III
// 给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 任意顺序 返回答案。
//
// 进阶：你的算法应该具有线性时间复杂度。你能否仅使用常数空间复杂度来实现？
//
// 示例 1：
// 输入：nums = [1,2,1,3,2,5]
// 输出：[3,5]
// 解释：[5, 3] 也是有效的答案。
//
// 示例 2：
// 输入：nums = [-1,0]
// 输出：[-1,0]
//
// 示例 3：
// 输入：nums = [0,1]
// 输出：[1,0]
//
// 提示：
// 2 <= nums.length <= 3 * 104
// -231 <= nums[i] <= 231 - 1
// 除两个只出现一次的整数外，nums 中的其他数字都出现两次
func singleNumberIII(nums []int) []int {
	numMap := make(map[int]int)
	for _, num := range nums {
		numMap[num]++
	}
	result := make([]int, 0)
	for k, v := range numMap {
		if v == 1 {
			result = append(result, k)
		}
	}
	return result
}

// 289. 生命游戏
// 根据 百度百科 ，生命游戏，简称为生命，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。
//
// 给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：
//
// 如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
// 如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
// 如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
// 如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
// 下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。给你 m x n 网格面板 board 的当前状态，返回下一个状态。
//
// 示例 1：
// 输入：board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
// 输出：[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
//
// 示例 2：
// 输入：board = [[1,1],[1,0]]
// 输出：[[1,1],[1,1]]
//
// 提示：
// m == board.length
// n == board[i].length
// 1 <= m, n <= 25
// board[i][j] 为 0 或 1
//
// 进阶：
// 你可以使用原地算法解决本题吗？请注意，面板上所有格子需要同时被更新：你不能先更新某些格子，然后使用它们的更新后的值再更新其他格子。
// 本题中，我们使用二维数组来表示面板。原则上，面板是无限的，但当活细胞侵占了面板边界时会造成问题。你将如何解决这些问题？
func gameOfLife(board [][]int) {
	m, n := len(board), len(board[0])

	// 使用中间状态实现原地元素
	// 活细胞死亡 1 -> -1 -> 0
	// 死细胞复活 0 -> 2 -> 1
	var updateCell func(row, col int)
	updateCell = func(row, col int) {
		live, dead := 0, 0
		for i := row - 1; i <= row+1; i++ {
			for j := col - 1; j <= col+1; j++ {
				if !inArea(i, j, m, n) {
					continue
				}
				if i == row && j == col {
					continue
				}
				// 活细胞
				if board[i][j] == 1 || board[i][j] == -1 {
					live++
				}
				// 死细胞
				if board[i][j] == 0 || board[i][j] == 2 {
					dead++
				}
			}
		}
		// 如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
		// 如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
		// 如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
		if board[row][col] == 1 {
			if live < 2 || live > 3 {
				// 活细胞死亡 1 -> -1 -> 0
				board[row][col] = -1
			}
		}
		// 如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
		if board[row][col] == 0 {
			if live == 3 {
				// 死细胞复活 0 -> 2 -> 1
				board[row][col] = 2
			}
		}
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			updateCell(i, j)
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if board[i][j] == -1 {
				board[i][j] = 0
			}
			if board[i][j] == 2 {
				board[i][j] = 1
			}
		}
	}
}

// 315. 计算右侧小于当前元素的个数
// 给你`一个整数数组 nums ，按要求返回一个新数组 counts 。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。
//
// 示例 1：
// 输入：nums = [5,2,6,1]
// 输出：[2,1,1,0]
// 解释：
// 5 的右侧有 2 个更小的元素 (2 和 1)
// 2 的右侧仅有 1 个更小的元素 (1)
// 6 的右侧有 1 个更小的元素 (1)
// 1 的右侧有 0 个更小的元素
//
// 示例 2：
// 输入：nums = [-1]
// 输出：[0]
//
// 示例 3：
// 输入：nums = [-1,-1]
// 输出：[0,0]
//
// 提示：
// 1 <= nums.length <= 105
// -104 <= nums[i] <= 104
func countSmaller(nums []int) []int {
	n := len(nums)

	tmpNums, counts := make([]int, n), make([]int, n)
	index, tmpIndex := make([]int, n), make([]int, n)

	for i := 0; i < n; i++ {
		index[i] = i
	}
	counts[n-1] = 0

	// 两个有序数组合并
	var merge = func(left, mid, right int) {
		i, j, idx := left, mid+1, left
		for i <= mid && j <= right {
			if nums[i] <= nums[j] {
				tmpNums[idx] = nums[i]
				tmpIndex[idx] = index[i]
				counts[index[i]] += j - mid - 1
				i++
			} else {
				tmpNums[idx] = nums[j]
				tmpIndex[idx] = index[j]
				j++
			}

			idx++
		}
		for i <= mid {
			tmpNums[idx] = nums[i]
			tmpIndex[idx] = index[i]
			counts[index[i]] += j - mid - 1
			i++
			idx++
		}
		for j <= right {
			tmpNums[idx] = nums[j]
			tmpIndex[idx] = index[j]
			j++
			idx++
		}
		for k := left; k <= right; k++ {
			index[k] = tmpIndex[k]
			nums[k] = tmpNums[k]
		}
	}

	// 归并排序
	var mergeSort func(left, right int)

	mergeSort = func(left, right int) {
		if left >= right {
			return
		}
		mid := (left + right) >> 1
		mergeSort(left, mid)
		mergeSort(mid+1, right)
		merge(left, mid, right)
	}
	mergeSort(0, n-1)
	return counts
}

// 321. 拼接最大数
// 给定长度分别为 m 和 n 的两个数组，其元素由 0-9 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 k (k <= m + n) 个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。
//
// 求满足该条件的最大数。结果返回一个表示该最大数的长度为 k 的数组。
// 说明: 请尽可能地优化你算法的时间和空间复杂度。
//
// 示例 1:
// 输入:
// nums1 = [3, 4, 6, 5]
// nums2 = [9, 1, 2, 5, 8, 3]
// k = 5
// 输出:
// [9, 8, 6, 5, 3]
//
// 示例 2:
// 输入:
// nums1 = [6, 7]
// nums2 = [6, 0, 4]
// k = 5
// 输出:
// [6, 7, 6, 0, 4]
//
// 示例 3:
// 输入:
// nums1 = [3, 9]
// nums2 = [8, 9]
// k = 3
// 输出:
// [9, 8, 9]
func maxNumber(nums1 []int, nums2 []int, k int) []int {
	m, n := len(nums1), len(nums2)
	result := make([]int, k)

	start, end := max(0, k-n), min(k, m)

	for i := start; i <= end; i++ {
		// nums1 取 i 个 ; nums2 取 k - i 个
		list1, list2 := getMaxSubsequence(nums1, i), getMaxSubsequence(nums2, k-i)
		nums := mergeMaxNumber(list1, list2)
		if compare(nums, result, 0, 0) {
			result = nums
		}
	}

	return result
}

func compare(nums1, nums2 []int, i, j int) bool {
	m, n := len(nums1), len(nums2)
	if j >= n {
		return true
	}
	if i >= m {
		return false
	}
	if nums1[i] > nums2[j] {
		return true
	}
	if nums1[i] < nums2[j] {
		return false
	}

	return compare(nums1, nums2, i+1, j+1)
}

// 合并数组
func mergeMaxNumber(nums1, nums2 []int) []int {
	m, n := len(nums1), len(nums2)
	if m == 0 {
		return nums2
	}
	if n == 0 {
		return nums1
	}
	result := make([]int, m+n)
	i, j, idx := 0, 0, 0
	for i < m || j < n {
		if compare(nums1, nums2, i, j) {
			result[idx] = nums1[i]
			i++
		} else {
			result[idx] = nums2[j]
			j++
		}
		idx++
	}

	return result
}

// 获取 nums 长度为k的最大子序列
func getMaxSubsequence(nums []int, k int) []int {
	result := make([]int, k)
	if k == 0 {
		return result
	}
	idx, rem := 0, len(nums)-k

	for _, num := range nums {
		for idx > 0 && rem > 0 && result[idx-1] < num {
			idx--
			rem--
		}
		if idx < k {
			result[idx] = num
			idx++
		} else {
			rem--
		}
	}
	return result
}

// 324. 摆动排序 II
// 给你一个整数数组 nums，将它重新排列成 nums[0] < nums[1] > nums[2] < nums[3]... 的顺序。
//
// 你可以假设所有输入数组都可以得到满足题目要求的结果。
//
// 示例 1：
// 输入：nums = [1,5,1,1,6,4]
// 输出：[1,6,1,5,1,4]
// 解释：[1,4,1,5,1,6] 同样是符合题目要求的结果，可以被判题程序接受。
//
// 示例 2：
// 输入：nums = [1,3,2,2,3,1]
// 输出：[2,3,1,3,1,2]
//
// 提示：
// 1 <= nums.length <= 5 * 104
// 0 <= nums[i] <= 5000
// 题目数据保证，对于给定的输入 nums ，总能产生满足题目要求的结果
//
// 进阶：你能用 O(n) 时间复杂度和 / 或原地 O(1) 额外空间来实现吗？
func wiggleSort(nums []int) {
	n := len(nums)
	if n < 2 {
		return
	}
	newNums := make([]int, n)
	copy(newNums, nums)
	sort.Ints(newNums)
	right := n - 1
	left := right >> 1
	for i := 0; i < n; i++ {
		if i&1 == 1 {
			nums[i] = newNums[right]
			right--
		} else {
			nums[i] = newNums[left]
			left--
		}
	}
}

// 334. 递增的三元子序列
// 给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。
//
// 如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：nums = [1,2,3,4,5]
// 输出：true
// 解释：任何 i < j < k 的三元组都满足题意
//
// 示例 2：
// 输入：nums = [5,4,3,2,1]
// 输出：false
// 解释：不存在满足题意的三元组
//
// 示例 3：
// 输入：nums = [2,1,5,0,4,6]
// 输出：true
// 解释：三元组 (3, 4, 5) 满足题意，因为 nums[3] == 0 < nums[4] == 4 < nums[5] == 6
//
// 提示：
// 1 <= nums.length <= 105
// -231 <= nums[i] <= 231 - 1
//
// 进阶：你能实现时间复杂度为 O(n) ，空间复杂度为 O(1) 的解决方案吗？
func increasingTriplet(nums []int) bool {
	n := len(nums)
	if n < 3 {
		return false
	}
	num1, num2 := nums[0], math.MaxInt32

	for i := 1; i < n; i++ {
		if nums[i] < num1 {
			num1 = nums[i]
		} else if nums[i] > num1 && nums[i] <= num2 {
			num2 = nums[i]
		} else if nums[i] > num2 {
			return true
		}
	}

	return false
}

// 406. 根据身高重建队列
// 假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
//
// 请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。
//
// 示例 1：
// 输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
// 输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
// 解释：
// 编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
// 编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
// 编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
// 编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
// 编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
// 编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
// 因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
//
// 示例 2：
// 输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
// 输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
//
// 提示：
// 1 <= people.length <= 2000
// 0 <= hi <= 106
// 0 <= ki < people.length
// 题目数据确保队列可以被重建
func reconstructQueue(people [][]int) [][]int {

	result := make([][]int, 0)
	// 按 身高 高 -> 低 , k 小 -> 大 排序
	sort.Slice(people, func(i, j int) bool {
		// 身高相同
		if people[i][0] == people[j][0] {
			return people[i][1] < people[j][1]
		}
		return people[i][0] > people[j][0]
	})

	for _, person := range people {
		idx := person[1]
		tmp := append([][]int{person}, result[idx:]...)
		result = append(result[:idx], tmp...)
	}

	return result

}

// 410. 分割数组的最大值
// 给定一个非负整数数组 nums 和一个整数 m ，你需要将这个数组分成 m 个非空的连续子数组。
//
// 设计一个算法使得这 m 个子数组各自和的最大值最小。
//
// 示例 1：
// 输入：nums = [7,2,5,10,8], m = 2
// 输出：18
// 解释：
// 一共有四种方法将 nums 分割为 2 个子数组。 其中最好的方式是将其分为 [7,2,5] 和 [10,8] 。
// 因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。
//
// 示例 2：
// 输入：nums = [1,2,3,4,5], m = 2
// 输出：9
//
// 示例 3：
// 输入：nums = [1,4,4], m = 3
// 输出：4
//
// 提示：
// 1 <= nums.length <= 1000
// 0 <= nums[i] <= 106
// 1 <= m <= min(50, nums.length)
func splitArray(nums []int, m int) int {
	maxVal, sum := 0, 0
	for _, num := range nums {
		maxVal = max(maxVal, num)
		sum += num
	}
	if m == 1 {
		return sum
	}
	// 思路 子数组的最大值是有范围的，即在区间 [max(nums),sum(nums)] 之中。
	// 令 l = max(nums)，h = sum(nums)  mid=(l+h)/2
	// 计算数组和最大值不大于mid对应的子数组个数 cnt(这个是关键！)
	// 如果 cnt>m，说明划分的子数组多了，即我们找到的 mid 偏小，故 l=mid+1l=mid+1；
	// 否则，说明划分的子数组少了，即 mid 偏大(或者正好就是目标值)，故 h=midh=mid。
	low, high := maxVal, sum
	for low < high {
		mid := (low + high) >> 1

		count, tmpSum := 1, 0
		for _, num := range nums {
			tmpSum += num
			if tmpSum > mid {
				count++
				tmpSum = num
			}
		}
		// count>m，说明划分的子数组多了， mid 偏小
		if count > m {
			low = mid + 1
		} else {
			high = mid
		}

	}
	return low
}

// 442. 数组中重复的数据
// 给定一个整数数组 a，其中1 ≤ a[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。
//
// 找到所有出现两次的元素。
// 你可以不用到任何额外空间并在O(n)时间复杂度内解决这个问题吗？
//
// 示例：
// 输入: [4,3,2,7,8,2,3,1]
// 输出: [2,3]
func findDuplicates(nums []int) []int {
	n := len(nums)
	result := make([]int, 0)
	// 最大值 是 n
	for _, num := range nums {
		index := (num - 1) % n
		nums[index] += n
	}
	for i, num := range nums {
		if num > 2*n {
			result = append(result, i+1)
		}
	}

	return result
}

// 457. 环形数组是否存在循环
// 存在一个不含 0 的 环形 数组 nums ，每个 nums[i] 都表示位于下标 i 的角色应该向前或向后移动的下标个数：
//
// 如果 nums[i] 是正数，向前（下标递增方向）移动 |nums[i]| 步
// 如果 nums[i] 是负数，向后（下标递减方向）移动 |nums[i]| 步
// 因为数组是 环形 的，所以可以假设从最后一个元素向前移动一步会到达第一个元素，而第一个元素向后移动一步会到达最后一个元素。
//
// 数组中的 循环 由长度为 k 的下标序列 seq 标识：
//
// 遵循上述移动规则将导致一组重复下标序列 seq[0] -> seq[1] -> ... -> seq[k - 1] -> seq[0] -> ...
// 所有 nums[seq[j]] 应当不是 全正 就是 全负
// k > 1
// 如果 nums 中存在循环，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：nums = [2,-1,1,2,2]
// 输出：true
// 解释：存在循环，按下标 0 -> 2 -> 3 -> 0 。循环长度为 3 。
//
// 示例 2：
// 输入：nums = [-1,2]
// 输出：false
// 解释：按下标 1 -> 1 -> 1 ... 的运动无法构成循环，因为循环的长度为 1 。根据定义，循环的长度必须大于 1 。
//
// 示例 3:
// 输入：nums = [-2,1,-1,-2,-2]
// 输出：false
// 解释：按下标 1 -> 2 -> 1 -> ... 的运动无法构成循环，因为 nums[1] 是正数，而 nums[2] 是负数。
// 所有 nums[seq[j]] 应当不是全正就是全负。
//
// 提示：
// 1 <= nums.length <= 5000
// -1000 <= nums[i] <= 1000
// nums[i] != 0
//
// 进阶：你能设计一个时间复杂度为 O(n) 且额外空间复杂度为 O(1) 的算法吗？
func circularArrayLoop(nums []int) bool {
	n := len(nums)
	if n < 2 {
		return false
	}
	next := func(cur int) int {
		return ((cur+nums[cur])%n + n) % n
	}
	for i, num := range nums {
		if num == 0 {
			continue
		}
		slow, fast := i, next(i)
		for nums[slow]*nums[fast] > 0 && nums[slow]*nums[next(fast)] > 0 {
			if slow == fast {
				if slow == next(slow) {
					break
				}
				return true
			}
			slow = next(slow)
			fast = next(next(fast))
		}
		add := i
		for nums[add]*nums[next(add)] > 0 {
			tmp := add
			add = next(add)
			nums[tmp] = 0
		}

	}

	return false
}

// 462. 最少移动次数使数组元素相等 II
// 给定一个非空整数数组，找到使所有数组元素相等所需的最小移动数，其中每次移动可将选定的一个元素加1或减1。 您可以假设数组的长度最多为10000。
//
// 例如:
// 输入: [1,2,3]
// 输出: 2
// 说明：
// 只有两个动作是必要的（记得每一步仅可使其中一个元素加1或减1）：
//
// [1,2,3]  =>  [2,2,3]  =>  [2,2,2]
func minMoves2(nums []int) int {
	move := 0
	sort.Ints(nums)
	left, right := 0, len(nums)-1
	for left < right {
		move += nums[right] - nums[left]
		left++
		right--
	}
	return move
}

// 475. 供暖器
// 冬季已经来临。 你的任务是设计一个有固定加热半径的供暖器向所有房屋供暖。
//
// 在加热器的加热半径范围内的每个房屋都可以获得供暖。
// 现在，给出位于一条水平线上的房屋 houses 和供暖器 heaters 的位置，请你找出并返回可以覆盖所有房屋的最小加热半径。
// 说明：所有供暖器都遵循你的半径标准，加热的半径也一样。
//
// 示例 1:
// 输入: houses = [1,2,3], heaters = [2]
// 输出: 1
// 解释: 仅在位置2上有一个供暖器。如果我们将加热半径设为1，那么所有房屋就都能得到供暖。
//
// 示例 2:
// 输入: houses = [1,2,3,4], heaters = [1,4]
// 输出: 1
// 解释: 在位置1, 4上有两个供暖器。我们需要将加热半径设为1，这样所有房屋就都能得到供暖。
//
// 示例 3：
// 输入：houses = [1,5], heaters = [2]
// 输出：3
//
// 提示：
// 1 <= houses.length, heaters.length <= 3 * 104
// 1 <= houses[i], heaters[i] <= 109
func findRadius(houses []int, heaters []int) int {
	sort.Ints(houses)
	sort.Ints(heaters)
	radius, index := 0, 0
	m, n := len(houses), len(heaters)
	for _, house := range houses {
		// 一直找到处于房屋右侧的供暖器
		for index < n && heaters[index] < house {
			index++
		}
		if index == 0 {
			radius = max(radius, heaters[index]-house)
		} else if index == n {
			// 最后
			radius = max(radius, houses[m-1]-heaters[n-1])
		} else {
			// house 在两个供暖中间
			radius = max(radius, min(heaters[index]-house, house-heaters[index-1]))
		}
	}
	return radius
}

// 495. 提莫攻击
// 在《英雄联盟》的世界中，有一个叫 “提莫” 的英雄。他的攻击可以让敌方英雄艾希（编者注：寒冰射手）进入中毒状态。
// 当提莫攻击艾希，艾希的中毒状态正好持续 duration 秒。
// 正式地讲，提莫在 t 发起发起攻击意味着艾希在时间区间 [t, t + duration - 1]（含 t 和 t + duration - 1）处于中毒状态。如果提莫在中毒影响结束 前 再次攻击，中毒状态计时器将会 重置 ，在新的攻击之后，中毒影响将会在 duration 秒后结束。
// 给你一个 非递减 的整数数组 timeSeries ，其中 timeSeries[i] 表示提莫在 timeSeries[i] 秒时对艾希发起攻击，以及一个表示中毒持续时间的整数 duration 。
//
// 返回艾希处于中毒状态的 总 秒数。
//
// 示例 1：
// 输入：timeSeries = [1,4], duration = 2
// 输出：4
// 解释：提莫攻击对艾希的影响如下：
// - 第 1 秒，提莫攻击艾希并使其立即中毒。中毒状态会维持 2 秒，即第 1 秒和第 2 秒。
// - 第 4 秒，提莫再次攻击艾希，艾希中毒状态又持续 2 秒，即第 4 秒和第 5 秒。
// 艾希在第 1、2、4、5 秒处于中毒状态，所以总中毒秒数是 4 。
//
// 示例 2：
// 输入：timeSeries = [1,2], duration = 2
// 输出：3
// 解释：提莫攻击对艾希的影响如下：
// - 第 1 秒，提莫攻击艾希并使其立即中毒。中毒状态会维持 2 秒，即第 1 秒和第 2 秒。
// - 第 2 秒，提莫再次攻击艾希，并重置中毒计时器，艾希中毒状态需要持续 2 秒，即第 2 秒和第 3 秒。
// 艾希在第 1、2、3 秒处于中毒状态，所以总中毒秒数是 3 。
//
// 提示：
// 1 <= timeSeries.length <= 104
// 0 <= timeSeries[i], duration <= 107
// timeSeries 按 非递减 顺序排列
func findPoisonedDuration(timeSeries []int, duration int) int {
	result := 0
	n := len(timeSeries)
	for i := 1; i < n; i++ {
		result += min(duration, timeSeries[i]-timeSeries[i-1])
	}
	if n > 0 {
		result += duration
	}

	return result
}

// 493. 翻转对
// 给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。
//
// 你需要返回给定数组中的重要翻转对的数量。
//
// 示例 1:
// 输入: [1,3,2,3,1]
// 输出: 2
//
// 示例 2:
// 输入: [2,4,3,5,1]
// 输出: 3
// 注意:
// 给定数组的长度不会超过50000。
// 输入数组中的所有数字都在32位整数的表示范围内。
func reversePairs(nums []int) int {
	n := len(nums)

	var mergeSort func(start, end int) int

	// 归并排序
	mergeSort = func(start, end int) int {
		if start == end {
			return 0
		}
		mid := (start + end) >> 1
		count1, count2 := mergeSort(start, mid), mergeSort(mid+1, end)
		result := count1 + count2
		i, j := start, mid+1
		for ; i <= mid; i++ {
			for j <= end && nums[i] > 2*nums[j] {
				j++
			}
			result += j - mid - 1
		}

		// 合并有序数组
		tmpNums := make([]int, end-start+1)
		index := 0
		i, j = start, mid+1
		for index < len(tmpNums) {
			if i > mid {
				tmpNums[index] = nums[j]
				index++
				j++
				continue
			}
			if j > end {
				tmpNums[index] = nums[i]
				index++
				i++
				continue
			}
			if nums[i] < nums[j] {
				tmpNums[index] = nums[i]
				index++
				i++
			} else {
				tmpNums[index] = nums[j]
				index++
				j++
			}
		}
		for k := 0; k < len(tmpNums); k++ {
			nums[start+k] = tmpNums[k]
		}
		return result
	}

	return mergeSort(0, n-1)
}

// 525. 连续数组
// 给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。
//
// 示例 1:
// 输入: nums = [0,1]
// 输出: 2
// 说明: [0, 1] 是具有相同数量 0 和 1 的最长连续子数组。
//
// 示例 2:
// 输入: nums = [0,1,0]
// 输出: 2
// 说明: [0, 1] (或 [1, 0]) 是具有相同数量0和1的最长连续子数组。
//
// 提示：
// 1 <= nums.length <= 105
// nums[i] 不是 0 就是 1
func findMaxLength(nums []int) int {
	indexMap := map[int]int{
		0: -1,
	}
	result := 0
	num := 0
	for i, n := range nums {
		if n == 1 {
			num++
		} else {
			num--
		}
		if last, ok := indexMap[num]; ok {
			result = max(result, i-last)
		} else {
			indexMap[num] = i
		}
	}
	return result
}

// 539. 最小时间差
// 给定一个 24 小时制（小时:分钟 "HH:MM"）的时间列表，找出列表中任意两个时间的最小时间差并以分钟数表示。
//
// 示例 1：
// 输入：timePoints = ["23:59","00:00"] 输出：1
//
// 示例 2：
// 输入：timePoints = ["00:00","23:59","00:00"] 输出：0
//
// 提示：
// 2 <= timePoints <= 2 * 104
// timePoints[i] 格式为 "HH:MM"
func findMinDifference(timePoints []string) int {
	n := len(timePoints)
	times := make([]int, n)
	for i := 0; i < n; i++ {
		h, _ := strconv.Atoi(timePoints[i][:2])
		m, _ := strconv.Atoi(timePoints[i][3:])
		times[i] = h*60 + m
	}
	sort.Ints(times)
	result := 24*60 - (times[n-1] - times[0])
	for i := 0; i < n-1; i++ {
		result = min(result, times[i+1]-times[i])
	}
	return result
}

// 547. 省份数量
// 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
//
// 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
//
// 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。
//
// 返回矩阵中 省份 的数量。
//
// 示例 1：
// 输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]] 输出：2
//
// 示例 2：
// 输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]] 输出：3
//
// 提示：
// 1 <= n <= 200
// n == isConnected.length
// n == isConnected[i].length
// isConnected[i][j] 为 1 或 0
// isConnected[i][i] == 1
// isConnected[i][j] == isConnected[j][i]
func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	visited := make([]bool, n)
	result := 0
	var dfs func(num int)

	dfs = func(num int) {
		for next, con := range isConnected[num] {
			if con == 1 && !visited[next] {
				visited[next] = true
				dfs(next)
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i)
			result++
		}
	}

	return result
}

// 565. 数组嵌套
// 索引从0开始长度为N的数组A，包含0到N - 1的所有整数。找到最大的集合S并返回其大小，其中 S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }且遵守以下的规则。
//
// 假设选择索引为i的元素A[i]为S的第一个元素，S的下一个元素应该是A[A[i]]，之后是A[A[A[i]]]... 以此类推，不断添加直到S出现重复的元素。
//
// 示例 1:
// 输入: A = [5,4,0,3,1,6,2]
// 输出: 4
// 解释:
// A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.
//
// 其中一种最长的 S[K]:
// S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
//
// 提示：
// N是[1, 20,000]之间的整数。
// A中不含有重复的元素。
// A中的元素大小在[0, N-1]之间。
func arrayNesting(nums []int) int {
	n := len(nums)
	visited := make([]bool, n)

	result := 0

	for i, num := range nums {
		if visited[i] {
			continue
		}
		start, count := num, 1
		visited[start] = true
		for nums[start] != num {
			start = nums[start]
			visited[start] = true
			count++
		}
		result = max(result, count)
	}

	return result

}

// 581. 最短无序连续子数组
// 给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
// 请你找出符合题意的 最短 子数组，并输出它的长度。
//
// 示例 1：
// 输入：nums = [2,6,4,8,10,9,15]
// 输出：5
// 解释：你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
//
// 示例 2：
// 输入：nums = [1,2,3,4]
// 输出：0
//
// 示例 3：
// 输入：nums = [1]
// 输出：0
//
// 提示：
// 1 <= nums.length <= 104
// -105 <= nums[i] <= 105
//
// 进阶：你可以设计一个时间复杂度为 O(n) 的解决方案吗？
func findUnsortedSubarray(nums []int) int {
	n := len(nums)
	// 从前往后 找最小索引
	minVal, maxVal := nums[n-1], nums[0]
	begin, end := 0, -1
	for i := range nums {
		if nums[i] >= maxVal {
			maxVal = nums[i]
		} else {
			end = i
		}
		if nums[n-i-1] <= minVal {
			minVal = nums[n-i-1]
		} else {
			begin = n - i - 1
		}
	}
	if end > begin {
		return end - begin + 1
	}
	return 0
}

// 611. 有效三角形的个数
// 给定一个包含非负整数的数组，你的任务是统计其中可以组成三角形三条边的三元组个数。
//
// 示例 1:
// 输入: [2,2,3,4] 输出: 3
// 解释:
// 有效的组合是:
// 2,3,4 (使用第一个 2)
// 2,3,4 (使用第二个 2)
// 2,2,3
//
// 注意:
// 数组长度不超过1000。
// 数组里整数的范围为 [0, 1000]。
func triangleNumber(nums []int) int {
	n := len(nums)
	result := 0
	sort.Ints(nums)
	for i := n - 1; i > 1; i-- {
		left, right := 0, i-1
		for left < right {
			if nums[left]+nums[right] > nums[i] {
				result += right - left
				right--
			} else {
				left++
			}

		}
	}

	return result
}

// 621. 任务调度器
// 给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。
//
// 然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
//
// 你需要计算完成所有任务所需要的 最短时间 。
//
// 示例 1：
// 输入：tasks = ["A","A","A","B","B","B"], n = 2
// 输出：8
// 解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B
//     在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。
//
// 示例 2：
// 输入：tasks = ["A","A","A","B","B","B"], n = 0
// 输出：6
// 解释：在这种情况下，任何大小为 6 的排列都可以满足要求，因为 n = 0
// ["A","A","A","B","B","B"]
// ["A","B","A","B","A","B"]
// ["B","B","B","A","A","A"]
// ...
// 诸如此类
//
// 示例 3：
// 输入：tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
// 输出：16
// 解释：一种可能的解决方案是：
//     A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> (待命) -> (待命) -> A -> (待命) -> (待命) -> A
//
// 提示：
// 1 <= task.length <= 104
// tasks[i] 是大写英文字母
// n 的取值范围为 [0, 100]
func leastInterval(tasks []byte, n int) int {
	var taskNums [26]int
	// 计算每个任务出现的次数
	for _, c := range tasks {
		taskNums[c-'A']++
	}
	m := len(tasks)
	// 找出出现次数最多的任务，假设出现次数为 maxCount
	maxCount := 0
	for _, count := range taskNums {
		maxCount = max(maxCount, count)
	}

	// 计算出现次数为 maxCount 的任务总数 lastCount，计算最终结果为 minTimes + lastCount
	lastCount := 0
	for _, count := range taskNums {
		if count == maxCount {
			lastCount++
		}
	}

	// 计算至少需要的时间 (max - 1) * (n + 1)，记为 minTimes
	result := (maxCount-1)*(n+1) + lastCount
	return max(m, result)
}

// 1995. 统计特殊四元组
// 给你一个 下标从 0 开始 的整数数组 nums ，返回满足下述条件的 不同 四元组 (a, b, c, d) 的 数目 ：
//
// nums[a] + nums[b] + nums[c] == nums[d] ，且
// a < b < c < d
//
// 示例 1：
// 输入：nums = [1,2,3,6]
// 输出：1
// 解释：满足要求的唯一一个四元组是 (0, 1, 2, 3) 因为 1 + 2 + 3 == 6 。
//
// 示例 2：
// 输入：nums = [3,3,6,4,5]
// 输出：0
// 解释：[3,3,6,4,5] 中不存在满足要求的四元组。
//
// 示例 3：
// 输入：nums = [1,1,1,3,5]
// 输出：4
// 解释：满足要求的 4 个四元组如下：
// - (0, 1, 2, 3): 1 + 1 + 1 == 3
// - (0, 1, 3, 4): 1 + 1 + 3 == 5
// - (0, 2, 3, 4): 1 + 1 + 3 == 5
// - (1, 2, 3, 4): 1 + 1 + 3 == 5
//
// 提示：
// 4 <= nums.length <= 50
// 1 <= nums[i] <= 100
func countQuadruplets(nums []int) int {
	n := len(nums)
	//nums[a] + nums[b]  == nums[d] - nums[c]
	counts := make(map[int]int)

	result := 0
	// 枚举b c
	for b := n - 3; b > 0; b-- {
		// 枚举 d
		for d := b + 2; d < n; d++ {
			counts[nums[d]-nums[b+1]]++
		}

		// 枚举 a
		for a := 0; a < b; a++ {
			result += counts[nums[a]+nums[b]]
		}

	}
	return result
}

// 632. 最小区间
// 你有 k 个 非递减排列 的整数列表。找到一个 最小 区间，使得 k 个列表中的每个列表至少有一个数包含在其中。
//
// 我们定义如果 b-a < d-c 或者在 b-a == d-c 时 a < c，则区间 [a,b] 比 [c,d] 小。
//
// 示例 1：
// 输入：nums = [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
// 输出：[20,24]
// 解释：
// 列表 1：[4, 10, 15, 24, 26]，24 在区间 [20,24] 中。
// 列表 2：[0, 9, 12, 20]，20 在区间 [20,24] 中。
// 列表 3：[5, 18, 22, 30]，22 在区间 [20,24] 中。
//
// 示例 2：
// 输入：nums = [[1,2,3],[1,2,3],[1,2,3]]
// 输出：[1,1]
//
// 示例 3：
// 输入：nums = [[10,10],[11,11]]
// 输出：[10,11]
//
// 示例 4：
// 输入：nums = [[10],[11]]
// 输出：[10,11]
//
// 示例 5：
// 输入：nums = [[1],[2],[3],[4],[5],[6],[7]]
// 输出：[1,7]
//
// 提示：
// nums.length == k
// 1 <= k <= 3500
// 1 <= nums[i].length <= 50
// -105 <= nums[i][j] <= 105
// nums[i] 按非递减顺序排列
func smallestRange(nums [][]int) []int {
	// 首先将 k  组数据升序合并成一组，并记录每个数字所属的组，
	k := len(nums)
	pairs := make([][]int, 0)
	for i := 0; i < k; i++ {
		for _, num := range nums[i] {
			pairs = append(pairs, []int{num, i})
		}
	}
	// 合并后升序排列
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i][0] < pairs[j][0]
	})
	fmt.Println(pairs)

	result := make([]int, 0)

	left, count := 0, 0
	indexMap := make(map[int]int)
	// 滑动窗口
	for _, pair := range pairs {
		num, idx := pair[0], pair[1]
		if indexMap[idx] == 0 {
			count++
		}
		indexMap[idx]++
		if count == k {
			for indexMap[pairs[left][1]] > 1 {
				indexMap[pairs[left][1]]--
				left++
			}
			if len(result) == 0 || result[1]-result[0] > num-pairs[left][0] {
				result = []int{num, pairs[left][0]}
			}

		}

	}
	return result
}

// 163. 缺失的区间
// 给定一个排序的整数数组 nums ，其中元素的范围在 闭区间 [lower, upper] 当中，返回不包含在数组中的缺失区间。
//
// 示例：
// 输入: nums = [0, 1, 3, 50, 75], lower = 0 和 upper = 99,
// 输出: ["2", "4->49", "51->74", "76->99"]
func findMissingRanges(nums []int, lower int, upper int) []string {
	n := len(nums)

	getRange := func(start, end int) string {
		if start == end {
			return strconv.Itoa(start)
		}
		return fmt.Sprintf("%d->%d", start, end)
	}

	result := make([]string, 0)
	if n == 0 {
		result = append(result, getRange(lower, upper))
		return result
	}

	if nums[0]-lower > 0 {
		result = append(result, getRange(lower, nums[0]-1))
	}
	for i := 1; i < n; i++ {
		if nums[i]-nums[i-1] > 1 {
			result = append(result, getRange(nums[i-1]+1, nums[i]-1))
		}
	}
	if upper-nums[n-1] > 0 {
		result = append(result, getRange(nums[n-1]+1, upper))
	}

	return result
}

// 747. 至少是其他数字两倍的最大数
// 给你一个整数数组 nums ，其中总是存在 唯一的 一个最大整数 。
//
// 请你找出数组中的最大元素并检查它是否 至少是数组中每个其他数字的两倍 。如果是，则返回 最大元素的下标 ，否则返回 -1 。
//
// 示例 1：
// 输入：nums = [3,6,1,0]
// 输出：1
// 解释：6 是最大的整数，对于数组中的其他整数，6 大于数组中其他元素的两倍。6 的下标是 1 ，所以返回 1 。
//
// 示例 2：
// 输入：nums = [1,2,3,4]
// 输出：-1
// 解释：4 没有超过 3 的两倍大，所以返回 -1 。
//
// 示例 3：
//
// 输入：nums = [1]
// 输出：0
// 解释：因为不存在其他数字，所以认为现有数字 1 至少是其他数字的两倍。
//
// 提示：
// 1 <= nums.length <= 50
// 0 <= nums[i] <= 100
// nums 中的最大元素是唯一的
func dominantIndex(nums []int) int {
	n := len(nums)
	if n < 2 {
		return 0
	}
	indexMap := make(map[int]int)
	for i, num := range nums {
		indexMap[num] = i
	}
	sort.Ints(nums)

	maxVal := nums[n-1]
	if maxVal < 2*nums[n-2] {
		return -1
	}
	return indexMap[maxVal]
}

// 649. Dota2 参议院
// Dota2 的世界里有两个阵营：Radiant(天辉)和 Dire(夜魇)
//
// Dota2 参议院由来自两派的参议员组成。现在参议院希望对一个 Dota2 游戏里的改变作出决定。他们以一个基于轮为过程的投票进行。在每一轮中，每一位参议员都可以行使两项权利中的一项：
//   1.禁止一名参议员的权利：
//     参议员可以让另一位参议员在这一轮和随后的几轮中丧失所有的权利。
//   2.宣布胜利：
// 如果参议员发现有权利投票的参议员都是同一个阵营的，他可以宣布胜利并决定在游戏中的有关变化。
// 给定一个字符串代表每个参议员的阵营。字母 “R” 和 “D” 分别代表了 Radiant（天辉）和 Dire（夜魇）。然后，如果有 n 个参议员，给定字符串的大小将是 n。
//
// 以轮为基础的过程从给定顺序的第一个参议员开始到最后一个参议员结束。这一过程将持续到投票结束。所有失去权利的参议员将在过程中被跳过。
//
// 假设每一位参议员都足够聪明，会为自己的政党做出最好的策略，你需要预测哪一方最终会宣布胜利并在 Dota2 游戏中决定改变。输出应该是 Radiant 或 Dire。
//
// 示例 1：
// 输入："RD"
// 输出："Radiant"
// 解释：第一个参议员来自 Radiant 阵营并且他可以使用第一项权利让第二个参议员失去权力，因此第二个参议员将被跳过因为他没有任何权利。然后在第二轮的时候，第一个参议员可以宣布胜利，因为他是唯一一个有投票权的人
//
// 示例 2：
// 输入："RDD"
// 输出："Dire"
// 解释：
// 第一轮中,第一个来自 Radiant 阵营的参议员可以使用第一项权利禁止第二个参议员的权利
// 第二个来自 Dire 阵营的参议员会被跳过因为他的权利被禁止
// 第三个来自 Dire 阵营的参议员可以使用他的第一项权利禁止第一个参议员的权利
// 因此在第二轮只剩下第三个参议员拥有投票的权利,于是他可以宣布胜利
//
// 提示：
// 给定字符串的长度在 [1, 10,000] 之间.
func predictPartyVictory(senate string) string {
	// 使用一个整数队列表示所有的参议员：1 代表 'Radiant' 阵营；0 代表 'Dire' 阵营。
	// 遍历队列：对于当前队头的参议员，如果另外一个阵营有禁令，则禁止当前参议员的权利；
	// 如果另外一个阵营没有禁令，则该参议员所在阵营的禁令数量加 1。
	people, bans := [2]int{}, [2]int{}
	queue := list.New()
	for _, s := range senate {
		x := 1
		if s == 'D' {
			x = 0
		}
		people[x]++
		queue.PushBack(x)
	}
	for people[0] > 0 && people[1] > 0 && queue.Len() > 0 {
		front := queue.Front()
		queue.Remove(front)
		x := front.Value.(int)
		// 存在禁令 该参议员 跳过
		if bans[x] > 0 {
			bans[x]--
			people[x]--
			continue
		}

		// 对对方的禁令+1;
		bans[x^1]++
		queue.PushBack(x)
	}
	if people[0] > 0 {
		return "Dire"
	}
	return "Radiant"
}

// 659. 分割数组为连续子序列
// 给你一个按升序排序的整数数组 num（可能包含重复数字），请你将它们分割成一个或多个长度至少为 3 的子序列，其中每个子序列都由连续整数组成。
//
// 如果可以完成上述分割，则返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入: [1,2,3,3,4,5]
// 输出: True
// 解释:
// 你可以分割出这样两个连续子序列 :
// 1, 2, 3
// 3, 4, 5
//
// 示例 2：
// 输入: [1,2,3,3,4,4,5,5]
// 输出: True
// 解释:
// 你可以分割出这样两个连续子序列 :
// 1, 2, 3, 4, 5
// 3, 4, 5
//
// 示例 3：
// 输入: [1,2,3,4,4,5]
// 输出: False
//
// 提示：
// 1 <= nums.length <= 10000
func isPossible(nums []int) bool {
	// nc[i]：存储原数组中数字i出现的次数 tail[i]：存储以数字i结尾的且符合题意的连续子序列个数
	// 1. 先去寻找一个长度为3的连续子序列i, i+1, i+2，找到后就将nc[i], nc[i+1],
	// nc[i+2]中对应数字消耗1个（即-1），并将tail[i+2]加1，即以i+2结尾的子序列个数+1。
	// 2. 如果后续发现有能够接在这个连续子序列的数字i+3，
	// 则延长以i+2为结尾的连续子序列到i+3，此时消耗nc[i+3]一个，由于子序列已延长，因此tail[i+2]减1，tail[i+3]加1
	// 在不满足上面的情况下
	// 3. 如果nc[i]为0，说明这个数字已经消耗完，可以不管了
	// 4. 如果nc[i]不为0，说明这个数字多出来了，且无法组成连续子序列，所以可以直接返回false了
	nc, tail := make(map[int]int), make(map[int]int)
	for _, num := range nums {
		nc[num]++
	}
	for _, num := range nums {
		if nc[num] == 0 {
			continue
		}
		if tail[num-1] > 0 {
			tail[num-1]--
			tail[num]++
		} else if nc[num+1] > 0 && nc[num+2] > 0 {
			nc[num+1]--
			nc[num+2]--
			tail[num+2]++
		} else {
			return false
		}
		nc[num]--
	}
	return true
}

// 667. 优美的排列 II
// 给你两个整数 n 和 k ，请你构造一个答案列表 answer ，该列表应当包含从 1 到 n 的 n 个不同正整数，并同时满足下述条件：
//
// 假设该列表是 answer = [a1, a2, a3, ... , an] ，那么列表 [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|] 中应该有且仅有 k 个不同整数。
// 返回列表 answer 。如果存在多种答案，只需返回其中 任意一种 。
//
// 示例 1：
// 输入：n = 3, k = 1
// 输出：[1, 2, 3]
// 解释：[1, 2, 3] 包含 3 个范围在 1-3 的不同整数，并且 [1, 1] 中有且仅有 1 个不同整数：1
//
// 示例 2：
// 输入：n = 3, k = 2
// 输出：[1, 3, 2]
// 解释：[1, 3, 2] 包含 3 个范围在 1-3 的不同整数，并且 [2, 1] 中有且仅有 2 个不同整数：1 和 2
//
// 提示：
// 1 <= k < n <= 104
func constructArray(n int, k int) []int {
	nums := make([]int, n)
	left, right := 1, n
	flag := true
	for i := 0; i < k-1; i++ {
		if flag {
			nums[i] = left
			left++
		} else {
			nums[i] = right
			right--
		}
		flag = !flag
	}
	k--
	for i := left; i <= right; i++ {
		if flag {
			nums[k] = i
		} else {
			nums[k] = right + left - i
		}
		k++
	}
	return nums
}

// 1996. 游戏中弱角色的数量
// 你正在参加一个多角色游戏，每个角色都有两个主要属性：攻击 和 防御 。给你一个二维整数数组 properties ，其中 properties[i] = [attacki, defensei] 表示游戏中第 i 个角色的属性。
//
// 如果存在一个其他角色的攻击和防御等级 都严格高于 该角色的攻击和防御等级，则认为该角色为 弱角色 。更正式地，如果认为角色 i 弱于 存在的另一个角色 j ，那么 attackj > attacki 且 defensej > defensei 。
//
// 返回 弱角色 的数量。
//
// 示例 1：
// 输入：properties = [[5,5],[6,3],[3,6]]
// 输出：0
// 解释：不存在攻击和防御都严格高于其他角色的角色。
//
// 示例 2：
// 输入：properties = [[2,2],[3,3]]
// 输出：1
// 解释：第一个角色是弱角色，因为第二个角色的攻击和防御严格大于该角色。
//
// 示例 3：
// 输入：properties = [[1,5],[10,4],[4,3]]
// 输出：1
// 解释：第三个角色是弱角色，因为第二个角色的攻击和防御严格大于该角色。
//
// 提示：
// 2 <= properties.length <= 105
// properties[i].length == 2
// 1 <= attacki, defensei <= 105
func numberOfWeakCharacters(properties [][]int) int {
	// 攻击 降序 防御 升序  然后使用栈 找出 最大的
	sort.Slice(properties, func(i, j int) bool {
		if properties[i][0] == properties[j][0] {
			return properties[i][1] < properties[j][1]
		} else {
			return properties[i][0] > properties[j][0]
		}
	})
	n := len(properties)
	count := 0
	maxDef := properties[0][1]
	for i := 1; i < n; i++ {
		if properties[i][1] < maxDef {
			count++
		} else if properties[i][1] > maxDef {
			maxDef = properties[i][1]
		}
	}
	return count
}

// 2006. 差的绝对值为 K 的数对数目
// 给你一个整数数组 nums 和一个整数 k ，请你返回数对 (i, j) 的数目，满足 i < j 且 |nums[i] - nums[j]| == k 。
//
// |x| 的值定义为：
// 如果 x >= 0 ，那么值为 x 。
// 如果 x < 0 ，那么值为 -x 。
//
// 示例 1：
// 输入：nums = [1,2,2,1], k = 1
// 输出：4
// 解释：差的绝对值为 1 的数对为：
// - [1,2,2,1]
// - [1,2,2,1]
// - [1,2,2,1]
// - [1,2,2,1]
//
// 示例 2：
// 输入：nums = [1,3], k = 3
// 输出：0
// 解释：没有任何数对差的绝对值为 3 。
//
// 示例 3：
// 输入：nums = [3,2,1,5,4], k = 2
// 输出：3
// 解释：差的绝对值为 2 的数对为：
// - [3,2,1,5,4]
// - [3,2,1,5,4]
// - [3,2,1,5,4]
//
// 提示：
// 1 <= nums.length <= 200
// 1 <= nums[i] <= 100
// 1 <= k <= 99
func countKDifference(nums []int, k int) int {
	diff := make(map[int]int)
	result := 0
	for _, num := range nums {
		if count, ok := diff[num-k]; ok {
			result += count
		}
		if count, ok := diff[num+k]; ok {
			result += count
		}
		diff[num]++
	}
	return result
}

// 1984. 学生分数的最小差值
// 给你一个 下标从 0 开始 的整数数组 nums ，其中 nums[i] 表示第 i 名学生的分数。另给你一个整数 k 。
//
// 从数组中选出任意 k 名学生的分数，使这 k 个分数间 最高分 和 最低分 的 差值 达到 最小化 。
//
// 返回可能的 最小差值 。
//
// 示例 1：
// 输入：nums = [90], k = 1
// 输出：0
// 解释：选出 1 名学生的分数，仅有 1 种方法：
// - [90] 最高分和最低分之间的差值是 90 - 90 = 0
// 可能的最小差值是 0
//
// 示例 2：
// 输入：nums = [9,4,1,7], k = 2
// 输出：2
// 解释：选出 2 名学生的分数，有 6 种方法：
// - [9,4,1,7] 最高分和最低分之间的差值是 9 - 4 = 5
// - [9,4,1,7] 最高分和最低分之间的差值是 9 - 1 = 8
// - [9,4,1,7] 最高分和最低分之间的差值是 9 - 7 = 2
// - [9,4,1,7] 最高分和最低分之间的差值是 4 - 1 = 3
// - [9,4,1,7] 最高分和最低分之间的差值是 7 - 4 = 3
// - [9,4,1,7] 最高分和最低分之间的差值是 7 - 1 = 6
// 可能的最小差值是 2
//
// 提示：
// 1 <= k <= nums.length <= 1000
// 0 <= nums[i] <= 105
func minimumDifference(nums []int, k int) int {
	if k == 1 {
		return 0
	}
	n := len(nums)
	sort.Ints(nums)
	result := nums[k-1] - nums[0]
	for i := k; i < n; i++ {
		result = min(result, nums[i]-nums[i-k+1])
	}
	return result
}

// 689. 三个无重叠子数组的最大和
// 给你一个整数数组 nums 和一个整数 k ，找出三个长度为 k 、互不重叠、且全部数字和（3 * k 项）最大的子数组，并返回这三个子数组。
//
// 以下标的数组形式返回结果，数组中的每一项分别指示每个子数组的起始位置（下标从 0 开始）。如果有多个结果，返回字典序最小的一个。
//
// 示例 1：
// 输入：nums = [1,2,1,2,6,7,5,1], k = 2
// 输出：[0,3,5]
// 解释：子数组 [1, 2], [2, 6], [7, 5] 对应的起始下标为 [0, 3, 5]。
// 也可以取 [2, 1], 但是结果 [1, 3, 5] 在字典序上更大。
//
// 示例 2：
// 输入：nums = [1,2,1,2,1,2,1,2,1], k = 2
// 输出：[0,2,4]
//
// 提示：
// 1 <= nums.length <= 2 * 104
// 1 <= nums[i] < 216
// 1 <= k <= floor(nums.length / 3)
func maxSumOfThreeSubarrays(nums []int, k int) []int {
	n := len(nums) - k + 1
	sums := make([]int, n)
	result := []int{-1, -1, -1}
	sum := 0
	for i := 0; i < k; i++ {
		sum += nums[i]
	}
	// sums[i] = num[i] + ...+ num[i +k -1]
	sums[0] = sum
	for i := 0; i < n-1; i++ {
		sum += nums[i+k] - nums[i]
		sums[i+1] = sum
	}
	// 左边最大 和 右边最大
	left, right := make([]int, n), make([]int, n)
	for i := 1; i < n; i++ {
		idx := left[i-1]
		if sums[i] > sums[idx] {
			idx = i
		}
		left[i] = idx
	}
	right[n-1] = n - 1
	for i := n - 2; i >= 0; i-- {
		idx := right[i+1]
		// 左边 优先
		if sums[i] >= sums[idx] {
			idx = i
		}
		right[i] = idx
	}

	// 取 中间
	for i := k; i < n-k; i++ {
		leftIdx, rightIdx := left[i-k], right[i+k]
		if result[0] == -1 || sums[leftIdx]+sums[i]+sums[rightIdx] >
			sums[result[0]]+sums[result[1]]+sums[result[2]] {
			result[0] = leftIdx
			result[1] = i
			result[2] = rightIdx
		}
	}

	return result
}

// 699. 掉落的方块
// 在无限长的数轴（即 x 轴）上，我们根据给定的顺序放置对应的正方形方块。
// 第 i 个掉落的方块（positions[i] = (left, side_length)）是正方形，其中 left 表示该方块最左边的点位置(positions[i][0])，side_length 表示该方块的边长(positions[i][1])。
// 每个方块的底部边缘平行于数轴（即 x 轴），并且从一个比目前所有的落地方块更高的高度掉落而下。在上一个方块结束掉落，并保持静止后，才开始掉落新方块。
// 方块的底边具有非常大的粘性，并将保持固定在它们所接触的任何长度表面上（无论是数轴还是其他方块）。邻接掉落的边不会过早地粘合在一起，因为只有底边才具有粘性。
//
// 返回一个堆叠高度列表 ans 。每一个堆叠高度 ans[i] 表示在通过 positions[0], positions[1], ..., positions[i] 表示的方块掉落结束后，目前所有已经落稳的方块堆叠的最高高度。
//
// 示例 1:
// 输入: [[1, 2], [2, 3], [6, 1]]
// 输出: [2, 5, 5]
// 解释:
// 第一个方块 positions[0] = [1, 2] 掉落：
// _aa
// _aa
// -------
// 方块最大高度为 2 。
//
// 第二个方块 positions[1] = [2, 3] 掉落：
// __aaa
// __aaa
// __aaa
// _aa__
// _aa__
// --------------
// 方块最大高度为5。
// 大的方块保持在较小的方块的顶部，不论它的重心在哪里，因为方块的底部边缘有非常大的粘性。
//
// 第三个方块 positions[1] = [6, 1] 掉落：
// __aaa
// __aaa
// __aaa
// _aa
// _aa___a
// --------------
// 方块最大高度为5。
//
// 因此，我们返回结果[2, 5, 5]。
//
// 示例 2:
// 输入: [[100, 100], [200, 100]]
// 输出: [100, 100]
// 解释: 相邻的方块不会过早地卡住，只有它们的底部边缘才能粘在表面上。
//
// 注意:
// 1 <= positions.length <= 1000.
// 1 <= positions[i][0] <= 10^8.
// 1 <= positions[i][1] <= 10^6.
func fallingSquares(positions [][]int) []int {
	n := len(positions)
	result := make([]int, n)
	maxHeight := positions[0][1]
	result[0] = maxHeight
	if n == 1 {
		return result
	}
	heights := make([]int, n)
	heights[0] = maxHeight

	for i := 1; i < n; i++ {
		height := positions[i][1]
		for j := i - 1; j >= 0; j-- {
			if positions[j][0]+positions[j][1] <= positions[i][0] ||
				positions[i][0]+positions[i][1] <= positions[j][0] {
				continue
			}
			height = max(height, heights[j]+positions[i][1])
		}
		heights[i] = height
		maxHeight = max(maxHeight, height)
		result[i] = maxHeight
	}

	return result
}

// 838. 推多米诺
// n 张多米诺骨牌排成一行，将每张多米诺骨牌垂直竖立。在开始时，同时把一些多米诺骨牌向左或向右推。
//
// 每过一秒，倒向左边的多米诺骨牌会推动其左侧相邻的多米诺骨牌。同样地，倒向右边的多米诺骨牌也会推动竖立在其右侧的相邻多米诺骨牌。
//
// 如果一张垂直竖立的多米诺骨牌的两侧同时有多米诺骨牌倒下时，由于受力平衡， 该骨牌仍然保持不变。
//
// 就这个问题而言，我们会认为一张正在倒下的多米诺骨牌不会对其它正在倒下或已经倒下的多米诺骨牌施加额外的力。
//
// 给你一个字符串 dominoes 表示这一行多米诺骨牌的初始状态，其中：
//
// dominoes[i] = 'L'，表示第 i 张多米诺骨牌被推向左侧，
// dominoes[i] = 'R'，表示第 i 张多米诺骨牌被推向右侧，
// dominoes[i] = '.'，表示没有推动第 i 张多米诺骨牌。
// 返回表示最终状态的字符串。
//
// 示例 1：
// 输入：dominoes = "RR.L"
// 输出："RR.L"
// 解释：第一张多米诺骨牌没有给第二张施加额外的力。
//
// 示例 2：
// 输入：dominoes = ".L.R...LR..L.."
// 输出："LL.RR.LLRRLL.."
//
// 提示：
// n == dominoes.length
// 1 <= n <= 105
// dominoes[i] 为 'L'、'R' 或 '.'
func pushDominoes(dominoes string) string {
	n := len(dominoes)

	result := make([]byte, n)
	nums := make([]int, n)
	for i := 0; i < n; i++ {
		if dominoes[i] == 'R' {
			num := 100000
			nums[i] += num
			num--
			for i+1 < n && dominoes[i+1] == '.' {
				nums[i+1] += num
				num--
				i++
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		if dominoes[i] == 'L' {
			num := -100000
			nums[i] += num
			num++
			for i-1 >= 0 && dominoes[i-1] == '.' {
				nums[i-1] += num
				num++
				i--
			}
		}
	}

	for i := 0; i < n; i++ {
		if nums[i] > 0 {
			result[i] = 'R'
		} else if nums[i] < 0 {
			result[i] = 'L'
		} else {
			result[i] = '.'
		}
	}

	return string(result)
}

// 713. 乘积小于K的子数组
// 给定一个正整数数组 nums和整数 k 。
//
// 请找出该数组内乘积小于 k 的连续的子数组的个数。
//
// 示例 1:
// 输入: nums = [10,5,2,6], k = 100
// 输出: 8
// 解释: 8个乘积小于100的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
// 需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
//
// 示例 2:
// 输入: nums = [1,2,3], k = 0
// 输出: 0
//
// 提示:
// 1 <= nums.length <= 3 * 104
// 1 <= nums[i] <= 1000
// 0 <= k <= 106
func numSubarrayProductLessThanK(nums []int, k int) int {
	if k <= 1 {
		return 0
	}
	n := len(nums)
	left := 0
	count := 0
	prod := 1
	for right := 0; right < n; right++ {
		prod *= nums[right]
		for left < n && prod >= k {
			prod /= nums[left]
			left++
		}

		count += right - left + 1
	}
	return count
}

// 2016. 增量元素之间的最大差值
// 给你一个下标从 0 开始的整数数组 nums ，该数组的大小为 n ，请你计算 nums[j] - nums[i] 能求得的 最大差值 ，其中 0 <= i < j < n 且 nums[i] < nums[j] 。
//
// 返回 最大差值 。如果不存在满足要求的 i 和 j ，返回 -1 。
//
// 示例 1：
// 输入：nums = [7,1,5,4]
// 输出：4
// 解释：
// 最大差值出现在 i = 1 且 j = 2 时，nums[j] - nums[i] = 5 - 1 = 4 。
// 注意，尽管 i = 1 且 j = 0 时 ，nums[j] - nums[i] = 7 - 1 = 6 > 4 ，但 i > j 不满足题面要求，所以 6 不是有效的答案。
//
// 示例 2：
// 输入：nums = [9,4,3,2]
// 输出：-1
// 解释：
// 不存在同时满足 i < j 和 nums[i] < nums[j] 这两个条件的 i, j 组合。
//
// 示例 3：
// 输入：nums = [1,5,2,10]
// 输出：9
// 解释：
// 最大差值出现在 i = 0 且 j = 3 时，nums[j] - nums[i] = 10 - 1 = 9 。
//
// 提示：
// n == nums.length
// 2 <= n <= 1000
// 1 <= nums[i] <= 109
func maximumDifference(nums []int) int {
	n := len(nums)
	minVal := nums[0]
	result := -1
	for i := 1; i < n; i++ {
		if nums[i] > minVal {
			result = max(result, nums[i]-minVal)
		}
		minVal = min(minVal, nums[i])
	}
	return result
}

// 905. 按奇偶排序数组
// 给你一个整数数组 nums，将 nums 中的的所有偶数元素移动到数组的前面，后跟所有奇数元素。
//
// 返回满足此条件的 任一数组 作为答案。
//
// 示例 1：
// 输入：nums = [3,1,2,4]
// 输出：[2,4,3,1]
// 解释：[4,2,3,1]、[2,4,1,3] 和 [4,2,1,3] 也会被视作正确答案。
//
// 示例 2：
// 输入：nums = [0]
// 输出：[0]
//
// 提示：
// 1 <= nums.length <= 5000
// 0 <= nums[i] <= 5000
func sortArrayByParity(nums []int) []int {
	n := len(nums)
	left, right := 0, n-1
	for left < right {
		// 左边的奇数
		for left < right && nums[left]&1 == 0 {
			left++
		}
		// 右边的偶数
		for left < right && nums[right]&1 == 1 {
			right--
		}
		// 交换
		if left < right {
			nums[left], nums[right] = nums[right], nums[left]
			left++
			right--
		}

	}
	return nums
}

// 1051. 高度检查器
// 学校打算为全体学生拍一张年度纪念照。根据要求，学生需要按照 非递减 的高度顺序排成一行。
//
// 排序后的高度情况用整数数组 expected 表示，其中 expected[i] 是预计排在这一行中第 i 位的学生的高度（下标从 0 开始）。
//
// 给你一个整数数组 heights ，表示 当前学生站位 的高度情况。heights[i] 是这一行中第 i 位学生的高度（下标从 0 开始）。
// 返回满足 heights[i] != expected[i] 的 下标数量 。
//
// 示例：
// 输入：heights = [1,1,4,2,1,3]
// 输出：3
// 解释：
// 高度：[1,1,4,2,1,3]
// 预期：[1,1,1,2,3,4]
// 下标 2 、4 、5 处的学生高度不匹配。
//
// 示例 2：
// 输入：heights = [5,1,2,3,4]
// 输出：5
// 解释：
// 高度：[5,1,2,3,4]
// 预期：[1,2,3,4,5]
// 所有下标的对应学生高度都不匹配。
//
// 示例 3：
// 输入：heights = [1,2,3,4,5]
// 输出：0
// 解释：
// 高度：[1,2,3,4,5]
// 预期：[1,2,3,4,5]
// 所有下标的对应学生高度都匹配。
//
// 提示：
// 1 <= heights.length <= 100
// 1 <= heights[i] <= 100
func heightChecker(heights []int) int {
	var bucket [101]int
	for _, height := range heights {
		bucket[height]++
	}
	count := 0
	for i, idx := 1, 0; i < 101; i++ {
		for bucket[i] > 0 {
			if heights[idx] != i {
				count++
			}
			idx++
			bucket[i]--
		}
	}

	return count
}

// 1089. 复写零
// 给你一个长度固定的整数数组 arr，请你将该数组中出现的每个零都复写一遍，并将其余的元素向右平移。
//
// 注意：请不要在超过该数组长度的位置写入元素。
//
// 要求：请对输入的数组 就地 进行上述修改，不要从函数返回任何东西。
//
// 示例 1：
// 输入：[1,0,2,3,0,4,5,0]
// 输出：null
// 解释：调用函数后，输入的数组将被修改为：[1,0,0,2,3,0,0,4]
//
// 示例 2：
// 输入：[1,2,3]
// 输出：null
// 解释：调用函数后，输入的数组将被修改为：[1,2,3]
//
// 提示：
// 1 <= arr.length <= 10000
// 0 <= arr[i] <= 9
func duplicateZeros(arr []int) {
	n, count := len(arr), 0
	lastZero := -1
	for i, num := range arr {
		if i+count+1 >= n {
			break
		}
		if num == 0 {
			count++
			lastZero = i
		}
	}
	j := n - 1
	for count > 0 && j > 0 {
		left := j - count
		arr[j] = arr[j-count]
		if arr[j] == 0 && left <= lastZero {
			j--
			arr[j] = 0
			count--
		}
		j--
	}
}

// 1184. 公交站间的距离
// 环形公交路线上有 n 个站，按次序从 0 到 n - 1 进行编号。我们已知每一对相邻公交站之间的距离，distance[i] 表示编号为 i 的车站和编号为 (i + 1) % n 的车站之间的距离。
//
// 环线上的公交车都可以按顺时针和逆时针的方向行驶。
//
// 返回乘客从出发点 start 到目的地 destination 之间的最短距离。
//
// 示例 1：
// 输入：distance = [1,2,3,4], start = 0, destination = 1
// 输出：1
// 解释：公交站 0 和 1 之间的距离是 1 或 9，最小值是 1。
//
// 示例 2：
//
// 输入：distance = [1,2,3,4], start = 0, destination = 2
// 输出：3
// 解释：公交站 0 和 2 之间的距离是 3 或 7，最小值是 3。
//
// 示例 3：
// 输入：distance = [1,2,3,4], start = 0, destination = 3
// 输出：4
// 解释：公交站 0 和 3 之间的距离是 6 或 4，最小值是 4。
//
// 提示：
// 1 <= n <= 10^4
// distance.length == n
// 0 <= start, destination < n
// 0 <= distance[i] <= 10^4
func distanceBetweenBusStops(distance []int, start int, destination int) int {
	result, sum := 0, 0
	if start > destination {
		start, destination = destination, start
	}
	for i, d := range distance {
		sum += d
		if i >= start && i < destination {
			result += d
		}
	}
	return min(result, sum-result)
}

// 1200. 最小绝对差
// 给你个整数数组 arr，其中每个元素都 不相同。
//
// 请你找到所有具有最小绝对差的元素对，并且按升序的顺序返回。
//
// 示例 1：
// 输入：arr = [4,2,1,3]
// 输出：[[1,2],[2,3],[3,4]]
//
// 示例 2：
// 输入：arr = [1,3,6,10,15]
// 输出：[[1,3]]
//
// 示例 3：
// 输入：arr = [3,8,-10,23,19,-4,-14,27]
// 输出：[[-14,-10],[19,23],[23,27]]
//
// 提示：
// 2 <= arr.length <= 10^5
// -10^6 <= arr[i] <= 10^6
func minimumAbsDifference(arr []int) [][]int {
	n := len(arr)
	result := make([][]int, 0)
	sort.Ints(arr)
	minDiff := arr[1] - arr[0]
	for i := 2; i < n; i++ {
		minDiff = min(minDiff, arr[i]-arr[i-1])
	}

	for i := 1; i < n; i++ {
		if arr[i]-arr[i-1] == minDiff {
			result = append(result, []int{arr[i-1], arr[i]})
		}
	}
	return result
}

// 1331. 数组序号转换
// 给你一个整数数组 arr ，请你将数组中的每个元素替换为它们排序后的序号。
//
// 序号代表了一个元素有多大。序号编号的规则如下：
//
// 序号从 1 开始编号。
// 一个元素越大，那么序号越大。如果两个元素相等，那么它们的序号相同。
// 每个数字的序号都应该尽可能地小。
//
// 示例 1：
// 输入：arr = [40,10,20,30]
// 输出：[4,1,2,3]
// 解释：40 是最大的元素。 10 是最小的元素。 20 是第二小的数字。 30 是第三小的数字。
//
// 示例 2：
// 输入：arr = [100,100,100]
// 输出：[1,1,1]
// 解释：所有元素有相同的序号。
//
// 示例 3：
// 输入：arr = [37,12,28,9,100,56,80,5,12]
// 输出：[5,3,4,2,8,6,7,1,3]
//
// 提示：
// 0 <= arr.length <= 105
// -109 <= arr[i] <= 109
func arrayRankTransform(arr []int) []int {
	n := len(arr)
	result := make([]int, n)
	if n == 0 {
		return result
	}
	tmp := make([]int, n)
	copy(tmp, arr)
	sort.Ints(tmp)
	rankMap := make(map[int]int)
	idx := 1
	for _, num := range tmp {
		if _, ok := rankMap[num]; !ok {
			rankMap[num] = idx
			idx++
		}
	}
	for i, num := range arr {
		result[i] = rankMap[num]
	}

	return result
}

// 1403. 非递增顺序的最小子序列
// 给你一个数组 nums，请你从中抽取一个子序列，满足该子序列的元素之和 严格 大于未包含在该子序列中的各元素之和。
//
// 如果存在多个解决方案，只需返回 长度最小 的子序列。如果仍然有多个解决方案，则返回 元素之和最大 的子序列。
//
// 与子数组不同的地方在于，「数组的子序列」不强调元素在原数组中的连续性，也就是说，它可以通过从数组中分离一些（也可能不分离）元素得到。
//
// 注意，题目数据保证满足所有约束条件的解决方案是 唯一 的。同时，返回的答案应当按 非递增顺序 排列。
//
// 示例 1：
// 输入：nums = [4,3,10,9,8]
// 输出：[10,9]
// 解释：子序列 [10,9] 和 [10,8] 是最小的、满足元素之和大于其他各元素之和的子序列。但是 [10,9] 的元素之和最大。
//
// 示例 2：
// 输入：nums = [4,4,7,6,7]
// 输出：[7,7,6]
// 解释：子序列 [7,7] 的和为 14 ，不严格大于剩下的其他元素之和（14 = 4 + 4 + 6）。因此，[7,6,7] 是满足题意的最小子序列。注意，元素按非递增顺序返回。
//
// 示例 3：
// 输入：nums = [6]
// 输出：[6]
//
// 提示：
// 1 <= nums.length <= 500
// 1 <= nums[i] <= 100
func minSubsequence(nums []int) []int {
	sort.Ints(nums)
	sum := 0
	for _, num := range nums {
		sum += num
	}
	n := len(nums)
	minSum := 0
	result := make([]int, 0)
	for i := n - 1; i >= 0; i-- {
		num := nums[i]
		minSum += num
		result = append(result, num)
		if minSum > sum-minSum {
			break
		}
	}
	return result

}

// 1413. 逐步求和得到正数的最小值
// 给你一个整数数组 nums 。你可以选定任意的 正数 startValue 作为初始值。
//
// 你需要从左到右遍历 nums 数组，并将 startValue 依次累加上 nums 数组中的值。
//
// 请你在确保累加和始终大于等于 1 的前提下，选出一个最小的 正数 作为 startValue 。
//
// 示例 1：
// 输入：nums = [-3,2,-3,4,2]
// 输出：5
// 解释：如果你选择 startValue = 4，在第三次累加时，和小于 1 。
//                累加求和
//                startValue = 4 | startValue = 5 | nums
//                  (4 -3 ) = 1  | (5 -3 ) = 2    |  -3
//                  (1 +2 ) = 3  | (2 +2 ) = 4    |   2
//                  (3 -3 ) = 0  | (4 -3 ) = 1    |  -3
//                  (0 +4 ) = 4  | (1 +4 ) = 5    |   4
//                  (4 +2 ) = 6  | (5 +2 ) = 7    |   2
// 示例 2：
// 输入：nums = [1,2]
// 输出：1
// 解释：最小的 startValue 需要是正数。
//
// 示例 3：
// 输入：nums = [1,-2,-3]
// 输出：5
//
// 提示：
// 1 <= nums.length <= 100
// -100 <= nums[i] <= 100
func minStartValue(nums []int) int {
	sum := 0
	minValue := math.MaxInt32
	for _, num := range nums {
		sum += num
		minValue = min(minValue, sum)
	}
	if minValue > 0 {
		return 1
	}
	return 1 - minValue
}

// 769. 最多能完成排序的块
// 给定一个长度为 n 的整数数组 arr ，它表示在 [0, n - 1] 范围内的整数的排列。
//
// 我们将 arr 分割成若干 块 (即分区)，并对每个块单独排序。将它们连接起来后，使得连接的结果和按升序排序后的原数组相同。
//
// 返回数组能分成的最多块数量。
//
// 示例 1:
// 输入: arr = [4,3,2,1,0]
// 输出: 1
// 解释:
// 将数组分成2块或者更多块，都无法得到所需的结果。
// 例如，分成 [4, 3], [2, 1, 0] 的结果是 [3, 4, 0, 1, 2]，这不是有序的数组。
//
// 示例 2:
// 输入: arr = [1,0,2,3,4]
// 输出: 4
// 解释:
// 我们可以把它分成两块，例如 [1, 0], [2, 3, 4]。
// 然而，分成 [1, 0], [2], [3], [4] 可以得到最多的块数。
//
// 提示:
// n == arr.length
// 1 <= n <= 10
// 0 <= arr[i] < n
// arr 中每个元素都 不同
func maxChunksToSorted(arr []int) int {

	result, maxNum := 0, 0
	for i, num := range arr {
		maxNum = max(num, maxNum)
		if maxNum == i {
			result++
		}
	}
	return result
}

// 768. 最多能完成排序的块 II
// 这个问题和“最多能完成排序的块”相似，但给定数组中的元素可以重复，输入数组最大长度为2000，其中的元素最大为10**8。
//
// arr是一个可能包含重复元素的整数数组，我们将这个数组分割成几个“块”，并将这些块分别进行排序。之后再连接起来，使得连接的结果和按升序排序后的原数组相同。
//
// 我们最多能将数组分成多少块？
//
// 示例 1:
// 输入: arr = [5,4,3,2,1]
// 输出: 1
// 解释:
// 将数组分成2块或者更多块，都无法得到所需的结果。
// 例如，分成 [5, 4], [3, 2, 1] 的结果是 [4, 5, 1, 2, 3]，这不是有序的数组。
//
// 示例 2:
// 输入: arr = [2,1,3,4,4]
// 输出: 4
// 解释:
// 我们可以把它分成两块，例如 [2, 1], [3, 4, 4]。
// 然而，分成 [2, 1], [3], [4], [4] 可以得到最多的块数。
//
// 注意:
// arr的长度在[1, 2000]之间。
// arr[i]的大小在[0, 10**8]之间。
func maxChunksToSortedII(arr []int) int {
	// 栈
	stack := list.New()
	for _, num := range arr {
		if stack.Len() > 0 {
			back := stack.Back()
			backNum := back.Value.(int)
			if backNum <= num {
				stack.PushBack(num)
				continue
			}
			stack.Remove(back)
			maxNum := backNum
			for stack.Len() > 0 {
				back = stack.Back()
				backNum = back.Value.(int)
				if backNum <= num {
					break
				}
				stack.Remove(back)
			}
			stack.PushBack(maxNum)
		} else {
			stack.PushBack(num)
		}
	}
	return stack.Len()
}

// 1450. 在既定时间做作业的学生人数
// 给你两个整数数组 startTime（开始时间）和 endTime（结束时间），并指定一个整数 queryTime 作为查询时间。
//
// 已知，第 i 名学生在 startTime[i] 时开始写作业并于 endTime[i] 时完成作业。
//
// 请返回在查询时间 queryTime 时正在做作业的学生人数。形式上，返回能够使 queryTime 处于区间 [startTime[i], endTime[i]]（含）的学生人数。
//
// 示例 1：
// 输入：startTime = [1,2,3], endTime = [3,2,7], queryTime = 4
// 输出：1
// 解释：一共有 3 名学生。
// 第一名学生在时间 1 开始写作业，并于时间 3 完成作业，在时间 4 没有处于做作业的状态。
// 第二名学生在时间 2 开始写作业，并于时间 2 完成作业，在时间 4 没有处于做作业的状态。
// 第三名学生在时间 3 开始写作业，预计于时间 7 完成作业，这是是唯一一名在时间 4 时正在做作业的学生。
//
// 示例 2：
// 输入：startTime = [4], endTime = [4], queryTime = 4
// 输出：1
// 解释：在查询时间只有一名学生在做作业。
//
// 示例 3：
// 输入：startTime = [4], endTime = [4], queryTime = 5
// 输出：0
//
// 示例 4：
// 输入：startTime = [1,1,1,1], endTime = [1,3,2,4], queryTime = 7
// 输出：0
//
// 示例 5：
// 输入：startTime = [9,8,7,6,5,4,3,2,1], endTime = [10,10,10,10,10,10,10,10,10], queryTime = 5
// 输出：5
//
// 提示：
// startTime.length == endTime.length
// 1 <= startTime.length <= 100
// 1 <= startTime[i] <= endTime[i] <= 1000
// 1 <= queryTime <= 1000
func busyStudent(startTime []int, endTime []int, queryTime int) int {
	result := 0
	n := len(startTime)
	for i := 0; i < n; i++ {
		start, end := startTime[i], endTime[i]
		if start <= queryTime && queryTime <= end {
			result++
		}
	}

	return result
}

// 1464. 数组中两元素的最大乘积
// 给你一个整数数组 nums，请你选择数组的两个不同下标 i 和 j，使 (nums[i]-1)*(nums[j]-1) 取得最大值。
//
// 请你计算并返回该式的最大值。
// 示例 1：
// 输入：nums = [3,4,5,2]
// 输出：12
// 解释：如果选择下标 i=1 和 j=2（下标从 0 开始），则可以获得最大值，(nums[1]-1)*(nums[2]-1) = (4-1)*(5-1) = 3*4 = 12 。
//
// 示例 2：
// 输入：nums = [1,5,4,5]
// 输出：16
// 解释：选择下标 i=1 和 j=3（下标从 0 开始），则可以获得最大值 (5-1)*(5-1) = 16 。
//
// 示例 3：
// 输入：nums = [3,7]
// 输出：12
//
// 提示：
// 2 <= nums.length <= 500
// 1 <= nums[i] <= 10^3
func maxProduct(nums []int) int {
	n := len(nums)
	if n == 2 {
		return (nums[0] - 1) * (nums[1] - 1)
	}
	sort.Ints(nums)
	return (nums[n-2] - 1) * (nums[n-1] - 1)
}

// 1470. 重新排列数组
// 给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。
//
// 请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。
//
// 示例 1：
// 输入：nums = [2,5,1,3,4,7], n = 3
// 输出：[2,3,5,4,1,7]
// 解释：由于 x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 ，所以答案为 [2,3,5,4,1,7]
//
// 示例 2：
// 输入：nums = [1,2,3,4,4,3,2,1], n = 4
// 输出：[1,4,2,3,3,2,4,1]
//
// 示例 3：
// 输入：nums = [1,1,2,2], n = 2
// 输出：[1,2,1,2]
//
// 提示：
// 1 <= n <= 500
// nums.length == 2n
// 1 <= nums[i] <= 10^3
func shuffle(nums []int, n int) []int {
	result := make([]int, 2*n)
	for i := 0; i < n; i++ {
		result[i<<1] = nums[i]
		result[(i<<1)+1] = nums[n+i]
	}
	return result
}

// 775. 全局倒置与局部倒置
// 给你一个长度为 n 的整数数组 nums ，表示由范围 [0, n - 1] 内所有整数组成的一个排列。
//
// 全局倒置 的数目等于满足下述条件不同下标对 (i, j) 的数目：
//
// 0 <= i < j < n
// nums[i] > nums[j]
// 局部倒置 的数目等于满足下述条件的下标 i 的数目：
//
// 0 <= i < n - 1
// nums[i] > nums[i + 1]
// 当数组 nums 中 全局倒置 的数量等于 局部倒置 的数量时，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：nums = [1,0,2]
// 输出：true
// 解释：有 1 个全局倒置，和 1 个局部倒置。
//
// 示例 2：
// 输入：nums = [1,2,0]
// 输出：false
// 解释：有 2 个全局倒置，和 1 个局部倒置。
//
// 提示：
// n == nums.length
// 1 <= n <= 5000
// 0 <= nums[i] < n
// nums 中的所有整数 互不相同
// nums 是范围 [0, n - 1] 内所有数字组成的一个排列
func isIdealPermutation(nums []int) bool {
	n := len(nums)
	minVal := math.MaxInt32
	for i := n - 1; i >= 2; i-- {
		minVal = min(minVal, nums[i])
		if nums[i-2] > minVal {
			return false
		}
	}
	return true
}

// 1608. 特殊数组的特征值
// 给你一个非负整数数组 nums 。如果存在一个数 x ，使得 nums 中恰好有 x 个元素 大于或者等于 x ，那么就称 nums 是一个 特殊数组 ，而 x 是该数组的 特征值 。
//
// 注意： x 不必 是 nums 的中的元素。
//
// 如果数组 nums 是一个 特殊数组 ，请返回它的特征值 x 。否则，返回 -1 。可以证明的是，如果 nums 是特殊数组，那么其特征值 x 是 唯一的 。
//
// 示例 1：
// 输入：nums = [3,5]
// 输出：2
// 解释：有 2 个元素（3 和 5）大于或等于 2 。
//
// 示例 2：
// 输入：nums = [0,0]
// 输出：-1
// 解释：没有满足题目要求的特殊数组，故而也不存在特征值 x 。
// 如果 x = 0，应该有 0 个元素 >= x，但实际有 2 个。
// 如果 x = 1，应该有 1 个元素 >= x，但实际有 0 个。
// 如果 x = 2，应该有 2 个元素 >= x，但实际有 0 个。
// x 不能取更大的值，因为 nums 中只有两个元素。
//
// 示例 3：
// 输入：nums = [0,4,3,0,4]
// 输出：3
// 解释：有 3 个元素大于或等于 3 。
//
// 示例 4：
// 输入：nums = [3,6,7,7,0]
// 输出：-1
//
// 提示：
// 1 <= nums.length <= 100
// 0 <= nums[i] <= 1000
func specialArray(nums []int) int {
	n := len(nums)
	sort.Ints(nums)
	if nums[0] >= n {
		return n
	}
	for i := 1; i < n; i++ {
		count := n - i
		if nums[i] >= count && nums[i-1] < count {
			return count
		}
	}
	return -1
}

// 1619. 删除某些元素后的数组均值
// 给你一个整数数组 arr ，请你删除最小 5% 的数字和最大 5% 的数字后，剩余数字的平均值。
//
// 与 标准答案 误差在 10-5 的结果都被视为正确结果。
//
// 示例 1：
// 输入：arr = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]
// 输出：2.00000
// 解释：删除数组中最大和最小的元素后，所有元素都等于 2，所以平均值为 2 。
//
// 示例 2：
// 输入：arr = [6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0]
// 输出：4.00000
//
// 示例 3：
// 输入：arr = [6,0,7,0,7,5,7,8,3,4,0,7,8,1,6,8,1,1,2,4,8,1,9,5,4,3,8,5,10,8,6,6,1,0,6,10,8,2,3,4]
// 输出：4.77778
//
// 示例 4：
// 输入：arr = [9,7,8,7,7,8,4,4,6,8,8,7,6,8,8,9,2,6,0,0,1,10,8,6,3,3,5,1,10,9,0,7,10,0,10,4,1,10,6,9,3,6,0,0,2,7,0,6,7,2,9,7,7,3,0,1,6,1,10,3]
// 输出：5.27778
//
// 示例 5：
// 输入：arr = [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]
// 输出：5.29167
//
// 提示：
// 20 <= arr.length <= 1000
// arr.length 是 20 的 倍数
// 0 <= arr[i] <= 105
func trimMean(arr []int) float64 {
	n := len(arr)
	sort.Ints(arr)
	removeCnt := n / 20
	var sum = 0.0
	count := n - 2*removeCnt
	for i := removeCnt; i < n-removeCnt; i++ {
		sum += float64(arr[i])
	}
	return sum / float64(count)
}

// 1640. 能否连接形成数组
// 给你一个整数数组 arr ，数组中的每个整数 互不相同 。另有一个由整数数组构成的数组 pieces，其中的整数也 互不相同 。请你以 任意顺序 连接 pieces 中的数组以形成 arr 。但是，不允许 对每个数组 pieces[i] 中的整数重新排序。
//
// 如果可以连接 pieces 中的数组形成 arr ，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：arr = [15,88], pieces = [[88],[15]]
// 输出：true
// 解释：依次连接 [15] 和 [88]
//
// 示例 2：
// 输入：arr = [49,18,16], pieces = [[16,18,49]]
// 输出：false
// 解释：即便数字相符，也不能重新排列 pieces[0]
//
// 示例 3：
// 输入：arr = [91,4,64,78], pieces = [[78],[4,64],[91]]
// 输出：true
// 解释：依次连接 [91]、[4,64] 和 [78]
//
// 提示：
// 1 <= pieces.length <= arr.length <= 100
// sum(pieces[i].length) == arr.length
// 1 <= pieces[i].length <= arr.length
// 1 <= arr[i], pieces[i][j] <= 100
// arr 中的整数 互不相同
// pieces 中的整数 互不相同（也就是说，如果将 pieces 扁平化成一维数组，数组中的所有整数互不相同）
func canFormArray(arr []int, pieces [][]int) bool {
	indexMap := make(map[int]int)
	for i, num := range arr {
		indexMap[num] = i
	}
	count := 0
	for _, piece := range pieces {
		index, ok := indexMap[piece[0]]
		if !ok {
			return false
		}
		for i := 1; i < len(piece); i++ {
			idx, ok1 := indexMap[piece[i]]
			if !ok1 || idx != index+i {
				return false
			}
		}
		count += len(piece)
	}

	return count != len(arr)
}

// 786. 第 K 个最小的素数分数
// 给你一个按递增顺序排序的数组 arr 和一个整数 k 。数组 arr 由 1 和若干 素数  组成，且其中所有整数互不相同。
//
// 对于每对满足 0 <= i < j < arr.length 的 i 和 j ，可以得到分数 arr[i] / arr[j] 。
// 那么第 k 个最小的分数是多少呢?  以长度为 2 的整数数组返回你的答案, 这里 answer[0] == arr[i] 且 answer[1] == arr[j] 。
//
// 示例 1：
// 输入：arr = [1,2,3,5], k = 3
// 输出：[2,5]
// 解释：已构造好的分数,排序后如下所示:
// 1/5, 1/3, 2/5, 1/2, 3/5, 2/3
// 很明显第三个最小的分数是 2/5
//
// 示例 2：
// 输入：arr = [1,7], k = 1
// 输出：[1,7]
//
// 提示：
// 2 <= arr.length <= 1000
// 1 <= arr[i] <= 3 * 104
// arr[0] == 1
// arr[i] 是一个 素数 ，i > 0
// arr 中的所有数字 互不相同 ，且按 严格递增 排序
// 1 <= k <= arr.length * (arr.length - 1) / 2
func kthSmallestPrimeFraction(arr []int, k int) []int {
	n := len(arr)
	left, right := 0.0, 1.0
	for {
		mid := (left + right) / 2.0
		i, count := -1, 0
		x, y := 0, 1
		for j := 1; j < n; j++ {
			for float64(arr[i+1])/float64(arr[j]) < mid {
				i++
				if arr[i]*y > arr[j]*x {
					x, y = arr[i], arr[j]
				}
			}
			count += i + 1
		}
		if count == k {
			return []int{x, y}
		}
		if count < k {
			left = mid
		} else {
			right = mid
		}
	}
}

// 1652. 拆炸弹
// 你有一个炸弹需要拆除，时间紧迫！你的情报员会给你一个长度为 n 的 循环 数组 code 以及一个密钥 k 。
//
// 为了获得正确的密码，你需要替换掉每一个数字。所有数字会 同时 被替换。
//
// 如果 k > 0 ，将第 i 个数字用 接下来 k 个数字之和替换。
// 如果 k < 0 ，将第 i 个数字用 之前 k 个数字之和替换。
// 如果 k == 0 ，将第 i 个数字用 0 替换。
// 由于 code 是循环的， code[n-1] 下一个元素是 code[0] ，且 code[0] 前一个元素是 code[n-1] 。
//
// 给你 循环 数组 code 和整数密钥 k ，请你返回解密后的结果来拆除炸弹！
//
// 示例 1：
// 输入：code = [5,7,1,4], k = 3
// 输出：[12,10,16,13]
// 解释：每个数字都被接下来 3 个数字之和替换。解密后的密码为 [7+1+4, 1+4+5, 4+5+7, 5+7+1]。注意到数组是循环连接的。
//
// 示例 2：
// 输入：code = [1,2,3,4], k = 0
// 输出：[0,0,0,0]
// 解释：当 k 为 0 时，所有数字都被 0 替换。
//
// 示例 3：
// 输入：code = [2,4,9,3], k = -2
// 输出：[12,5,6,13]
// 解释：解密后的密码为 [3+9, 2+3, 4+2, 9+4] 。注意到数组是循环连接的。如果 k 是负数，那么和为 之前 的数字。
//
// 提示：
// n == code.length
// 1 <= n <= 100
// 1 <= code[i] <= 100
// -(n - 1) <= k <= n - 1
func decrypt(code []int, k int) []int {
	n := len(code)
	result := make([]int, n)
	if k == 0 {
		return result
	}
	num := 0
	if k > 0 {
		for i := 0; i < k; i++ {
			num += code[i]
		}
		next := k - 1
		for i := 0; i < n; i++ {
			num -= code[i]
			next++
			next %= n
			num += code[next]
			result[i] = num
		}
	} else {
		for i := n + k; i < n; i++ {
			num += code[i]
		}
		last := n + k
		for i := 0; i < n; i++ {
			result[i] = num
			num += code[i]
			num -= code[last]
			last++
			last %= n
		}
	}
	return result
}

// 面试题 17.09. 第 k 个数
// 有些数的素因子只有 3，5，7，请设计一个算法找出第 k 个数。注意，不是必须有这些素因子，而是必须不包含其他的素因子。例如，前几个数按顺序应该是 1，3，5，7，9，15，21。
//
// 示例 1:
// 输入: k = 5
// 输出: 9
func getKthMagicNumber(k int) int {
	idx1, idx2, idx3 := 0, 0, 0
	nums := make([]int, k)
	nums[0] = 1
	for i := 1; i < k; i++ {
		nums[i] = min(nums[idx1]*3, nums[idx2]*5, nums[idx3]*7)
		if nums[i] == nums[idx1]*3 {
			idx1++
		}
		if nums[i] == nums[idx2]*5 {
			idx2++
		}
		if nums[i] == nums[idx3]*7 {
			idx3++
		}
	}

	return nums[k-1]
}

// 1800. 最大升序子数组和
// 给你一个正整数组成的数组 nums ，返回 nums 中一个 升序 子数组的最大可能元素和。
//
// 子数组是数组中的一个连续数字序列。
//
// 已知子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，若对所有 i（l <= i < r），numsi < numsi+1 都成立，则称这一子数组为 升序 子数组。注意，大小为 1 的子数组也视作 升序 子数组。
//
//  示例 1：
// 输入：nums = [10,20,30,5,10,50]
// 输出：65
// 解释：[5,10,50] 是元素和最大的升序子数组，最大元素和为 65 。
//
// 示例 2：
// 输入：nums = [10,20,30,40,50]
// 输出：150
// 解释：[10,20,30,40,50] 是元素和最大的升序子数组，最大元素和为 150 。
//
// 示例 3：
// 输入：nums = [12,17,15,13,10,11,12]
// 输出：33
// 解释：[10,11,12] 是元素和最大的升序子数组，最大元素和为 33 。
//
// 示例 4：
// 输入：nums = [100,10,1]
// 输出：100
//
// 提示：
// 1 <= nums.length <= 100
// 1 <= nums[i] <= 100
func maxAscendingSum(nums []int) int {
	n := len(nums)
	dp := make([]int, n)
	dp[0] = nums[0]
	result := dp[0]
	for i := 1; i < n; i++ {
		if nums[i] > nums[i-1] {
			dp[i] = nums[i] + dp[i-1]
		} else {
			dp[i] = nums[i]
		}
		result = max(result, dp[i])
	}

	return result
}

// 870. 优势洗牌
// 给定两个大小相等的数组 nums1 和 nums2，nums1 相对于 nums 的优势可以用满足 nums1[i] > nums2[i] 的索引 i 的数目来描述。
//
// 返回 nums1 的任意排列，使其相对于 nums2 的优势最大化。
//
// 示例 1：
// 输入：nums1 = [2,7,11,15], nums2 = [1,10,4,11]
// 输出：[2,11,7,15]
//
// 示例 2：
// 输入：nums1 = [12,24,8,32], nums2 = [13,25,32,11]
// 输出：[24,32,8,12]
//
// 提示：
// 1 <= nums1.length <= 105
// nums2.length == nums1.length
// 0 <= nums1[i], nums2[i] <= 109
func advantageCount(nums1 []int, nums2 []int) []int {
	n := len(nums1)
	result := make([]int, n)
	sort.Ints(nums1)
	nums := make([][]int, n)
	for i := 0; i < n; i++ {
		nums[i] = make([]int, 2)
		nums[i][0] = nums2[i]
		nums[i][1] = i
	}
	sort.Slice(nums, func(i, j int) bool {
		return nums[i][0] < nums[j][0]
	})
	left, right := 0, n-1
	for _, num := range nums1 {
		if num <= nums[left][0] {
			result[nums[right][1]] = num // 下等马对上等马
			right--
		} else {
			result[nums[left][1]] = num // 下等马对下等马
			left++
		}
	}

	return result
}

// 904. 水果成篮
// 你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树上的水果 种类 。
//
// 你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：
//
// 你只有 两个 篮子，并且每个篮子只能装 单一类型 的水果。每个篮子能够装的水果总量没有限制。
// 你可以选择任意一棵树开始采摘，你必须从 每棵 树（包括开始采摘的树）上 恰好摘一个水果 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
// 一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
// 给你一个整数数组 fruits ，返回你可以收集的水果的 最大 数目。
//
// 示例 1：
// 输入：fruits = [1,2,1]
// 输出：3
// 解释：可以采摘全部 3 棵树。
//
// 示例 2：
// 输入：fruits = [0,1,2,2]
// 输出：3
// 解释：可以采摘 [1,2,2] 这三棵树。
// 如果从第一棵树开始采摘，则只能采摘 [0,1] 这两棵树。
//
// 示例 3：
// 输入：fruits = [1,2,3,2,2]
// 输出：4
// 解释：可以采摘 [2,3,2,2] 这四棵树。
// 如果从第一棵树开始采摘，则只能采摘 [1,2] 这两棵树。
//
// 示例 4：
// 输入：fruits = [3,3,3,1,2,1,1,2,3,3,4]
// 输出：5
// 解释：可以采摘 [1,2,1,1,2] 这五棵树。
//
// 提示：
// 1 <= fruits.length <= 105
// 0 <= fruits[i] < fruits.length
func totalFruit(fruits []int) int {
	n := len(fruits)
	if n <= 2 {
		return n
	}
	result := 0
	// 第一种
	first := fruits[0]
	index := 1
	for index < n && fruits[index] == first {
		index++
	}
	if index == n {
		return n
	}
	second := fruits[index]
	index++
	start := 0
	for ; index < n; index++ {
		// 第三种水果
		if fruits[index] != first && fruits[index] != second {
			result = max(result, index-start)
			first = fruits[index-1]
			second = fruits[index]
			start = index - 1
			// 找到 前一种水果的开始位置
			for fruits[start-1] == first {
				start--
			}
		}
	}
	result = max(result, index-start)
	return result
}

// 886. 可能的二分法
// 给定一组 n 人（编号为 1, 2, ..., n）， 我们想把每个人分进任意大小的两组。每个人都可能不喜欢其他人，那么他们不应该属于同一组。
//
// 给定整数 n 和数组 dislikes ，其中 dislikes[i] = [ai, bi] ，表示不允许将编号为 ai 和  bi的人归入同一组。当可以用这种方法将所有人分进两组时，返回 true；否则返回 false。
//
// 示例 1：
// 输入：n = 4, dislikes = [[1,2],[1,3],[2,4]]
// 输出：true
// 解释：group1 [1,4], group2 [2,3]
//
// 示例 2：
// 输入：n = 3, dislikes = [[1,2],[1,3],[2,3]]
// 输出：false
//
// 示例 3：
// 输入：n = 5, dislikes = [[1,2],[2,3],[3,4],[4,5],[1,5]]
// 输出：false
//
// 提示：
// 1 <= n <= 2000
// 0 <= dislikes.length <= 104
// dislikes[i].length == 2
// 1 <= dislikes[i][j] <= n
// ai < bi
// dislikes 中每一组都 不同
func possibleBipartition(n int, dislikes [][]int) bool {
	// 染色
	uncolor, white, black := 0, 1, 2
	relations := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		relations[i] = make([]int, 0)
	}
	for _, dislike := range dislikes {
		a, b := dislike[0], dislike[1]
		relations[a] = append(relations[a], b)
		relations[b] = append(relations[b], a)
	}
	colors := make([]int, n+1)

	var dfs func(num int) bool

	dfs = func(num int) bool {
		if colors[num] == uncolor {
			colors[num] = white
		}
		relation := relations[num]
		dislike := white
		if colors[num] == white {
			dislike = black
		}
		for _, p := range relation {
			if colors[p] == uncolor {
				colors[p] = dislike
				if !dfs(p) {
					return false
				}
			} else if colors[p] != dislike {
				return false
			}
		}
		return true
	}

	for i := 1; i <= n; i++ {
		if !dfs(i) {
			return false
		}
	}
	return true
}
