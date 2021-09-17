package arrays

import (
	"fmt"
	"log"
	"math"
	"math/bits"
	"sort"
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
		sum := numbers[left] + numbers[target]
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

	result := []int{}
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
	max, count := 0, 0
	for _, num := range nums {
		if num != 0 {
			count = 0
		} else {
			count++
			if count > max {
				max = count
			}
		}
	}
	return max
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
func findRelativeRanks(nums []int) []string {
	// 所有运动员的成绩都不相同
	indexMap := make(map[int]int)
	for i, num := range nums {
		indexMap[num] = i
	}
	sort.Ints(nums)
	size := len(nums)
	result := make([]string, size)
	var getRank = func(rank int) string {
		switch rank {
		case 1:
			return "Gold Medal"
		case 2:
			return "Silver Medal"
		case 3:
			return "Bronze Medal"
		default:
			return fmt.Sprintf("%d", rank)
		}
	}
	for i, num := range nums {
		rank := size - i
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
func matrixReshape(nums [][]int, r int, c int) [][]int {
	rows, cols := len(nums), len(nums[0])
	if rows*cols != r*c {
		return nums
	}
	var result = make([][]int, r)
	for i := 0; i < r; i++ {
		result[i] = make([]int, c)
	}
	rowIndex, colIndex := 0, 0
	for _, rowNum := range nums {
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
	max := sum
	for i := k; i < size; i++ {
		sum += nums[i] - nums[i-k]
		if sum > max {
			max = sum
		}
	}
	return float64(max) / float64(k)
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
func imageSmoother(M [][]int) [][]int {
	rows, cols := len(M), len(M[0])

	result := make([][]int, rows)
	for i := 0; i < rows; i++ {
		nums := make([]int, cols)
		for j := 0; j < cols; j++ {
			var count, sum = 1, M[i][j]
			left, right, up, down := false, false, false, false
			if i-1 >= 0 {
				count++
				up = true
				sum += M[i-1][j]
			}
			if j-1 >= 0 {
				count++
				left = true
				sum += M[i][j-1]
			}
			if i+1 < rows {
				count++
				down = true
				sum += M[i+1][j]
			}
			if j+1 < cols {
				count++
				right = true
				sum += M[i][j+1]
			}

			if left && up {
				count++
				sum += M[i-1][j-1]
			}
			if left && down {
				count++
				sum += M[i+1][j-1]
			}
			if right && up {
				count++
				sum += M[i-1][j+1]
			}
			if right && down {
				count++
				sum += M[i+1][j+1]
			}
			result[i][j] = sum / count
		}
		result[i] = nums
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
	size := len(nums)
	if size <= 2 {
		return true
	}

	count := 0
	for i := 1; i < size; i++ {
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
	max, count := 1, 1
	for i := 1; i < size; i++ {
		if nums[i] > nums[i-1] {
			count++
		} else {
			count = 1
		}
		if count > max {
			max = count
		}
	}
	return max
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
	size := len(nums)
	max, result := 0, size
	countMap := map[int]int{}
	indexMap := map[int]int{}
	for i, num := range nums {
		count := countMap[num]
		if count == 0 {
			indexMap[num] = i
		}
		count++
		countMap[num] = count
		length := i - indexMap[num] + 1
		if count > max {
			max = count
			result = length
		} else if count == max && length < result {
			result = length
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

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
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
		count := 0
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
