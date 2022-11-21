package dp

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// 动态规划

// 70. 爬楼梯
// 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
//
// 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
//
// 注意：给定 n 是一个正整数。
//
// 示例 1：输入： 2 输出： 2
// 解释： 有两种方法可以爬到楼顶。
// 1.  1 阶 + 1 阶
// 2.  2 阶
// 示例 2：输入： 3 输出： 3
// 解释： 有三种方法可以爬到楼顶。
// 1.  1 阶 + 1 阶 + 1 阶
// 2.  1 阶 + 2 阶
// 3.  2 阶 + 1 阶
func climbStairs(n int) int {
	dp := make([]int, n+1)
	if n <= 2 {
		return n
	}
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

// 121. 买卖股票的最佳时机
// 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
//
// 如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
//
// 注意：你不能在买入股票前卖出股票。
// 示例 1: 输入: [7,1,5,3,6,4] 输出: 5
// 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
//     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
// 示例 2: 输入: [7,6,4,3,1] 输出: 0
// 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
func maxProfit(prices []int) int {

	min, max := (1<<31)-1, 0

	for _, price := range prices {
		if max < price-min {
			max = price - min
		}
		if price < min {
			min = price
		}
	}

	return max
}

// 198. 打家劫舍
// 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
//
// 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
// 示例 1：
// 输入：[1,2,3,1] 输出：4
// 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
//     偷窃到的最高金额 = 1 + 3 = 4 。
// 示例 2：
// 输入：[2,7,9,3,1] 输出：12
// 解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
//     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
// 提示：0 <= nums.length <= 100 0 <= nums[i] <= 400
func rob(nums []int) int {
	l := len(nums)
	dp := make([]int, l)
	if l == 0 {
		return 0
	}
	if l == 1 {
		return nums[0]
	}
	dp[0] = nums[0]
	for i := 1; i < l; i++ {
		dp[i] = nums[i]
		if i > 1 {
			dp[i] += dp[i-2]
		}

		if dp[i-1] > dp[i] {
			dp[i] = dp[i-1]
		}
	}

	return dp[l-1]
}

// 392. 判断子序列
// 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
//
// 你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。
//
// 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
//
// 示例 1: s = "abc", t = "ahbgdc"
// 返回 true.
//
// 示例 2: s = "axc", t = "ahbgdc"
// 返回 false.
//
// 后续挑战 :
//
// 如果有大量输入的 S，称作S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
//
// 致谢:
//
// 特别感谢 @pbrother 添加此问题并且创建所有测试用例。
func isSubsequence(s string, t string) bool {
	l1, l2 := len(s), len(t)
	i, j := 0, 0
	for i < l1 && j < l2 {
		if s[i] == t[j] {
			i++
		}
		j++
	}

	return i == l1
}

// 746. 使用最小花费爬楼梯
// 数组的每个索引作为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
//
// 每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
//
// 您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。
//
// 示例 1:
//
// 输入: cost = [10, 15, 20] 输出: 15
// 解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
// 示例 2:
// 输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1] 输出: 6
// 解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。
// 注意：
//
// cost 的长度将会在 [2, 1000]。
// 每一个 cost[i] 将会是一个Integer类型，范围为 [0, 999]。
func minCostClimbingStairs(cost []int) int {
	size := len(cost)
	dp := make([]int, size)
	if size == 1 {
		return cost[0]
	}
	dp[0] = cost[0]
	dp[1] = cost[1]
	for i := 2; i < size; i++ {
		dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
	}
	return min(dp[size-1], dp[size-2])
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
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

// 213. 打家劫舍 II
// 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
//
// 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
//
// 示例 1：
// 输入：nums = [2,3,2] 输出：3
// 解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
//
// 示例 2：
// 输入：nums = [1,2,3,1] 输出：4
// 解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
//     偷窃到的最高金额 = 1 + 3 = 4 。
//
// 示例 3：
// 输入：nums = [0] 输出：0
//
// 提示：
// 1 <= nums.length <= 100
// 0 <= nums[i] <= 1000
func robII(nums []int) int {
	size := len(nums)
	if size == 0 {
		return 0
	}
	if size == 1 {
		return nums[0]
	}
	// dp := make([]int, size)
	myRob := func(arr []int) int {
		if len(arr) == 0 {
			return 0
		}
		if len(arr) == 1 {
			return arr[0]
		}
		// lastNum 前一房间 currNum 当前房间
		lastNum, currNum := 0, 0
		for _, num := range arr {
			lastNum, currNum = currNum, max(lastNum+num, currNum)
		}

		return currNum
	}
	return max(myRob(nums[:size-1]), myRob(nums[1:]))
}

// 10. 正则表达式匹配
// 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
//
// '.' 匹配任意单个字符
// '*' 匹配零个或多个前面的那一个元素
// 所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
//
// 示例 1：
// 输入：s = "aa" p = "a" 输出：false
// 解释："a" 无法匹配 "aa" 整个字符串。
//
// 示例 2:
// 输入：s = "aa" p = "a*" 输出：true
// 解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
//
// 示例 3：
// 输入：s = "ab" p = ".*" 输出：true
// 解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
//
// 示例 4：
// 输入：s = "aab" p = "c*a*b" 输出：true
// 解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
//
// 示例 5：
// 输入：s = "mississippi" p = "mis*is*p*." 输出：false
//
// 提示：
// 0 <= s.length <= 20
// 0 <= p.length <= 30
// s 可能为空，且只包含从 a-z 的小写字母。
// p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
// 保证每次出现字符 * 时，前面都匹配到有效的字符
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]bool, n+1)
	}
	dp[m][n] = true
	// p 的 最后 匹配 *  * 匹配为0个元素
	for j := n - 2; j >= 0; j-- {
		dp[m][j] = p[j+1] == '*' && dp[m][j+2]
	}

	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			c1, c2 := s[i], p[j]
			match := c1 == c2 || c2 == '.'
			// 后面是 *
			if j < n-1 && p[j+1] == '*' {
				// dp[i][j + 2] * 匹配为0个元素
				// dp[i + 1][j]
				dp[i][j] = dp[i][j+2] || (match && dp[i+1][j])
			} else {
				dp[i][j] = match && dp[i+1][j+1]
			}
		}
	}

	return dp[0][0]
}

// 32. 最长有效括号
// 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
//
// 示例 1：
// 输入：s = "(()" 输出：2
// 解释：最长有效括号子串是 "()"
//
// 示例 2：
// 输入：s = ")()())" 输出：4
// 解释：最长有效括号子串是 "()()"
//
// 示例 3：
// 输入：s = "" 输出：0
//
// 提示：
// 0 <= s.length <= 3 * 104
// s[i] 为 '(' 或 ')'
func longestValidParentheses(s string) int {
	size := len(s)

	dp := make([]int, size)
	result := 0
	for i := 1; i < size; i++ {
		c := s[i]
		if c == ')' {
			if s[i-1] == '(' {
				lastLen := 0
				if i > 2 {
					lastLen = dp[i-2]
				}
				dp[i] = lastLen + 2
			} else {
				lastLeft := i - dp[i-1] - 1
				if lastLeft >= 0 && s[lastLeft] == '(' {
					lastLen := 0
					if lastLeft-1 >= 0 {
						lastLen = dp[lastLeft-1]
					}
					dp[i] = dp[i-1] + 2 + lastLen
				}
			}
			result = max(dp[i], result)
		}
	}

	return result
}

// 45. 跳跃游戏 II
// 给定一个非负整数数组，你最初位于数组的第一个位置。
//
// 数组中的每个元素代表你在该位置可以跳跃的最大长度。
// 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
// 假设你总是可以到达数组的最后一个位置。
//
// 示例 1:
// 输入: [2,3,1,1,4] 输出: 2
// 解释: 跳到最后一个位置的最小跳跃数是 2。
//     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
//
// 示例 2:
// 输入: [2,3,0,1,4] 输出: 2
//
// 提示:
// 1 <= nums.length <= 1000
// 0 <= nums[i] <= 105
func jump(nums []int) int {
	n := len(nums)
	start, end, step := 0, 0, 0

	for end < n-1 {
		maxJump := end
		for i := start; i <= end; i++ {
			if nums[i]+i > maxJump {

			}
			maxJump = max(maxJump, nums[i]+i)
		}
		start = end + 1
		end = maxJump
		step++
	}
	return step
}

// 55. 跳跃游戏
// 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
// 数组中的每个元素代表你在该位置可以跳跃的最大长度。
// 判断你是否能够到达最后一个下标。
//
// 示例 1：
// 输入：nums = [2,3,1,1,4] 输出：true
// 解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
//
// 示例 2：
// 输入：nums = [3,2,1,0,4] 输出：false
// 解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
//
// 提示：
// 1 <= nums.length <= 3 * 104
// 0 <= nums[i] <= 105
func canJump(nums []int) bool {
	n := len(nums)
	if n == 1 {
		return true
	}
	if nums[0] >= n {
		return true
	}

	pos := n - 1
	for i := n - 1; i >= 0; i-- {
		if i+nums[i] >= pos {
			pos = i
		}
	}
	return pos == 0
}

// 62. 不同路径
// 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
//
// 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
// 问总共有多少条不同的路径？
//
// 示例 1：
// 输入：m = 3, n = 7 输出：28
//
// 示例 2：
// 输入：m = 3, n = 2 输出：3
// 解释：
// 从左上角开始，总共有 3 条路径可以到达右下角。
// 1. 向右 -> 向下 -> 向下
// 2. 向下 -> 向下 -> 向右
// 3. 向下 -> 向右 -> 向下
//
// 示例 3：
// 输入：m = 7, n = 3 输出：28
//
// 示例 4：
// 输入：m = 3, n = 3 输出：6
//
// 提示：
// 1 <= m, n <= 100
// 题目数据保证答案小于等于 2 * 109
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = 1
	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0]
	}
	for j := 1; j < n; j++ {
		dp[0][j] = dp[0][j-1]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

// 63. 不同路径 II
// 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
// 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
//
// 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
// 网格中的障碍物和空位置分别用 1 和 0 来表示。
//
// 示例 1：
// 输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]] 输出：2
// 解释：
// 3x3 网格的正中间有一个障碍物。
// 从左上角到右下角一共有 2 条不同的路径：
// 1. 向右 -> 向右 -> 向下 -> 向下
// 2. 向下 -> 向下 -> 向右 -> 向右
//
// 示例 2：
// 输入：obstacleGrid = [[0,1],[0,0]] 输出：1
//
// 提示：
// m == obstacleGrid.length
// n == obstacleGrid[i].length
// 1 <= m, n <= 100
// obstacleGrid[i][j] 为 0 或 1
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	if obstacleGrid[0][0] == 1 {
		return 0
	}
	dp[0][0] = 1
	for i := 1; i < m; i++ {
		if obstacleGrid[i][0] == 1 {
			break
		}
		dp[i][0] = dp[i-1][0]
	}
	for j := 1; j < n; j++ {
		if obstacleGrid[0][j] == 1 {
			break
		}
		dp[0][j] = dp[0][j-1]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if obstacleGrid[i][j] == 1 {
				continue
			}
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

// 64. 最小路径和
// 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
// 说明：每次只能向下或者向右移动一步。
//
// 示例 1：
// 输入：grid = [[1,3,1],[1,5,1],[4,2,1]] 输出：7
// 解释：因为路径 1→3→1→1→1 的总和最小。
//
// 示例 2：
// 输入：grid = [[1,2,3],[4,5,6]] 输出：12
//
// 提示：
// m == grid.length
// n == grid[i].length
// 1 <= m, n <= 200
// 0 <= grid[i][j] <= 100
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])

	for i := 1; i < m; i++ {

		grid[i][0] += grid[i-1][0]
	}
	for j := 1; j < n; j++ {
		grid[0][j] += grid[0][j-1]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			grid[i][j] += min(grid[i-1][j], grid[i][j-1])
		}
	}

	return grid[m-1][n-1]
}

// 72. 编辑距离
// 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
//
// 你可以对一个单词进行如下三种操作：
// 插入一个字符
// 删除一个字符
// 替换一个字符
//
// 示例 1：
// 输入：word1 = "horse", word2 = "ros" 输出：3
// 解释：
// horse -> rorse (将 'h' 替换为 'r')
// rorse -> rose (删除 'r')
// rose -> ros (删除 'e')
//
// 示例 2：
// 输入：word1 = "intention", word2 = "execution" 输出：5
// 解释：
// intention -> inention (删除 't')
// inention -> enention (将 'i' 替换为 'e')
// enention -> exention (将 'n' 替换为 'x')
// exention -> exection (将 'n' 替换为 'c')
// exection -> execution (插入 'u')
//
// 提示：
// 0 <= word1.length, word2.length <= 500
// word1 和 word2 由小写英文字母组成
func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	if word1 == word2 {
		return 0
	}
	if m*n == 0 {
		return m + n
	}

	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i <= m; i++ {
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 3 种情况
			// 1 删除 当前元素
			tmpMin := dp[i][j+1] + 1
			// 2 增加 一个元素
			tmpMin = min(tmpMin, dp[i+1][j]+1)
			// 替换
			replace := 1
			if word1[i] == word2[j] {
				replace = 0
			}
			tmpMin = min(tmpMin, dp[i][j]+replace)
			dp[i+1][j+1] = tmpMin
		}
	}

	return dp[m][n]
}

// 91. 解码方法
// 一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：
//
// 'A' -> 1
// 'B' -> 2
// ...
// 'Z' -> 26
// 要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
//
// "AAJF" ，将消息分组为 (1 1 10 6)
// "KJF" ，将消息分组为 (11 10 6)
// 注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
//
// 给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
//
// 题目数据保证答案肯定是一个 32 位 的整数。
//
// 示例 1：
// 输入：s = "12" 输出：2
// 解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
//
// 示例 2：
// 输入：s = "226" 输出：3
// 解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
//
// 示例 3：
// 输入：s = "0" 输出：0
// 解释：没有字符映射到以 0 开头的数字。
// 含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
// 由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
//
// 示例 4：
// 输入：s = "06" 输出：0
// 解释："06" 不能映射到 "F" ，因为字符串含有前导 0（"6" 和 "06" 在映射中并不等价）。
//
// 提示：
// 1 <= s.length <= 100
// s 只包含数字，并且可能包含前导零。
func numDecodings(s string) int {
	n := len(s)
	dp := make([]int, n)
	if s[0] == '0' {
		return 0
	}
	dp[0] = 1

	for i := 1; i < n; i++ {
		if s[i] > '0' {
			dp[i] += dp[i-1]
		}
		if s[i-1] == '1' || (s[i-1] == '2' && s[i] <= '6') {
			if i-2 >= 0 {
				dp[i] += dp[i-2]
			} else {
				dp[i]++
			}
		}
	}
	return dp[n-1]

}

// 97. 交错字符串
// 给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
//
// 两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：
//
// s = s1 + s2 + ... + sn
// t = t1 + t2 + ... + tm
// |n - m| <= 1
// 交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
// 提示：a + b 意味着字符串 a 和 b 连接。
//
// 示例 1：
// 输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac" 输出：true
//
// 示例 2：
// 输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc" 输出：false
//
// 示例 3：
// 输入：s1 = "", s2 = "", s3 = "" 输出：true
//
// 提示：
// 0 <= s1.length, s2.length <= 100
// 0 <= s3.length <= 200
// s1、s2、和 s3 都由小写英文字母组成
func isInterleave(s1 string, s2 string, s3 string) bool {
	m, n := len(s1), len(s2)
	if m+n != len(s3) {
		return false
	}
	dp := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true

	for i := 0; i < m; i++ {
		dp[i+1][0] = dp[i][0] && s1[i] == s3[i]
	}
	for j := 0; j < n; j++ {
		dp[0][j+1] = dp[0][j] && s2[j] == s3[j]
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			//
			dp[i+1][j+1] = (dp[i][j+1] && s1[i] == s3[i+j+1]) || (dp[i+1][j] && s2[j] == s3[i+j+1])
		}
	}

	return dp[m][n]
}

// 115. 不同的子序列
// 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
//
// 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
//
// 题目数据保证答案符合 32 位带符号整数范围。
//
// 示例 1：
// 输入：s = "rabbbit", t = "rabbit" 输出：3
// 解释：
// 如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
// rabbbit
// rabbbit
// rabbbit
//
// 示例 2：
// 输入：s = "babgbag", t = "bag" 输出：5
// 解释：
// 如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。
// babgbag
// babgbag
// babgbag
// babgbag
// babgbag
//
// 提示：
// 0 <= s.length, t.length <= 1000
// s 和 t 由英文字母组成
func numDistinct(s string, t string) int {
	m, n := len(s), len(t)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i <= m; i++ {
		dp[i][0] = 1
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			dp[i][j] = dp[i-1][j]
			if s[i-1] == t[j-1] {
				dp[i][j] += dp[i-1][j-1]
			}
		}
	}

	return dp[m][n]
}

// 120. 三角形最小路径和
// 给定一个三角形 triangle ，找出自顶向下的最小路径和。
// 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
//
// 示例 1：
// 输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]] 输出：11
// 解释：如下面简图所示：
//    2
//   3 4
//  6 5 7
// 4 1 8 3
// 自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
//
// 示例 2：
// 输入：triangle = [[-10]] 输出：-10
//
// 提示：
// 1 <= triangle.length <= 200
// triangle[0].length == 1
// triangle[i].length == triangle[i - 1].length + 1
// -104 <= triangle[i][j] <= 104
//
// 进阶：
// 你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题吗？
func minimumTotal(triangle [][]int) int {
	n := len(triangle)
	dp := make([]int, n)
	copy(dp, triangle[n-1])
	for i := n - 2; i >= 0; i-- {
		for j := 0; j <= i; j++ {
			dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]
		}

	}
	return dp[0]
}

// 123. 买卖股票的最佳时机 III
//
// 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
// 设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
// 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
//
// 示例 1:
// 输入：prices = [3,3,5,0,0,3,1,4] 输出：6
// 解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
//     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
//
// 示例 2：
// 输入：prices = [1,2,3,4,5] 输出：4
// 解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
//     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
//     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
//
// 示例 3：
// 输入：prices = [7,6,4,3,1]  输出：0
// 解释：在这个情况下, 没有交易完成, 所以最大利润为 0。
//
// 示例 4：
// 输入：prices = [1] 输出：0
//
// 提示：
// 1 <= prices.length <= 105
// 0 <= prices[i] <= 105
func maxProfitIII(prices []int) int {
	size := len(prices)
	// dp0：初始化状态
	// dp1：第一次买入
	// dp2：第一次卖出
	// dp3：第二次买入
	// dp4：第二次卖出
	// dp1 = Math.max(dp1,dp0 - prices[i]);
	// dp2 = Math.max(dp2,dp1 + prices[i]);
	// dp3 = Math.max(dp3,dp2 - prices[i]);
	// dp4 = Math.max(dp4,dp3 + prices[i]);

	dp1, dp2, dp3, dp4 := -prices[0], 0, -prices[0], 0

	for i := 1; i < size; i++ {

		dp4 = max(dp4, dp3+prices[i])
		dp3 = max(dp3, dp2-prices[i])
		dp2 = max(dp2, dp1+prices[i])
		dp1 = max(dp1, -prices[i])
	}
	// 返回最大值
	return dp4
}

// 131. 分割回文串
// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
// 回文串 是正着读和反着读都一样的字符串。
//
// 示例 1：
// 输入：s = "aab" 输出：[["a","a","b"],["aa","b"]]
//
// 示例 2：
// 输入：s = "a" 输出：[["a"]]
//
// 提示：
// 1 <= s.length <= 16
// s 仅由小写英文字母组成
func partition(s string) [][]string {
	result := make([][]string, 0)
	size := len(s)
	//状态：dp[i][j] 表示 s.substring(i,j) 是否是回文
	dp := make([][]bool, size)
	for i := 0; i < size; i++ {
		dp[i] = make([]bool, size)
		dp[i][i] = true
	}

	var back func(start int, part []string)

	checkPalindrome := func(str string, start, end int) bool {
		for start < end {
			if str[start] != str[end] {
				return false
			}
			start++
			end--
		}
		return true
	}

	back = func(start int, part []string) {
		if start == size {
			tmpPart := make([]string, len(part))
			copy(tmpPart, part)
			result = append(result, tmpPart)
			return
		}
		for i := start; i < size; i++ {
			// 不是回文
			if !dp[start][i] && !checkPalindrome(s, start, i) {
				continue
			}
			dp[start][i] = true
			part = append(part, s[start:i+1])
			back(i+1, part)
			part = part[:len(part)-1]
		}
	}

	back(0, make([]string, 0))

	return result
}

// 132. 分割回文串 II
// 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文。
// 返回符合要求的 最少分割次数 。
//
// 示例 1：
// 输入：s = "aab" 输出：1
// 解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
//
// 示例 2：
// 输入：s = "a" 输出：0
//
// 示例 3：
// 输入：s = "ab" 输出：1
//
// 提示：
// 1 <= s.length <= 2000
// s 仅由小写英文字母组成
func minCut(s string) int {
	size := len(s)
	// 预处理, 记录所有的 回文结束位置 i 的 开始位置
	// idxList[i] 表示以s[i] 结尾的回文起始位置left列表
	idxList := make([][]int, size)
	for i := 0; i < size; i++ {
		idxList[i] = make([]int, 0)
	}

	getCutIdx := func(left, right int) {
		for left >= 0 && right < size && s[left] == s[right] {
			idxList[right] = append(idxList[right], left)
			left--
			right++
		}
	}

	for i := 0; i < size; i++ {
		getCutIdx(i, i+1)
		getCutIdx(i-1, i+1)
	}
	dp := make([]int, size+1)
	for i := 0; i < size; i++ {
		dp[i+1] = dp[i] + 1
		for _, idx := range idxList[i] {
			dp[i+1] = min(dp[i+1], dp[idx]+1)
		}
	}

	return dp[size] - 1
}

// 139. 单词拆分
// 给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
//
// 说明：
// 拆分时可以重复使用字典中的单词。
// 你可以假设字典中没有重复的单词。
//
// 示例 1：
// 输入: s = "leetcode", wordDict = ["leet", "code"] 输出: true
// 解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
//
// 示例 2：
// 输入: s = "applepenapple", wordDict = ["apple", "pen"]  输出: true
// 解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
//     注意你可以重复使用字典中的单词。
//
// 示例 3：
// 输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"] 输出: false
func wordBreak(s string, wordDict []string) bool {
	size := len(s)
	dp := make([]bool, size+1)
	dp[0] = true

	for i := 1; i <= size; i++ {
		if dp[i] {
			continue
		}
		for _, word := range wordDict {
			wordLen := len(word)
			if i >= wordLen && dp[i-wordLen] && word == s[i-wordLen:i] {
				dp[i] = true
			}
		}
	}
	return dp[size]
}

// 140. 单词拆分 II
// 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。
//
// 说明：
// 分隔时可以重复使用字典中的单词。
// 你可以假设字典中没有重复的单词。
//
// 示例 1：
// 输入: s = "catsanddog" wordDict = ["cat", "cats", "and", "sand", "dog"]
// 输出:
// [
//  "cats and dog",
//  "cat sand dog"
// ]
//
// 示例 2：
// 输入: s = "pineapplepenapple" wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
// 输出:
// [
//  "pine apple pen apple",
//  "pineapple pen apple",
//  "pine applepen apple"
// ]
// 解释: 注意你可以重复使用字典中的单词。
//
// 示例 3：
// 输入: s = "catsandog" wordDict = ["cats", "dog", "sand", "and", "cat"]
// 输出: []
func wordBreakII(s string, wordDict []string) []string {
	size := len(s)
	dp := make([]bool, size+1)
	dp[0] = true
	dpStr := make([][]string, size+1)
	for i := 0; i <= size; i++ {
		dpStr[i] = make([]string, 0)
	}

	for i := 1; i <= size; i++ {
		for _, word := range wordDict {
			wordLen := len(word)
			if i >= wordLen && dp[i-wordLen] && word == s[i-wordLen:i] {
				dp[i] = true
				if len(dpStr[i-wordLen]) == 0 {
					dpStr[i] = append(dpStr[i], word)
				} else {
					for _, str := range dpStr[i-wordLen] {
						str += " " + word
						dpStr[i] = append(dpStr[i], str)
					}
				}

			}
		}
	}
	fmt.Println(dpStr)
	return dpStr[size]
}

// 152. 乘积最大子数组
// 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
//
// 示例 1:
// 输入: [2,3,-2,4]
// 输出: 6
// 解释: 子数组 [2,3] 有最大乘积 6。
//
// 示例 2:
// 输入: [-2,0,-1]
// 输出: 0
// 解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
func maxProduct(nums []int) int {
	size := len(nums)
	if size == 0 {
		return 0
	}
	if size == 1 {
		return nums[0]
	}
	minNums, maxNums := make([]int, size), make([]int, size)
	minNums[0], maxNums[0] = nums[0], nums[0]

	for i := 1; i < size; i++ {
		maxNums[i] = max(maxNums[i-1]*nums[i], nums[i])
		minNums[i] = min(minNums[i-1]*nums[i], nums[i])
		if nums[i] < 0 {
			if minNums[i-1] < 0 {
				maxNums[i] = max(maxNums[i], minNums[i-1]*nums[i])
			}
			if maxNums[i-1] > 0 {
				minNums[i] = min(maxNums[i-1]*nums[i], minNums[i])
			}

		}
	}
	result := maxNums[0]
	for _, num := range maxNums {
		result = max(result, num)
	}
	return result
}

// 174. 地下城游戏
// 一些恶魔抓住了公主（P）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（K）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。
//
// 骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。
// 有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。
// 为了尽快到达公主，骑士决定每次只向右或向下移动一步。
//
// 编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。
//
// 例如，考虑到如下布局的地下城，如果骑士遵循最佳路径 右 -> 右 -> 下 -> 下，则骑士的初始健康点数至少为 7。
// -2 (K)	-3	3
// -5	-10	1
// 10	30	-5 (P)
//
//
// 说明:
// 骑士的健康点数没有上限。
// 任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间。
func calculateMinimumHP(dungeon [][]int) int {
	m, n := len(dungeon), len(dungeon[0])
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
		for j := 0; j <= n; j++ {
			dp[i][j] = math.MaxInt32
		}
	}
	dp[m-1][n], dp[m][n-1] = 1, 1

	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			minhp := min(dp[i+1][j], dp[i][j+1])

			dp[i][j] = max(minhp-dungeon[i][j], 1)
		}
	}
	return dp[0][0]
}

// 188. 买卖股票的最佳时机 IV
// 给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
//
// 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
// 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
//
// 示例 1：
// 输入：k = 2, prices = [2,4,1] 输出：2
// 解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
//
// 示例 2：
// 输入：k = 2, prices = [3,2,6,5,0,3] 输出：7
// 解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
//     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。
//
// 提示：
// 0 <= k <= 100
// 0 <= prices.length <= 1000
// 0 <= prices[i] <= 1000
func maxProfitIV(k int, prices []int) int {
	n := len(prices)
	if n < 2 {
		return 0
	}
	// dp[i][0] 表示  i 次交易的 利润
	// 0 表示买入, 1 表示卖出
	dp := make([][]int, k+1)
	for i := 0; i <= k; i++ {
		dp[i] = make([]int, 2)
		// 只交易 k - i 次
		dp[i][0] = -prices[0]
	}

	for i := 1; i < n; i++ {
		// 第 j 次交易的最大利润
		for j := k; j > 0; j-- {
			// 前一天买入 今天卖出
			// 前一天 j 次交易卖出的最大利润, 前一天j 次交易买入 今天卖出的利润
			dp[j][1] = max(dp[j][1], dp[j][0]+prices[i])
			// 前一天卖出 今天买入
			// 前一天 j 次交易买入的最大利润, 前一天 j - 1 次交易买出 今天买入的利润
			dp[j][0] = max(dp[j][0], dp[j-1][1]-prices[i])
		}
	}
	return dp[k][1]
}

// 221. 最大正方形
// 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
//
// 示例 1：
// 输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
// 输出：4
//
// 示例 2：
// 输入：matrix = [["0","1"],["1","0"]]
// 输出：1
//
// 示例 3：
// 输入：matrix = [["0"]]
// 输出：0
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 300
// matrix[i][j] 为 '0' 或 '1'
func maximalSquare(matrix [][]byte) int {
	m, n := len(matrix), len(matrix[0])
	sides := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		sides[i] = make([]int, n+1)
	}
	maxSide := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if matrix[i][j] == '1' {
				minSide := min(sides[i][j+1], sides[i+1][j])
				minSide = min(minSide, sides[i][j])
				sides[i+1][j+1] = minSide + 1
				maxSide = max(maxSide, sides[i+1][j+1])
			} else {
				sides[i+1][j+1] = 0
			}
		}
	}

	return maxSide * maxSide
}

// 279. 完全平方数
//
// 给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
// 给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。
// 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
//
// 示例 1：
// 输入：n = 12
// 输出：3
// 解释：12 = 4 + 4 + 4
//
// 示例 2：
// 输入：n = 13
// 输出：2
// 解释：13 = 4 + 9
//
// 提示：
// 1 <= n <= 104
func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 1; i <= n; i++ {
		// i 个 1
		dp[i] = i
		for j := 1; i-j*j >= 0; j++ {
			dp[i] = min(dp[i], 1+dp[i-j*j])
		}
	}
	return dp[n]
}

// 300. 最长递增子序列
// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
//
// 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
//
// 示例 1：
// 输入：nums = [10,9,2,5,3,7,101,18]
// 输出：4
// 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
//
// 示例 2：
// 输入：nums = [0,1,0,3,2,3]
// 输出：4
//
// 示例 3：
// 输入：nums = [7,7,7,7,7,7,7]
// 输出：1
//
// 提示：
// 1 <= nums.length <= 2500
// -104 <= nums[i] <= 104
//
// 进阶：
// 你可以设计时间复杂度为 O(n2) 的解决方案吗？
// 你能将算法的时间复杂度降低到 O(n log(n)) 吗?
func lengthOfLIS(nums []int) int {
	n := len(nums)
	dp := make([]int, n)
	result := 1
	dp[0] = 1
	for i := 1; i < n; i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[j] < nums[i] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		result = max(result, dp[i])
	}

	return result
}

// 309. 最佳买卖股票时机含冷冻期
// 给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。
//
// 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
//
// 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
// 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
//
// 示例:
// 输入: [1,2,3,0,2]
// 输出: 3
// 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
func maxProfitV(prices []int) int {
	n := len(prices)
	if n == 0 {
		return 0
	}
	// dp 表示第i天的利润 0 表示卖出, 1 表示买入
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0], dp[0][1] = 0, -prices[0]
	for i := 1; i < n; i++ {
		// 第i天的利润（卖出） = 第i-1天的利润（卖出） | 第i-1天的利润（买入） + i天的价格
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		// 第i天的利润（买入） = 第i-1天的利润（买入）  第i-2天的利润（卖出） - i天的价格
		oldProfit := 0
		if i >= 2 {
			oldProfit = dp[i-2][0]
		}
		dp[i][1] = max(dp[i-1][1], oldProfit-prices[i])
	}
	return max(dp[n-1][0], dp[n-1][1])
}

// 312. 戳气球
// 有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。
//
// 现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。
//
// 求所能获得硬币的最大数量。
//
// 示例 1：
// 输入：nums = [3,1,5,8]
// 输出：167
// 解释：
// nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
// coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
//
// 示例 2：
// 输入：nums = [1,5]
// 输出：10
//
// 提示：
// n == nums.length
// 1 <= n <= 500
// 0 <= nums[i] <= 100
func maxCoins(nums []int) int {
	n, m := len(nums), len(nums)+2
	newNums := make([]int, m)
	newNums[0], newNums[m-1] = 1, 1
	for i := 0; i < n; i++ {
		newNums[i+1] = nums[i]
	}
	// f(0,len-1) = max {f(0,i) + f(i,len-1) + nums[0]*nums[i]*nums[len-1]}
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, m)
	}
	for i := m - 2; i >= 0; i-- {
		for j := i + 2; j < m; j++ {
			dp[i][j] = 0
			for k := i + 1; k < j; k++ {
				dp[i][j] = max(dp[i][j], dp[i][k]+dp[k][j]+newNums[i]*newNums[k]*newNums[j])
			}
		}
	}
	return dp[0][m-1]
}

// 322. 零钱兑换
// 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
//
// 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
//
// 你可以认为每种硬币的数量是无限的。
//
// 示例 1：
// 输入：coins = [1, 2, 5], amount = 11
// 输出：3
// 解释：11 = 5 + 5 + 1
//
// 示例 2：
// 输入：coins = [2], amount = 3
// 输出：-1
//
// 示例 3：
// 输入：coins = [1], amount = 0
// 输出：0
//
// 示例 4：
// 输入：coins = [1], amount = 1
// 输出：1
//
// 示例 5：
// 输入：coins = [1], amount = 2
// 输出：2
//
// 提示：
// 1 <= coins.length <= 12
// 1 <= coins[i] <= 231 - 1
// 0 <= amount <= 104
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	if amount == 0 {
		return 0
	}

	for i := 1; i <= amount; i++ {
		dp[i] = math.MaxInt64
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		for _, coin := range coins {
			if i-coin < 0 {
				continue
			}
			if dp[i-coin] == math.MaxInt64 {
				continue
			}
			dp[i] = min(dp[i], dp[i-coin]+1)
		}
	}
	if dp[amount] == math.MaxInt64 {
		return -1
	}

	return dp[amount]
}

// 354. 俄罗斯套娃信封问题
// 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
//
// 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
//
// 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
//
// 注意：不允许旋转信封。
//
// 示例 1：
// 输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
// 输出：3
// 解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
//
// 示例 2：
// 输入：envelopes = [[1,1],[1,1],[1,1]]
// 输出：1
//
// 提示：
// 1 <= envelopes.length <= 5000
// envelopes[i].length == 2
// 1 <= wi, hi <= 104
func maxEnvelopes(envelopes [][]int) int {
	// w 升序 h 降序  然后找 h的最长递增子序列
	sort.Slice(envelopes, func(i, j int) bool {
		if envelopes[i][0] == envelopes[j][0] {
			return envelopes[i][1] > envelopes[j][1]
		} else {
			return envelopes[i][0] < envelopes[j][0]
		}
	})
	n := len(envelopes)
	heights := make([]int, n)
	for i := 0; i < n; i++ {
		heights[i] = envelopes[i][1]
	}
	return lengthOfLIS(heights)
}

// 357. 计算各个位数不同的数字个数
// 给定一个非负整数 n，计算各位数字都不同的数字 x 的个数，其中 0 ≤ x < 10n 。
//
// 示例:
// 输入: 2
// 输出: 91
// 解释: 答案应为除去 11,22,33,44,55,66,77,88,99 外，在 [0,100) 区间内的所有数字。
func countNumbersWithUniqueDigits(n int) int {
	dp := make([]int, 11)

	// 排列组合 从10位数字中找出1 ~ 10个数字都不同的组合
	dp[0] = 1
	dp[1] = 9
	// 1 -> 10
	// 2 -> 9 * 9
	// 3 -> 9 * 9 * 8
	// 4 -> 9 * 9 * 8 * 7
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] * (11 - i)
	}
	result := 0
	for i := 0; i <= min(10, n); i++ {
		result += dp[i]
	}
	return result
}

// 368. 最大整除子集
// 给你一个由 无重复 正整数组成的集合 nums ，请你找出并返回其中最大的整除子集 answer ，子集中每一元素对 (answer[i], answer[j]) 都应当满足：
// answer[i] % answer[j] == 0 ，或
// answer[j] % answer[i] == 0
// 如果存在多个有效解子集，返回其中任何一个均可。
//
// 示例 1：
// 输入：nums = [1,2,3] 输出：[1,2]
// 解释：[1,3] 也会被视为正确答案。
//
// 示例 2：
// 输入：nums = [1,2,4,8] 输出：[1,2,4,8]
//
// 提示：
// 1 <= nums.length <= 1000
// 1 <= nums[i] <= 2 * 109
// nums 中的所有整数 互不相同
func largestDivisibleSubset(nums []int) []int {
	// 先算dp 在求集合
	n := len(nums)
	sort.Ints(nums)
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		dp[i] = 1
	}
	maxSize, maxIndex := 1, 0
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if nums[i]%nums[j] == 0 {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		if dp[i] > maxSize {
			maxIndex = i
			maxSize = dp[i]
		}
	}
	// 求集合
	result := make([]int, 0)
	for i := maxIndex; i >= 0; i-- {
		if nums[maxIndex]%nums[i] == 0 && maxSize == dp[i] {
			result = append(result, nums[i])
			maxSize--
			maxIndex = i
		}
	}
	left, right := 0, len(result)-1

	for left < right {
		result[left], result[right] = result[right], result[left]
		left++
		right--
	}

	return result
}

// 375. 猜数字大小 II
// 我们正在玩一个猜数游戏，游戏规则如下：
//
// 我从 1 到 n 之间选择一个数字，你来猜我选了哪个数字。
//
// 每次你猜错了，我都会告诉你，我选的数字比你的大了或者小了。
//
// 然而，当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。直到你猜到我选的数字，你才算赢得了这个游戏。
//
// 示例:
// n = 10, 我选择了8.
// 第一轮: 你猜我选择的数字是5，我会告诉你，我的数字更大一些，然后你需要支付5块。
// 第二轮: 你猜是7，我告诉你，我的数字更大一些，你支付7块。
// 第三轮: 你猜是9，我告诉你，我的数字更小一些，你支付9块。
//
// 游戏结束。8 就是我选的数字。
// 你最终要支付 5 + 7 + 9 = 21 块钱。
// 给定 n ≥ 1，计算你至少需要拥有多少现金才能确保你能赢得这个游戏。
func getMoneyAmount(n int) int {
	// dp[i][j] 表示 i ~ j 直接的坏情况下最小开销的代价
	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := n; i >= 1; i-- {
		for j := i; j <= n; j++ {
			if i == j {
				continue
			}
			dp[i][j] = math.MaxInt32
			for k := i; k <= j; k++ {
				dp[i][j] = min(dp[i][j], k+max(dp[i][k-1], dp[k+1][j]))
			}
		}
	}

	return dp[1][n]
}

// 376. 摆动序列
// 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。
//
// 例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。
//
// 相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
// 子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。
//
// 给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。
//
// 示例 1：
// 输入：nums = [1,7,4,9,2,5] 输出：6
// 解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。
//
// 示例 2：
// 输入：nums = [1,17,5,10,13,15,10,5,16,8] 输出：7
// 解释：这个序列包含几个长度为 7 摆动序列。
// 其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。
//
// 示例 3：
// 输入：nums = [1,2,3,4,5,6,7,8,9] 输出：2
//
// 提示：
// 1 <= nums.length <= 1000
// 0 <= nums[i] <= 1000
// 进阶：你能否用 O(n) 时间复杂度完成此题?
func wiggleMaxLength(nums []int) int {
	n := len(nums)
	if n < 2 {
		return n
	}
	up, down := 1, 1
	for i := 1; i < n; i++ {
		if nums[i] > nums[i-1] {
			up = down + 1
		} else if nums[i] < nums[i-1] {
			down = up + 1
		}
	}
	return max(up, down)
}

// 377. 组合总和 Ⅳ
// 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
//
// 题目数据保证答案符合 32 位整数范围。
//
// 示例 1：
// 输入：nums = [1,2,3], target = 4 输出：7
// 解释：
// 所有可能的组合为：
// (1, 1, 1, 1)
// (1, 1, 2)
// (1, 2, 1)
// (1, 3)
// (2, 1, 1)
// (2, 2)
// (3, 1)
// 请注意，顺序不同的序列被视作不同的组合。
//
// 示例 2：
// 输入：nums = [9], target = 3 输出：0
//
// 提示：
// 1 <= nums.length <= 200
// 1 <= nums[i] <= 1000
// nums 中的所有元素 互不相同
// 1 <= target <= 1000
//
// 进阶：如果给定的数组中含有负数会发生什么？问题会产生何种变化？如果允许负数出现，需要向题目中添加哪些限制条件？
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	for _, num := range nums {
		if num <= target {
			dp[num] = 1
		}
	}
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			if i-num >= 0 {
				dp[i] += dp[i-num]
			}
		}
	}

	return dp[target]
}

// 403. 青蛙过河
// 一只青蛙想要过河。 假定河流被等分为若干个单元格，并且在每一个单元格内都有可能放有一块石子（也有可能没有）。 青蛙可以跳上石子，但是不可以跳入水中。
//
// 给你石子的位置列表 stones（用单元格序号 升序 表示）， 请判定青蛙能否成功过河（即能否在最后一步跳至最后一块石子上）。
// 开始时， 青蛙默认已站在第一块石子上，并可以假定它第一步只能跳跃一个单位（即只能从单元格 1 跳至单元格 2 ）。
// 如果青蛙上一步跳跃了 k 个单位，那么它接下来的跳跃距离只能选择为 k - 1、k 或 k + 1 个单位。 另请注意，青蛙只能向前方（终点的方向）跳跃。
//
// 示例 1：
// 输入：stones = [0,1,3,5,6,8,12,17]
// 输出：true
// 解释：青蛙可以成功过河，按照如下方案跳跃：跳 1 个单位到第 2 块石子, 然后跳 2 个单位到第 3 块石子, 接着 跳 2 个单位到第 4 块石子, 然后跳 3 个单位到第 6 块石子, 跳 4 个单位到第 7 块石子, 最后，跳 5 个单位到第 8 个石子（即最后一块石子）。
//
// 示例 2：
// 输入：stones = [0,1,2,3,4,8,9,11]
// 输出：false
// 解释：这是因为第 5 和第 6 个石子之间的间距太大，没有可选的方案供青蛙跳跃过去。
//
// 提示：
// 2 <= stones.length <= 2000
// 0 <= stones[i] <= 231 - 1
// stones[0] == 0
func canCross(stones []int) bool {
	n := len(stones)
	dp := make([][]bool, n)
	// dp[i][k] 表示 能否到达第i块石子, 上一次的跳跃距离为k
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	dp[0][0] = true
	for i := 1; i < n; i++ {
		if stones[i]-stones[i-1] > i {
			return false
		}
	}
	for i := 1; i < n; i++ {
		for j := i - 1; j >= 0; j-- {
			k := stones[i] - stones[j]
			if k > j+1 {
				break
			}
			dp[i][k] = dp[j][k-1] || dp[j][k] || dp[j][k+1]
			if i == n-1 && dp[i][k] {
				return true
			}
		}
	}

	return false
}

// 413. 等差数列划分
// 如果一个数列 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该数列为等差数列。
//
// 例如，[1,3,5,7,9]、[7,7,7,7] 和 [3,-1,-5,-9] 都是等差数列。
// 给你一个整数数组 nums ，返回数组 nums 中所有为等差数组的 子数组 个数。
//
// 子数组 是数组中的一个连续序列。
//
// 示例 1：
// 输入：nums = [1,2,3,4] 输出：3
// 解释：nums 中有三个子等差数组：[1, 2, 3]、[2, 3, 4] 和 [1,2,3,4] 自身。
//
// 示例 2：
// 输入：nums = [1]
// 输出：0
//
// 提示：
// 1 <= nums.length <= 5000
// -1000 <= nums[i] <= 1000
func numberOfArithmeticSlices(nums []int) int {
	n := len(nums)
	counts := make([]int, n)
	for i := 2; i < n; i++ {
		first, second, third := nums[i-2], nums[i-1], nums[i]
		if first+third == second<<1 {
			counts[i] = counts[i-1] + 1
		}
	}

	result := 0
	for _, count := range counts {
		result += count
	}

	return result
}

// 416. 分割等和子集
// 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
//
// 示例 1：
// 输入：nums = [1,5,11,5]
// 输出：true
// 解释：数组可以分割成 [1, 5, 5] 和 [11] 。
//
// 示例 2：
// 输入：nums = [1,2,3,5]
// 输出：false
// 解释：数组不能分割成两个元素和相等的子集。
//
// 提示：
// 1 <= nums.length <= 200
// 1 <= nums[i] <= 100
func canPartition(nums []int) bool {
	n := len(nums)
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum&1 == 1 {
		return false
	}
	target := sum >> 1
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, target+1)
	}
	sort.Ints(nums)
	dp[0][0] = true
	if nums[0] < target {
		dp[0][nums[0]] = true
	} else if nums[0] == target {
		return true
	}
	for i := 1; i < n; i++ {
		for j := 0; j <= target; j++ {
			dp[i][j] = dp[i-1][j]
			if !dp[i][j] && nums[i] <= j {
				dp[i][j] = dp[i-1][j-nums[i]]
			}
		}
		if dp[i][target] {
			return true
		}
	}

	return false
}

// 446. 等差数列划分 II - 子序列
// 给你一个整数数组 nums ，返回 nums 中所有 等差子序列 的数目。
//
// 如果一个序列中 至少有三个元素 ，并且任意两个相邻元素之差相同，则称该序列为等差序列。
//
// 例如，[1, 3, 5, 7, 9]、[7, 7, 7, 7] 和 [3, -1, -5, -9] 都是等差序列。
// 再例如，[1, 1, 2, 5, 7] 不是等差序列。
// 数组中的子序列是从数组中删除一些元素（也可能不删除）得到的一个序列。
//
// 例如，[2,5,10] 是 [1,2,1,2,4,1,5,10] 的一个子序列。
// 题目数据保证答案是一个 32-bit 整数。
//
// 示例 1：
// 输入：nums = [2,4,6,8,10]
// 输出：7
// 解释：所有的等差子序列为：
// [2,4,6]
// [4,6,8]
// [6,8,10]
// [2,4,6,8]
// [4,6,8,10]
// [2,4,6,8,10]
// [2,6,10]
//
// 示例 2：
// 输入：nums = [7,7,7,7,7]
// 输出：16
// 解释：数组中的任意子序列都是等差子序列。
//
// 提示：
// 1  <= nums.length <= 1000
// -231 <= nums[i] <= 231 - 1
func numberOfArithmeticSlicesII(nums []int) int {
	n := len(nums)
	// dp[i][j]为以nums[i]和nums[j]为最后两个元素的等差数列的个数
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
	}
	count := 0
	indexMap := make(map[int][]int)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			// preNum + nums[j] = 2 * nums[i
			preNum := nums[i] - nums[j] + nums[i]
			if indexs, ok := indexMap[preNum]; ok {
				for _, index := range indexs {
					dp[i][j] += dp[index][i] + 1
				}
			}
			count += dp[i][j]
		}
		indexMap[nums[i]] = append(indexMap[nums[i]], i)
	}

	return count
}

// 464. 我能赢吗
// 在 "100 game" 这个游戏中，两名玩家轮流选择从 1 到 10 的任意整数，累计整数和，先使得累计整数和达到或超过 100 的玩家，即为胜者。
//
// 如果我们将游戏规则改为 “玩家不能重复使用整数” 呢？
//
// 例如，两个玩家可以轮流从公共整数池中抽取从 1 到 15 的整数（不放回），直到累计整数和 >= 100。
//
// 给定一个整数 maxChoosableInteger （整数池中可选择的最大数）和另一个整数 desiredTotal（累计和），判断先出手的玩家是否能稳赢（假设两位玩家游戏时都表现最佳）？
//
// 你可以假设 maxChoosableInteger 不会大于 20， desiredTotal 不会大于 300。
//
// 示例：
// 输入：
// maxChoosableInteger = 10
// desiredTotal = 11
// 输出：
// false
//
// 解释：
// 无论第一个玩家选择哪个整数，他都会失败。
// 第一个玩家可以选择从 1 到 10 的整数。
// 如果第一个玩家选择 1，那么第二个玩家只能选择从 2 到 10 的整数。
// 第二个玩家可以通过选择整数 10（那么累积和为 11 >= desiredTotal），从而取得胜利.
// 同样地，第一个玩家选择任意其他整数，第二个玩家都会赢。
func canIWin(maxChoosableInteger int, desiredTotal int) bool {
	if maxChoosableInteger >= desiredTotal {
		return true
	}
	// sum(1 ~ maxChoosableInteger) 总和 <  desiredTotal
	if maxChoosableInteger*(1+maxChoosableInteger)/2 < desiredTotal {
		return false
	}

	// dp[state] 表示 state 状态下
	dp := make([]int, 1<<maxChoosableInteger)

	var dfs func(state, total int) int

	dfs = func(state, total int) int {
		if dp[state] != 0 {
			return dp[state]
		}

		for i := 1; i <= maxChoosableInteger; i++ {
			cur := 1 << (i - 1)
			if state&cur != 0 {
				continue
			}
			// 我赢 || 下一回合输
			if total <= i || dfs(state|cur, total-i) == -1 {
				dp[state] = 1
				return 1
			}
		}

		dp[state] = -1
		return dp[state]
	}

	return dfs(0, desiredTotal) == 1
}

// 467. 环绕字符串中唯一的子字符串
// 把字符串 s 看作是“abcdefghijklmnopqrstuvwxyz”的无限环绕字符串，所以 s 看起来是这样的："...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....".
//
// 现在我们有了另一个字符串 p 。你需要的是找出 s 中有多少个唯一的 p 的非空子串，尤其是当你的输入是字符串 p ，你需要输出字符串 s 中 p 的不同的非空子串的数目。
// 注意: p 仅由小写的英文字母组成，p 的大小可能超过 10000。
//
// 示例 1:
// 输入: "a"
// 输出: 1
// 解释: 字符串 S 中只有一个"a"子字符。
//
// 示例 2:
// 输入: "cac"
// 输出: 2
// 解释: 字符串 S 中的字符串“cac”只有两个子串“a”、“c”。.
//
// 示例 3:
// 输入: "zab"
// 输出: 6
// 解释: 在字符串 S 中有六个子串“z”、“a”、“b”、“za”、“ab”、“zab”。.
func findSubstringInWraproundString(p string) int {

	n := len(p)
	if n == 0 {
		return 0
	}
	dp := [26]int{}
	count := 0
	for i := 0; i < n; i++ {
		c := p[i]
		// p[i - 1] 是 c 的 上一位  p[i]-p[i-1] == 1 || p[i]-p[i-1] == -25
		if i > 0 && (c-p[i-1]+25)%26 == 0 {
			count++
		} else {
			count = 1
		}
		dp[c-'a'] = max(dp[c-'a'], count)
	}

	result := 0
	for _, num := range dp {
		result += num
	}
	return result
}

// 1218. 最长定差子序列
// 给你一个整数数组 arr 和一个整数 difference，请你找出并返回 arr 中最长等差子序列的长度，该子序列中相邻元素之间的差等于 difference 。
//
// 子序列 是指在不改变其余元素顺序的情况下，通过删除一些元素或不删除任何元素而从 arr 派生出来的序列。
//
// 示例 1：
// 输入：arr = [1,2,3,4], difference = 1
// 输出：4
// 解释：最长的等差子序列是 [1,2,3,4]。
//
// 示例 2：
// 输入：arr = [1,3,5,7], difference = 1
// 输出：1
// 解释：最长的等差子序列是任意单个元素。
//
// 示例 3：
// 输入：arr = [1,5,7,8,5,3,4,2,1], difference = -2
// 输出：4
// 解释：最长的等差子序列是 [7,5,3,1]。
//
// 提示：
// 1 <= arr.length <= 105
// -104 <= arr[i], difference <= 104
func longestSubsequence(arr []int, difference int) int {
	result := 0
	subMap := make(map[int]int)
	for _, num := range arr {
		count := subMap[num-difference] + 1
		subMap[num] = count
		result = max(result, count)
	}

	return result
}

// 466. 统计重复个数
// 定义 str = [s, n] 表示 str 由 n 个字符串 s 连接构成。
//
// 例如，str == ["abc", 3] =="abcabcabc" 。
// 如果可以从 s2 中删除某些字符使其变为 s1，则称字符串 s1 可以从字符串 s2 获得。
//
// 例如，根据定义，s1 = "abc" 可以从 s2 = "abdbec" 获得，仅需要删除加粗且用斜体标识的字符。
// 现在给你两个字符串 s1 和 s2 和两个整数 n1 和 n2 。由此构造得到两个字符串，其中 str1 = [s1, n1]、str2 = [s2, n2] 。
//
// 请你找出一个最大整数 m ，以满足 str = [str2, m] 可以从 str1 获得。
//
// 示例 1：
// 输入：s1 = "acb", n1 = 4, s2 = "ab", n2 = 2
// 输出：2
//
// 示例 2：
// 输入：s1 = "acb", n1 = 1, s2 = "acb", n2 = 1
// 输出：1
//
// 提示：
// 1 <= s1.length, s2.length <= 100
// s1 和 s2 由小写英文字母组成
// 1 <= n1, n2 <= 106
func getMaxRepetitions(s1 string, n1 int, s2 string, n2 int) int {
	m, n := len(s1), len(s2)
	if m*n*n1*n2 == 0 {
		return 0
	}
	if m*n1 < n*n2 {
		return 0
	}
	// dp[i]表示从s2的i字符开始匹配s1，可以匹配多少个字符
	dp := make([]int, n)
	for i := 0; i < n; i++ {
		k := i
		for j := 0; j < m; j++ {
			if s1[j] == s2[k] {
				k++
				k %= n
				dp[i]++
			}
		}
	}
	count, index := 0, 0

	for i := 0; i < n1; i++ {
		num := dp[index]
		index += num
		index %= n
		count += num
	}

	return count / n / n2
}

// 474. 一和零
// 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
// 请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
// 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
//
// 示例 1：
// 输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
// 输出：4
// 解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
// 其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
//
// 示例 2：
// 输入：strs = ["10", "0", "1"], m = 1, n = 1
// 输出：2
// 解释：最大的子集是 {"0", "1"} ，所以答案是 2 。
//
// 提示：
// 1 <= strs.length <= 600
// 1 <= strs[i].length <= 100
// strs[i] 仅由 '0' 和 '1' 组成
// 1 <= m, n <= 100
func findMaxForm(strs []string, m int, n int) int {
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for _, s := range strs {
		zeroCount := strings.Count(s, "0")
		oneCount := len(s) - zeroCount
		for i := m; i >= zeroCount; i-- {
			for j := n; j >= oneCount; j-- {
				dp[i][j] =
					max(dp[i][j], dp[i-zeroCount][j-oneCount]+1)
			}
		}
	}

	return dp[m][n]
}

// PredictTheWinner
// 486. 预测赢家
// 给你一个整数数组 nums 。玩家 1 和玩家 2 基于这个数组设计了一个游戏。
//
// 玩家 1 和玩家 2 轮流进行自己的回合，玩家 1 先手。开始时，两个玩家的初始分值都是 0 。每一回合，玩家从数组的任意一端取一个数字（即，nums[0] 或 nums[nums.length - 1]），取到的数字将会从数组中移除（数组长度减 1 ）。玩家选中的数字将会加到他的得分上。当数组中没有剩余数字可取时，游戏结束。
//
// 如果玩家 1 能成为赢家，返回 true 。如果两个玩家得分相等，同样认为玩家 1 是游戏的赢家，也返回 true 。你可以假设每个玩家的玩法都会使他的分数最大化。
//
// 示例 1：
// 输入：nums = [1,5,2]
// 输出：false
// 解释：一开始，玩家 1 可以从 1 和 2 中进行选择。
// 如果他选择 2（或者 1 ），那么玩家 2 可以从 1（或者 2 ）和 5 中进行选择。如果玩家 2 选择了 5 ，那么玩家 1 则只剩下 1（或者 2 ）可选。
// 所以，玩家 1 的最终分数为 1 + 2 = 3，而玩家 2 为 5 。
// 因此，玩家 1 永远不会成为赢家，返回 false 。
//
// 示例 2：
// 输入：nums = [1,5,233,7]
// 输出：true
// 解释：玩家 1 一开始选择 1 。然后玩家 2 必须从 5 和 7 中进行选择。无论玩家 2 选择了哪个，玩家 1 都可以选择 233 。
// 最终，玩家 1（234 分）比玩家 2（12 分）获得更多的分数，所以返回 true，表示玩家 1 可以成为赢家。
//
// 提示：
// 1 <= nums.length <= 20
// 0 <= nums[i] <= 107
func PredictTheWinner(nums []int) bool {
	n := len(nums)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		dp[i][i] = nums[i]
	}
	for i := n - 2; i >= 0; i-- {
		for j := i + 1; j < n; j++ {
			// 选择[i]
			a := nums[i] - dp[i+1][j]
			// 选择[j]
			b := nums[j] - dp[i][j-1]
			// dp[i][j]
			dp[i][j] = max(a, b)
		}
	}

	return dp[0][n-1] >= 0
}

// 494. 目标和
// 给你一个整数数组 nums 和一个整数 target 。
//
// 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
//
// 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
// 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
//
// 示例 1：
// 输入：nums = [1,1,1,1,1], target = 3
// 输出：5
// 解释：一共有 5 种方法让最终目标和为 3 。
// -1 + 1 + 1 + 1 + 1 = 3
// +1 - 1 + 1 + 1 + 1 = 3
// +1 + 1 - 1 + 1 + 1 = 3
// +1 + 1 + 1 - 1 + 1 = 3
// +1 + 1 + 1 + 1 - 1 = 3
//
// 示例 2：
// 输入：nums = [1], target = 1
// 输出：1
//
// 提示：
// 1 <= nums.length <= 20
// 0 <= nums[i] <= 1000
// 0 <= sum(nums[i]) <= 1000
// -1000 <= target <= 1000
func findTargetSumWays(nums []int, target int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum < target {
		return 0
	}
	// a + b = sum  a -b = target
	// 2 * b = sum - target
	if sum-target < 0 || (sum-target)&1 == 1 {
		// 奇数 不能被2整除
		return 0
	}
	b := (sum - target) >> 1
	// 在 nums 中寻找 a 的个数
	dp := make([]int, b+1)
	dp[0] = 1
	for _, num := range nums {
		for i := b; i >= num; i-- {
			dp[i] += dp[i-num]
		}
	}

	return dp[b]
}

// 514. 自由之路
// 电子游戏“辐射4”中，任务“通向自由”要求玩家到达名为“Freedom Trail Ring”的金属表盘，并使用表盘拼写特定关键词才能开门。
// 给定一个字符串 ring，表示刻在外环上的编码；给定另一个字符串 key，表示需要拼写的关键词。您需要算出能够拼写关键词中所有字符的最少步数。
// 最初，ring 的第一个字符与12:00方向对齐。您需要顺时针或逆时针旋转 ring 以使 key 的一个字符在 12:00 方向对齐，然后按下中心按钮，以此逐个拼写完 key 中的所有字符。
// 旋转 ring 拼出 key 字符 key[i] 的阶段中：
//
// 您可以将 ring 顺时针或逆时针旋转一个位置，计为1步。旋转的最终目的是将字符串 ring 的一个字符与 12:00 方向对齐，并且这个字符必须等于字符 key[i] 。
// 如果字符 key[i] 已经对齐到12:00方向，您需要按下中心按钮进行拼写，这也将算作 1 步。按完之后，您可以开始拼写 key 的下一个字符（下一阶段）, 直至完成所有拼写。
//
// 示例：
// 输入: ring = "godding", key = "gd"
// 输出: 4
// 解释:
//  对于 key 的第一个字符 'g'，已经在正确的位置, 我们只需要1步来拼写这个字符。
//  对于 key 的第二个字符 'd'，我们需要逆时针旋转 ring "godding" 2步使它变成 "ddinggo"。
//  当然, 我们还需要1步进行拼写。
//  因此最终的输出是 4。
//
// 提示：
// ring 和 key 的字符串长度取值范围均为 1 至 100；
// 两个字符串中都只有小写字符，并且均可能存在重复字符；
// 字符串 key 一定可以由字符串 ring 旋转拼出。
func findRotateSteps(ring string, key string) int {
	const inf = math.MaxInt32 >> 1
	m, n := len(ring), len(key)
	indexMap := make(map[byte][]int)
	for i := 0; i < m; i++ {
		c := ring[i]
		indexMap[c] = append(indexMap[c], i)
	}
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		for j := 0; j < n; j++ {
			dp[i][j] = inf
		}
	}
	// dp[i][j] ring[i] 到 key[j] 的距离
	first := key[0]
	for _, idx := range indexMap[first] {
		dp[idx][0] = min(idx, m-idx) + 1
	}

	for j := 1; j < n; j++ {
		c := key[j]
		for _, idx := range indexMap[c] {
			for _, last := range indexMap[key[j-1]] {
				distance := abs(idx - last)
				// idx 到 last 的 最小距离
				l := min(distance, m-distance)
				dp[idx][j] = min(dp[idx][j], dp[last][j-1]+l+1)
			}
		}
	}

	result := inf
	for i := 0; i < m; i++ {
		result = min(result, dp[i][n-1])
	}
	return result
}

// 516. 最长回文子序列
// 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
// 子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
//
// 示例 1：
// 输入：s = "bbbab" 输出：4
// 解释：一个可能的最长回文子序列为 "bbbb" 。
//
// 示例 2：
// 输入：s = "cbbd" 输出：2
// 解释：一个可能的最长回文子序列为 "bb" 。
//
// 提示：
// 1 <= s.length <= 1000
// s 仅由小写英文字母组成
func longestPalindromeSubseq(s string) int {
	n := len(s)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		dp[i][i] = 1
	}
	for i := n - 1; i >= 0; i-- {
		for j := i + 1; j < n; j++ {
			if s[i] == s[j] {
				dp[i][j] = max(dp[i][j], dp[i+1][j-1]+2)
			} else {
				dp[i][j] = max(dp[i+1][j], dp[i][j-1])
			}
		}
	}

	return dp[0][n-1]
}

// 518. 零钱兑换 II
// 给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
// 请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
// 假设每一种面额的硬币有无限个。
// 题目数据保证结果符合 32 位带符号整数。
//
// 示例 1：
// 输入：amount = 5, coins = [1, 2, 5]
// 输出：4
// 解释：有四种方式可以凑成总金额：
// 5=5
// 5=2+2+1
// 5=2+1+1+1
// 5=1+1+1+1+1
//
// 示例 2：
// 输入：amount = 3, coins = [2]
// 输出：0
// 解释：只用面额 2 的硬币不能凑成总金额 3 。
//
// 示例 3：
// 输入：amount = 10, coins = [10]
// 输出：1
//
// 提示：
// 1 <= coins.length <= 300
// 1 <= coins[i] <= 5000
// coins 中的所有值 互不相同
// 0 <= amount <= 5000
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for i := coin; i <= amount; i++ {
			dp[i] += dp[i-coin]
		}
	}
	return dp[amount]
}

// 546. 移除盒子
// 给出一些不同颜色的盒子，盒子的颜色由数字表示，即不同的数字表示不同的颜色。
//
// 你将经过若干轮操作去去掉盒子，直到所有的盒子都去掉为止。每一轮你可以移除具有相同颜色的连续 k 个盒子（k >= 1），这样一轮之后你将得到 k * k 个积分。
//
// 当你将所有盒子都去掉之后，求你能获得的最大积分和。
//
// 示例 1：
// 输入：boxes = [1,3,2,2,2,3,4,3,1]
// 输出：23
// 解释：
// [1, 3, 2, 2, 2, 3, 4, 3, 1]
// ----> [1, 3, 3, 4, 3, 1] (3*3=9 分)
// ----> [1, 3, 3, 3, 1] (1*1=1 分)
// ----> [1, 1] (3*3=9 分)
// ----> [] (2*2=4 分)
//
// 示例 2：
// 输入：boxes = [1,1,1] 输出：9
//
// 示例 3：
// 输入：boxes = [1] 输出：1
//
// 提示：
// 1 <= boxes.length <= 100
// 1 <= boxes[i] <= 100
func removeBoxes(boxes []int) int {
	dp := [100][100][100]int{}
	n := len(boxes)
	// dp[i][j][k]代表的是区间段[i, j]（在i前有k个元素与boxes[i]相同的数字）消除可得的最大值
	var calPoint func(l, r, k int) int

	calPoint = func(l, r, k int) int {
		if l > r {
			return 0
		}
		// 左右两部分 左 l ~ r - 1  右 r ~ r + k (k+1个 boxes[r])
		for r > l && boxes[r] == boxes[r-1] {
			r--
			k++
		}
		if dp[l][r][k] == 0 {
			// r ~ r + k 共有 k 个 boxes[r]
			dp[l][r][k] = calPoint(l, r-1, 0) + (k+1)*(k+1)
			for i := l; i < r; i++ {
				if boxes[i] == boxes[r] {
					dp[l][r][k] = max(dp[l][r][k], calPoint(l, i, k+1)+calPoint(i+1, r-1, 0))
				}
			}
		}
		return dp[l][r][k]
	}

	return calPoint(0, n-1, 0)
}

var (
	Mod    = 1_000_000_007
	DirCol = []int{1, -1, 0, 0}
	DirRow = []int{0, 0, 1, -1}
)

// 552. 学生出勤记录 II
// 可以用字符串表示一个学生的出勤记录，其中的每个字符用来标记当天的出勤情况（缺勤、迟到、到场）。记录中只含下面三种字符：
// 'A'：Absent，缺勤
// 'L'：Late，迟到
// 'P'：Present，到场
// 如果学生能够 同时 满足下面两个条件，则可以获得出勤奖励：
//
// 按 总出勤 计，学生缺勤（'A'）严格 少于两天。
// 学生 不会 存在 连续 3 天或 连续 3 天以上的迟到（'L'）记录。
// 给你一个整数 n ，表示出勤记录的长度（次数）。请你返回记录长度为 n 时，可能获得出勤奖励的记录情况 数量 。
// 答案可能很大，所以返回对 109 + 7 取余 的结果。
//
// 示例 1：
// 输入：n = 2 输出：8
// 解释：
// 有 8 种长度为 2 的记录将被视为可奖励：
// "PP" , "AP", "PA", "LP", "PL", "AL", "LA", "LL"
// 只有"AA"不会被视为可奖励，因为缺勤次数为 2 次（需要少于 2 次）。
//
// 示例 2：
// 输入：n = 1 输出：3
//
// 示例 3：
// 输入：n = 10101 输出：183236316
//
// 提示：
// 1 <= n <= 105
func checkRecord(n int) int {
	if n == 1 {
		return 3
	}
	// "PP" , "AP", "PA", "LP", "PL", "AL", "LA", "LL"
	// LL结尾包含A的数量
	// LL结尾不包含A的数量 "LL"
	// L结尾包含A的数量 "AL"
	// L结尾不包含A的数量 "PL"
	// 其他包含A的数量 "AP", "PA", "LA"
	// 其他不包含A的数量 "PP" , "LP"

	dp0, dp1, dp2, dp3, dp4, dp5 := 0, 1, 1, 1, 3, 2

	for i := 3; i <= n; i++ {
		tmp4, tmp5 := dp4, dp5
		// A 结尾
		dp4 += dp0 + dp1 + dp2 + dp3 + dp5
		dp4 %= Mod
		// P 结尾
		dp5 += dp1 + dp3
		dp5 %= Mod
		// L 结尾
		dp0 = dp2
		dp1 = dp3
		dp2 = tmp4
		dp3 = tmp5
		dp0 %= Mod
		dp1 %= Mod
		dp2 %= Mod
		dp3 %= Mod
	}

	dp0 += dp1 + dp2 + dp3 + dp4 + dp5
	dp0 %= Mod

	return dp0
}

// 576. 出界的路径数
// 给你一个大小为 m x n 的网格和一个球。球的起始坐标为 [startRow, startColumn] 。你可以将球移到在四个方向上相邻的单元格内（可以穿过网格边界到达网格之外）。你 最多 可以移动 maxMove 次球。
//
// 给你五个整数 m、n、maxMove、startRow 以及 startColumn ，找出并返回可以将球移出边界的路径数量。因为答案可能非常大，返回对 109 + 7 取余 后的结果。
//
// 示例 1：
// 输入：m = 2, n = 2, maxMove = 2, startRow = 0, startColumn = 0
// 输出：6
//
// 示例 2：
// 输入：m = 1, n = 3, maxMove = 3, startRow = 0, startColumn = 1
// 输出：12
//
// 提示：
// 1 <= m, n <= 50
// 0 <= maxMove <= 50
// 0 <= startRow < m
// 0 <= startColumn < n
func findPaths(m int, n int, maxMove int, startRow int, startColumn int) int {
	dp := make([][][]int, m+2)
	for i := 0; i <= m+1; i++ {
		dp[i] = make([][]int, n+2)
		for j := 0; j <= n+1; j++ {
			dp[i][j] = make([]int, maxMove+1)
		}
		// 第一列和最后一列
		dp[i][0][0] = 1
		dp[i][n+1][0] = 1
	}
	for j := 0; j <= n+1; j++ {
		// 第一行和最后一行
		dp[0][j][0] = 1
		dp[m+1][j][0] = 1
	}

	for k := 1; k <= maxMove; k++ {
		for i := 1; i <= m; i++ {
			for j := 1; j <= n; j++ {
				for s := 0; s < 4; s++ {
					prevRow, prevCol := i+DirRow[s], j+DirCol[s]
					dp[i][j][k] += dp[prevRow][prevCol][k-1]
					dp[i][j][k] %= Mod
				}
			}
		}
	}

	result := 0
	for k := 1; k <= maxMove; k++ {
		result += dp[startRow+1][startColumn+1][k]
		result %= Mod
	}
	return result
}

// 583. 两个字符串的删除操作
// 给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
//
// 示例：
// 输入: "sea", "eat"
// 输出: 2
// 解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
//
// 提示：
// 给定单词的长度不超过500。
// 给定单词中的字符只含有小写字母。
func minDistanceII(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	if word1 == word2 {
		return 0
	}
	if m*n == 0 {
		return max(m, n)
	}
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if word1[i] == word2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
		}
	}

	return m + n - 2*dp[m][n]
}

// 600. 不含连续1的非负整数
// 给定一个正整数 n，找出小于或等于 n 的非负整数中，其二进制表示不包含 连续的1 的个数。
//
// 示例 1:
// 输入: 5 输出: 5
// 解释:
// 下面是带有相应二进制表示的非负整数<= 5：
// 0 : 0
// 1 : 1
// 2 : 10
// 3 : 11
// 4 : 100
// 5 : 101
// 其中，只有整数3违反规则（有两个连续的1），其他5个满足规则。
// 说明: 1 <= n <= 109
func findIntegers(n int) int {
	// dp[i] 表示 小于1 << i  符合要求的数
	dp := [31]int{}
	dp[0] = 1 // 0
	dp[1] = 2 // 0 1   ->  0 1 2
	for i := 2; i < 31; i++ {
		// 10开头 dp[i-2]  0开头 dp[i-1]
		dp[i] = dp[i-2] + dp[i-1]
	}
	sum := 0
	m, idx := 1<<30, 30
	preBit := 0
	for m > 0 {
		if m&n != 0 {
			sum += dp[idx]
			if preBit == 1 {
				sum--
				break
			}

			preBit = 1
		} else {
			preBit = 0
		}
		idx--
		m >>= 1
	}

	return sum + 1
}

// 646. 最长数对链
// 给出 n 个数对。 在每一个数对中，第一个数字总是比第二个数字小。
//
// 现在，我们定义一种跟随关系，当且仅当 b < c 时，数对(c, d) 才可以跟在 (a, b) 后面。我们用这种形式来构造一个数对链。
//
// 给定一个数对集合，找出能够形成的最长数对链的长度。你不需要用到所有的数对，你可以以任何顺序选择其中的一些数对来构造。
//
// 示例：
// 输入：[[1,2], [2,3], [3,4]]
// 输出：2
// 解释：最长的数对链是 [1,2] -> [3,4]
//
// 提示：
// 给出数对的个数在 [1, 1000] 范围内。
func findLongestChain(pairs [][]int) int {
	n := len(pairs)
	dp := make([]int, n)
	dp[0] = 1
	//
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i][0] == pairs[j][0] {
			return pairs[i][1] < pairs[j][1]
		}
		return pairs[i][0] < pairs[j][0]
	})
	result := 1
	for i := 1; i < n; i++ {
		num := pairs[i][0]
		idx := i - 1
		for idx >= 0 && pairs[idx][1] >= num {
			idx--
		}
		if idx >= 0 {
			dp[i] = dp[idx] + 1
		} else {
			dp[i] = 1
		}
		result = max(result, dp[i])
	}

	return result
}

// 639. 解码方法 II
// 一条包含字母 A-Z 的消息通过以下的方式进行了编码：
//
// 'A' -> 1
// 'B' -> 2
// ...
// 'Z' -> 26
// 要 解码 一条已编码的消息，所有的数字都必须分组，然后按原来的编码方案反向映射回字母（可能存在多种方式）。例如，"11106" 可以映射为：
//
// "AAJF" 对应分组 (1 1 10 6)
// "KJF" 对应分组 (11 10 6)
// 注意，像 (1 11 06) 这样的分组是无效的，因为 "06" 不可以映射为 'F' ，因为 "6" 与 "06" 不同。
//
// 除了 上面描述的数字字母映射方案，编码消息中可能包含 '*' 字符，可以表示从 '1' 到 '9' 的任一数字（不包括 '0'）。例如，编码字符串 "1*" 可以表示 "11"、"12"、"13"、"14"、"15"、"16"、"17"、"18" 或 "19" 中的任意一条消息。对 "1*" 进行解码，相当于解码该字符串可以表示的任何编码消息。
//
// 给你一个字符串 s ，由数字和 '*' 字符组成，返回 解码 该字符串的方法 数目 。
//
// 由于答案数目可能非常大，返回对 109 + 7 取余 的结果。
//
// 示例 1：
// 输入：s = "*"
// 输出：9
// 解释：这一条编码消息可以表示 "1"、"2"、"3"、"4"、"5"、"6"、"7"、"8" 或 "9" 中的任意一条。
// 可以分别解码成字符串 "A"、"B"、"C"、"D"、"E"、"F"、"G"、"H" 和 "I" 。
// 因此，"*" 总共有 9 种解码方法。
//
// 示例 2：
// 输入：s = "1*"
// 输出：18
// 解释：这一条编码消息可以表示 "11"、"12"、"13"、"14"、"15"、"16"、"17"、"18" 或 "19" 中的任意一条。
// 每种消息都可以由 2 种方法解码（例如，"11" 可以解码成 "AA" 或 "K"）。
// 因此，"1*" 共有 9 * 2 = 18 种解码方法。
//
// 示例 3：
// 输入：s = "2*"
// 输出：15
// 解释：这一条编码消息可以表示 "21"、"22"、"23"、"24"、"25"、"26"、"27"、"28" 或 "29" 中的任意一条。
// "21"、"22"、"23"、"24"、"25" 和 "26" 由 2 种解码方法，但 "27"、"28" 和 "29" 仅有 1 种解码方法。
// 因此，"2*" 共有 (6 * 2) + (3 * 1) = 12 + 3 = 15 种解码方法。
//
// 提示：
// 1 <= s.length <= 105
// s[i] 是 0 - 9 中的一位数字或字符 '*'
func numDecodingsII(s string) int {
	n := len(s)
	dp := make([]int, n+1)
	dp[0] = 1

	for i := 0; i < n; i++ {
		// * 是 1~9
		if s[i] == '*' {
			dp[i+1] += dp[i] * 9
		} else if s[i] == '0' {

			if i > 0 {
				if s[i-1] == '*' {
					dp[i+1] += dp[i-1] * 2
				} else if s[i-1] == '1' || s[i-1] == '2' {
					dp[i+1] += dp[i-1]
				}
			}
			continue
		} else {
			// dp[i]种情况
			dp[i+1] += dp[i]
		}
		if i > 0 {
			if s[i-1] == '1' {
				if s[i] == '*' {
					// 11 ~ 19
					dp[i+1] += dp[i-1] * 9
				} else {
					dp[i+1] += dp[i-1]
				}
			} else if s[i-1] == '2' {
				if s[i] == '*' {
					// 21 ~ 26
					dp[i+1] += dp[i-1] * 6
				} else if s[i] <= '6' {
					dp[i+1] += dp[i-1]
				}
			} else if s[i-1] == '*' {
				if s[i] == '*' {
					// 11 ~ 26
					dp[i+1] += dp[i-1] * 15
				} else if s[i] <= '6' {
					dp[i+1] += dp[i-1] * 2
				} else {
					dp[i+1] += dp[i-1]
				}
			}
		}
		dp[i+1] %= Mod
	}

	return dp[n]
}

// 664. 奇怪的打印机
// 有台奇怪的打印机有以下两个特殊要求：
//
// 打印机每次只能打印由 同一个字符 组成的序列。
// 每次可以在从起始到结束的任意位置打印新字符，并且会覆盖掉原来已有的字符。
// 给你一个字符串 s ，你的任务是计算这个打印机打印它需要的最少打印次数。
//
// 示例 1：
// 输入：s = "aaabbb"
// 输出：2
// 解释：首先打印 "aaa" 然后打印 "bbb"。
//
// 示例 2：
// 输入：s = "aba"
// 输出：2
// 解释：首先打印 "aaa" 然后在第二个位置打印 "b" 覆盖掉原来的字符 'a'。
//
// 提示：
// 1 <= s.length <= 100
// s 由小写英文字母组成
func strangePrinter(s string) int {
	n := len(s)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		dp[i][i] = 1
	}
	for l := 1; l < n; l++ {
		for i := 0; i < n-l; i++ {
			j := i + l
			dp[i][j] = l + 1
			for k := i; k < j; k++ {
				total := dp[i][k] + dp[k+1][j]
				if s[j] == s[k] {
					total--
				}
				dp[i][j] = min(dp[i][j], total)
			}
		}
	}
	return dp[0][n-1]
}

// 673. 最长递增子序列的个数
// 给定一个未排序的整数数组 nums ， 返回最长递增子序列的个数 。
//
// 注意 这个数列必须是 严格 递增的。
//
// 示例 1:
// 输入: [1,3,5,4,7]
// 输出: 2
// 解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
//
// 示例 2:
// 输入: [2,2,2,2,2]
// 输出: 5
// 解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。
//
// 提示:
// 1 <= nums.length <= 2000
// -106 <= nums[i] <= 106
func findNumberOfLIS(nums []int) int {
	count, maxLen, n := 0, 0, len(nums)
	lens, counts := make([]int, n), make([]int, n)
	for i := 0; i < n; i++ {
		counts[i] = 1
	}
	for i := 0; i < n; i++ {
		num := nums[i]
		for j := 0; j < i; j++ {
			if num <= nums[j] {
				continue
			}
			if lens[j] >= lens[i] {
				lens[i] = lens[j] + 1
				counts[i] = counts[j]
			} else if lens[j]+1 == lens[i] {
				counts[i] += counts[j]
			}
		}
		maxLen = max(maxLen, lens[i])
	}
	for i := 0; i < n; i++ {
		if lens[i] == maxLen {
			count += counts[i]
		}
	}
	return count
}

// 688. 骑士在棋盘上的概率
// 在一个 n x n 的国际象棋棋盘上，一个骑士从单元格 (row, column) 开始，并尝试进行 k 次移动。行和列是 从 0 开始 的，所以左上单元格是 (0,0) ，右下单元格是 (n - 1, n - 1) 。
//
// 象棋骑士有8种可能的走法，如下图所示。每次移动在基本方向上是两个单元格，然后在正交方向上是一个单元格。
// 每次骑士要移动时，它都会随机从8种可能的移动中选择一种(即使棋子会离开棋盘)，然后移动到那里。
// 骑士继续移动，直到它走了 k 步或离开了棋盘。
// 返回 骑士在棋盘停止移动后仍留在棋盘上的概率 。
//
// 示例 1：
// 输入: n = 3, k = 2, row = 0, column = 0
// 输出: 0.0625
// 解释: 有两步(到(1,2)，(2,1))可以让骑士留在棋盘上。
// 在每一个位置上，也有两种移动可以让骑士留在棋盘上。
// 骑士留在棋盘上的总概率是0.0625。
//
// 示例 2：
// 输入: n = 1, k = 0, row = 0, column = 0
// 输出: 1.00000
//
// 提示:
// 1 <= n <= 25
// 0 <= k <= 100
// 0 <= row, column <= n
func knightProbability(n int, k int, row int, column int) float64 {
	dp := make([][][]float64, n)
	for i := 0; i < n; i++ {
		dp[i] = make([][]float64, n)
		for j := 0; j < n; j++ {
			dp[i][j] = make([]float64, k+1)
			dp[i][j][0] = 1.0
		}
	}
	dr := []int{2, 2, 1, 1, -1, -1, -2, -2}
	dc := []int{1, -1, 2, -2, 2, -2, 1, -1}
	// 逆向思维 从 棋盘上的 格子 反推到 row, col
	for l := 1; l <= k; l++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				for m := 0; m < 8; m++ {
					r, c := i+dr[m], j+dc[m]
					if inArea(r, c, n, n) {
						dp[r][c][l] += dp[i][j][l-1] / 8.0
					}
				}
			}
		}
	}
	return dp[row][column][k]
}
func inArea(row, col, rows, cols int) bool {
	return row >= 0 && row < rows && col >= 0 && col < cols
}

// 691. 贴纸拼词
// 我们有 n 种不同的贴纸。每个贴纸上都有一个小写的英文单词。
//
// 您想要拼写出给定的字符串 target ，方法是从收集的贴纸中切割单个字母并重新排列它们。如果你愿意，你可以多次使用每个贴纸，每个贴纸的数量是无限的。
//
// 返回你需要拼出 target 的最小贴纸数量。如果任务不可能，则返回 -1 。
// 注意：在所有的测试用例中，所有的单词都是从 1000 个最常见的美国英语单词中随机选择的，并且 target 被选择为两个随机单词的连接。
//
// 示例 1：
// 输入： stickers = ["with","example","science"], target = "thehat"
// 输出：3
// 解释：
// 我们可以使用 2 个 "with" 贴纸，和 1 个 "example" 贴纸。
// 把贴纸上的字母剪下来并重新排列后，就可以形成目标 “thehat“ 了。
// 此外，这是形成目标字符串所需的最小贴纸数量。
//
// 示例 2:
// 输入：stickers = ["notice","possible"], target = "basicbasic"
// 输出：-1
// 解释：我们不能通过剪切给定贴纸的字母来形成目标“basicbasic”。
//
// 提示:
// n == stickers.length
// 1 <= n <= 50
// 1 <= stickers[i].length <= 10
// 1 <= target <= 15
// stickers[i] 和 target 由小写英文单词组成
func minStickers(stickers []string, target string) int {

	m := 1 << len(target)
	dp := make([]int, m)
	for i := 0; i < m; i++ {
		dp[i] = -1
	}
	dp[0] = 0
	for _, sticker := range stickers {
		for status := 0; status < m; status++ {
			if dp[status] == -1 {
				continue
			}
			curStatus := status
			for _, c := range sticker {
				for i, t := range target {
					if c == t && curStatus&(1<<i) == 0 {
						curStatus |= 1 << i
						break
					}

				}
			}
			if dp[curStatus] == -1 {
				dp[curStatus] = dp[status] + 1
			} else {
				dp[curStatus] = min(dp[curStatus], dp[status]+1)
			}
		}
	}
	return dp[m-1]
}

// 1994. 好子集的数目
// 给你一个整数数组 nums 。如果 nums 的一个子集中，所有元素的乘积可以表示为一个或多个 互不相同的质数 的乘积，那么我们称它为 好子集 。
//
// 比方说，如果 nums = [1, 2, 3, 4] ：
// [2, 3] ，[1, 2, 3] 和 [1, 3] 是 好 子集，乘积分别为 6 = 2*3 ，6 = 2*3 和 3 = 3 。
// [1, 4] 和 [4] 不是 好 子集，因为乘积分别为 4 = 2*2 和 4 = 2*2 。
// 请你返回 nums 中不同的 好 子集的数目对 109 + 7 取余 的结果。
//
// nums 中的 子集 是通过删除 nums 中一些（可能一个都不删除，也可能全部都删除）元素后剩余元素组成的数组。如果两个子集删除的下标不同，那么它们被视为不同的子集。
//
// 示例 1：
// 输入：nums = [1,2,3,4]
// 输出：6
// 解释：好子集为：
// - [1,2]：乘积为 2 ，可以表示为质数 2 的乘积。
// - [1,2,3]：乘积为 6 ，可以表示为互不相同的质数 2 和 3 的乘积。
// - [1,3]：乘积为 3 ，可以表示为质数 3 的乘积。
// - [2]：乘积为 2 ，可以表示为质数 2 的乘积。
// - [2,3]：乘积为 6 ，可以表示为互不相同的质数 2 和 3 的乘积。
// - [3]：乘积为 3 ，可以表示为质数 3 的乘积。
//
// 示例 2：
// 输入：nums = [4,2,3,15]
// 输出：5
// 解释：好子集为：
// - [2]：乘积为 2 ，可以表示为质数 2 的乘积。
// - [2,3]：乘积为 6 ，可以表示为互不相同质数 2 和 3 的乘积。
// - [2,15]：乘积为 30 ，可以表示为互不相同质数 2，3 和 5 的乘积。
// - [3]：乘积为 3 ，可以表示为质数 3 的乘积。
// - [15]：乘积为 15 ，可以表示为互不相同质数 3 和 5 的乘积。
//
// 提示：
// 1 <= nums.length <= 105
// 1 <= nums[i] <= 30
func numberOfGoodSubsets(nums []int) int {
	// 所有质数
	primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

	freq := [31]int{}
	for _, num := range nums {
		freq[num]++
	}
	maxStatus := 1 << len(primes)
	dp := make([]int, maxStatus)
	dp[0] = 1
	for i := 0; i < freq[1]; i++ {
		dp[0] *= 2
		dp[0] %= Mod
	}
next:
	// 2 ~ 30 的所有数
	for num := 2; num <= 30; num++ {
		if freq[num] == 0 {
			continue
		}
		// 数组 i
		status := 0
		for j, prime := range primes {
			if num%(prime*prime) == 0 {
				continue next
			}
			if num%prime == 0 {
				status |= 1 << j
			}
		}
		// 动态规划 num 表示为 二进制 status
		for mask := maxStatus - 1; mask > 0; mask-- {
			if mask&status == status {
				dp[mask] += dp[mask^status] * freq[num] % Mod
				dp[mask] %= Mod
			}

		}
	}
	result := 0
	for _, f := range dp[1:] {
		result += f
		result %= Mod
	}
	return result
}

// 712. 两个字符串的最小ASCII删除和
// 给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和 。
//
// 示例 1:
// 输入: s1 = "sea", s2 = "eat"
// 输出: 231
// 解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
// 在 "eat" 中删除 "t" 并将 116 加入总和。
// 结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。
//
// 示例 2:
// 输入: s1 = "delete", s2 = "leet"
// 输出: 403
// 解释: 在 "delete" 中删除 "dee" 字符串变成 "let"，
// 将 100[d]+101[e]+101[e] 加入总和。在 "leet" 中删除 "e" 将 101[e] 加入总和。
// 结束时，两个字符串都等于 "let"，结果即为 100+101+101+101 = 403 。
// 如果改为将两个字符串转换为 "lee" 或 "eet"，我们会得到 433 或 417 的结果，比答案更大。
//
// 提示:
// 0 <= s1.length, s2.length <= 1000
// s1 和 s2 由小写英文字母组成
func minimumDeleteSum(s1 string, s2 string) int {
	m, n := len(s1), len(s2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	// 状态：使用 dp[i][j]表示s1前i个字符和s2前j个字符的最小和结果
	// 转移：如果当前字符相等 dp[i][j] = dp[i-1][j-1]
	// dp[i][j] =
	// min(dp[i-1][j]+ascii(s1[i]),dp[i][j-1]+ascii(s2[j])) 表示删除，加上ascii的值更新
	//
	// 注意边界，对于空串就是将其所有ascii值相加
	for i := 0; i < m; i++ {
		dp[i+1][0] = dp[i][0] + int(s1[i])
	}
	for j := 0; j < n; j++ {
		dp[0][j+1] = dp[0][j] + int(s2[j])
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if s1[i] == s2[j] {
				dp[i+1][j+1] = dp[i][j]
			} else {
				dp[i+1][j+1] = min(dp[i][j+1]+int(s1[i]), dp[i+1][j]+int(s2[j]))
			}
		}
	}
	return dp[m][n]
}

// 714. 买卖股票的最佳时机含手续费
// 给定一个整数数组 prices，其中 prices[i]表示第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。
//
// 你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
// 返回获得利润的最大值。
// 注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
//
// 示例 1：
// 输入：prices = [1, 3, 2, 8, 4, 9], fee = 2
// 输出：8
// 解释：能够达到的最大利润:
// 在此处买入 prices[0] = 1
// 在此处卖出 prices[3] = 8
// 在此处买入 prices[4] = 4
// 在此处卖出 prices[5] = 9
// 总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8
//
// 示例 2：
// 输入：prices = [1,3,7,5,10,3], fee = 3
// 输出：6
//
// 提示：
// 1 <= prices.length <= 5 * 104
// 1 <= prices[i] < 5 * 104
// 0 <= fee < 5 * 104
func maxProfitVI(prices []int, fee int) int {
	n := len(prices)

	// dp 表示第i天的利润 0 表示卖出, 1 表示买入
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0], dp[0][1] = 0, -prices[0]

	for i := 1; i < n; i++ {
		// 第i天的利润（卖出） = 第i-1天的利润（卖出） | 第i-1天的利润（买入） + i天的价格 - 手续费
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i]-fee)
		// 第i天的利润（买入） = 第i-1天的利润（买入） | 第i-1天的利润（卖出） - i天的价格
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
	}
	return max(dp[n-1][0], dp[n-1][1])
}

// 1706. 球会落何处
// 用一个大小为 m x n 的二维网格 grid 表示一个箱子。你有 n 颗球。箱子的顶部和底部都是开着的。
//
// 箱子中的每个单元格都有一个对角线挡板，跨过单元格的两个角，可以将球导向左侧或者右侧。
//
// 将球导向右侧的挡板跨过左上角和右下角，在网格中用 1 表示。
// 将球导向左侧的挡板跨过右上角和左下角，在网格中用 -1 表示。
// 在箱子每一列的顶端各放一颗球。每颗球都可能卡在箱子里或从底部掉出来。如果球恰好卡在两块挡板之间的 "V" 形图案，或者被一块挡导向到箱子的任意一侧边上，就会卡住。
//
// 返回一个大小为 n 的数组 answer ，其中 answer[i] 是球放在顶部的第 i 列后从底部掉出来的那一列对应的下标，如果球卡在盒子里，则返回 -1 。
//
// 示例 1：
// 输入：grid = [[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]]
// 输出：[1,-1,-1,-1,-1]
// 解释：示例如图：
// b0 球开始放在第 0 列上，最终从箱子底部第 1 列掉出。
// b1 球开始放在第 1 列上，会卡在第 2、3 列和第 1 行之间的 "V" 形里。
// b2 球开始放在第 2 列上，会卡在第 2、3 列和第 0 行之间的 "V" 形里。
// b3 球开始放在第 3 列上，会卡在第 2、3 列和第 0 行之间的 "V" 形里。
// b4 球开始放在第 4 列上，会卡在第 2、3 列和第 1 行之间的 "V" 形里。
//
// 示例 2：
// 输入：grid = [[-1]]
// 输出：[-1]
// 解释：球被卡在箱子左侧边上。
//
// 示例 3：
// 输入：grid = [[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]]
// 输出：[0,1,2,3,4,-1]
//
// 提示：
// m == grid.length
// n == grid[i].length
// 1 <= m, n <= 100
// grid[i][j] 为 1 或 -1
func findBall(grid [][]int) []int {
	m, n := len(grid), len(grid[0])
	result := make([]int, n)
	// 默认位置
	for j := 0; j < n; j++ {
		result[j] = j
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if result[j] == -1 {
				continue
			}
			// 球所在列
			col := result[j]

			if grid[i][col] == 1 && col+1 < n && grid[i][col+1] == 1 {
				// 右移
				result[j]++
			} else if grid[i][col] == -1 && col-1 >= 0 && grid[i][col-1] == -1 {
				// 左移
				result[j]--
			} else {
				result[j] = -1
			}
		}
	}

	return result
}

// 718. 最长重复子数组
// 给两个整数数组 nums1 和 nums2 ，返回 两个数组中 公共的 、长度最长的子数组的长度 。
//
// 示例 1：
// 输入：nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
// 输出：3
// 解释：长度最长的公共子数组是 [3,2,1] 。
//
// 示例 2：
// 输入：nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
// 输出：5
//
// 提示：
// 1 <= nums1.length, nums2.length <= 1000
// 0 <= nums1[i], nums2[i] <= 100
func findLength(nums1 []int, nums2 []int) int {
	m, n := len(nums1), len(nums2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	result := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if nums1[i] != nums2[j] {
				continue
			}
			dp[i+1][j+1] = dp[i][j] + 1
			result = max(result, dp[i+1][j+1])
		}
	}
	return result
}

// 730. 统计不同回文子序列
// 给定一个字符串 s，返回 s 中不同的非空「回文子序列」个数 。
//
// 通过从 s 中删除 0 个或多个字符来获得子序列。
//
// 如果一个字符序列与它反转后的字符序列一致，那么它是「回文字符序列」。
//
// 如果有某个 i , 满足 ai != bi ，则两个序列 a1, a2, ... 和 b1, b2, ... 不同。
//
// 注意：
// 结果可能很大，你需要对 109 + 7 取模 。
//
// 示例 1：
// 输入：s = 'bccb'
// 输出：6
// 解释：6 个不同的非空回文子字符序列分别为：'b', 'c', 'bb', 'cc', 'bcb', 'bccb'。
// 注意：'bcb' 虽然出现两次但仅计数一次。
//
// 示例 2：
// 输入：s = 'abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba'
// 输出：104860361
// 解释：共有 3104860382 个不同的非空回文子序列，104860361 对 109 + 7 取模后的值。
//
// 提示：
// 1 <= s.length <= 1000
// s[i] 仅包含 'a', 'b', 'c' 或 'd'
func countPalindromicSubsequences(s string) int {
	n := len(s)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		dp[i][i] = 1
	}

	for i := n - 2; i >= 0; i-- {
		for j := i + 1; j < n; j++ {
			if s[i] == s[j] {
				// i+1 ~ j-1 之间的回文字符串 + s[i] s[j]
				dp[i][j] = dp[i+1][j-1]<<1 + 2

				// 去除重复的 和 s[i] 相同的字符
				l, r := i+1, j-1
				c := s[i]
				for l <= r && s[l] != c {
					l++
				}
				for l <= r && s[r] != c {
					r--
				}
				if l == r {
					// 只有一个 c
					dp[i][j]--
				} else if l < r {
					// 2 个 c 减去 l + 1 ~ r -1 之间的 回文
					dp[i][j] -= dp[l+1][r-1] + 2
				}

			} else {
				dp[i][j] = dp[i][j-1] + dp[i+1][j] - dp[i+1][j-1]
			}
			if dp[i][j] >= 0 {
				dp[i][j] %= Mod
			} else {
				dp[i][j] += Mod
			}

		}
	}
	return dp[0][n-1]
}

// 剑指 Offer II 091. 粉刷房子
// 假如有一排房子，共 n 个，每个房子可以被粉刷成红色、蓝色或者绿色这三种颜色中的一种，你需要粉刷所有的房子并且使其相邻的两个房子颜色不能相同。
//
// 当然，因为市场上不同颜色油漆的价格不同，所以房子粉刷成不同颜色的花费成本也是不同的。每个房子粉刷成不同颜色的花费是以一个 n x 3 的正整数矩阵 costs 来表示的。
//
// 例如，costs[0][0] 表示第 0 号房子粉刷成红色的成本花费；costs[1][2] 表示第 1 号房子粉刷成绿色的花费，以此类推。
//
// 请计算出粉刷完所有房子最少的花费成本。
//
// 示例 1：
// 输入: costs = [[17,2,17],[16,16,5],[14,3,19]]
// 输出: 10
// 解释: 将 0 号房子粉刷成蓝色，1 号房子粉刷成绿色，2 号房子粉刷成蓝色。
//     最少花费: 2 + 5 + 3 = 10。
//
// 示例 2：
// 输入: costs = [[7,6,2]]
// 输出: 2
//
// 提示:
// costs.length == n
// costs[i].length == 3
// 1 <= n <= 100
// 1 <= costs[i][j] <= 20
// 注意：本题与主站 256 题相同：https://leetcode-cn.com/problems/paint-house/
func minCost(costs [][]int) int {
	n := len(costs)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 3)
	}
	// 第一个房间颜色
	for j := 0; j < 3; j++ {
		dp[0][j] = costs[0][j]
	}

	for i := 1; i < n; i++ {
		// 红色
		dp[i][0] = costs[i][0] + min(dp[i-1][1], dp[i-1][2])
		// 蓝色
		dp[i][1] = costs[i][1] + min(dp[i-1][0], dp[i-1][2])
		// 绿色
		dp[i][2] = costs[i][2] + min(dp[i-1][0], dp[i-1][1])
	}
	return min(dp[n-1][0], min(dp[n-1][1], dp[n-1][2]))
}

// 740. 删除并获得点数
// 给你一个整数数组 nums ，你可以对它进行一些操作。
//
// 每次操作中，选择任意一个 nums[i] ，删除它并获得 nums[i] 的点数。之后，你必须删除 所有 等于 nums[i] - 1 和 nums[i] + 1 的元素。
//
// 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
//
// 示例 1：
// 输入：nums = [3,4,2]
// 输出：6
// 解释：
// 删除 4 获得 4 个点数，因此 3 也被删除。
// 之后，删除 2 获得 2 个点数。总共获得 6 个点数。
//
// 示例 2：
// 输入：nums = [2,2,3,3,3,4]
// 输出：9
// 解释：
// 删除 3 获得 3 个点数，接着要删除两个 2 和 4 。
// 之后，再次删除 3 获得 3 个点数，再次删除 3 获得 3 个点数。
// 总共获得 9 个点数。
//
// 提示：
// 1 <= nums.length <= 2 * 104
// 1 <= nums[i] <= 104
func deleteAndEarn(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	maxNum := 0
	for _, num := range nums {
		maxNum = max(maxNum, num)
	}
	counts := make([]int, maxNum+1)
	for _, num := range nums {
		counts[num]++
	}
	dp := make([]int, maxNum+1)
	dp[1] = counts[1]
	result := dp[1]
	for i := 2; i <= maxNum; i++ {
		dp[i] = max(dp[i-1], dp[i-2]+i*counts[i])
		result = max(result, dp[i])
	}
	return result
}

// 741. 摘樱桃
// 一个N x N的网格(grid) 代表了一块樱桃地，每个格子由以下三种数字的一种来表示：
// 0 表示这个格子是空的，所以你可以穿过它。
// 1 表示这个格子里装着一个樱桃，你可以摘到樱桃然后穿过它。
// -1 表示这个格子里有荆棘，挡着你的路。
// 你的任务是在遵守下列规则的情况下，尽可能的摘到最多樱桃：
//
// 从位置 (0, 0) 出发，最后到达 (N-1, N-1) ，只能向下或向右走，并且只能穿越有效的格子（即只可以穿过值为0或者1的格子）；
// 当到达 (N-1, N-1) 后，你要继续走，直到返回到 (0, 0) ，只能向上或向左走，并且只能穿越有效的格子；
// 当你经过一个格子且这个格子包含一个樱桃时，你将摘到樱桃并且这个格子会变成空的（值变为0）；
// 如果在 (0, 0) 和 (N-1, N-1) 之间不存在一条可经过的路径，则没有任何一个樱桃能被摘到。
//
// 示例 1:
// 输入: grid =
// [[0, 1, -1],
//  [1, 0, -1],
//  [1, 1,  1]]
// 输出: 5
// 解释：
// 玩家从（0,0）点出发，经过了向下走，向下走，向右走，向右走，到达了点(2, 2)。
// 在这趟单程中，总共摘到了4颗樱桃，矩阵变成了[[0,1,-1],[0,0,-1],[0,0,0]]。
// 接着，这名玩家向左走，向上走，向上走，向左走，返回了起始点，又摘到了1颗樱桃。
// 在旅程中，总共摘到了5颗樱桃，这是可以摘到的最大值了。
//
// 说明:
// grid 是一个 N * N 的二维数组，N的取值范围是1 <= N <= 50。
// 每一个 grid[i][j] 都是集合 {-1, 0, 1}其中的一个数。
// 可以保证起点 grid[0][0] 和终点 grid[N-1][N-1] 的值都不会是 -1。
func cherryPickup(grid [][]int) int {
	n := len(grid)
	dp := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]int, n+1)
		for j := 0; j <= n; j++ {
			dp[i][j] = math.MinInt32
		}
	}
	dp[n-1][n-1] = grid[n-1][n-1]

	// k表示一共要走的步数， 通过一个循环递增，来降低一个维度，从而不需要使用三维数组 ,
	// 当前走第k步，一共要走2*n-2步（n-1）*2,下标的话就是2n-3，注意是倒序的
	for k := 2*n - 3; k >= 0; k-- {
		for i1 := max(0, k-n+1); i1 <= min(n-1, k); i1++ {
			for i2 := i1; i2 <= min(n-1, k); i2++ {
				// i1、j2的关联：一共要走k步，k<2*n，因此起点为Math.max(0,k-n+1),
				// 限定了i1的范围，因此 j1 = k -i1 =
				// k - (k-n+1) = n-1,也就是当i1取最大，j1的下标也只能为n-1
				// i2的优化：从i1开始计算，表明第二个人一定走在i1的下面
				j1, j2 := k-i1, k-i2
				if grid[i1][j1] == -1 || grid[i2][j2] == -1 {
					// 遇到荆棘
					dp[i1][i2] = math.MinInt32
				} else {
					if i1 != i2 || j1 != j2 {
						// 不重合在同一个点，则获取的最大值=A的格子+B的格子+AB往哪个方向走，也就是上一个状态是怎么来得，
						dp[i1][i2] = grid[i1][j1] + grid[i2][j2] +
							max(
								max(dp[i1][i2+1], dp[i1+1][i2]),
								max(dp[i1][i2], dp[i1+1][i2+1]))
					} else {
						// 重合在一个点，grid[i1][j1] == grid[i2][j2]，取一个即可，后面是4个方向
						dp[i1][i2] = grid[i1][j1] +
							max(
								max(dp[i1][i2+1], dp[i1+1][i2]),
								max(dp[i1][i2], dp[i1+1][i2+1]))
					}
				}
			}
		}
	}

	return max(0, dp[0][0])
}

// 764. 最大加号标志
// 在一个 n x n 的矩阵 grid 中，除了在数组 mines 中给出的元素为 0，其他每个元素都为 1。mines[i] = [xi, yi]表示 grid[xi][yi] == 0
//
// 返回  grid 中包含 1 的最大的 轴对齐 加号标志的阶数 。如果未找到加号标志，则返回 0 。
//
// 一个 k 阶由 1 组成的 “轴对称”加号标志 具有中心网格 grid[r][c] == 1 ，以及4个从中心向上、向下、向左、向右延伸，长度为 k-1，由 1 组成的臂。注意，只有加号标志的所有网格要求为 1 ，别的网格可能为 0 也可能为 1 。
//
// 示例 1：
// 输入: n = 5, mines = [[4, 2]]
// 输出: 2
// 解释: 在上面的网格中，最大加号标志的阶只能是2。一个标志已在图中标出。
//
// 示例 2：
// 输入: n = 1, mines = [[0, 0]]
// 输出: 0
// 解释: 没有加号标志，返回 0 。
//
// 提示：
// 1 <= n <= 500
// 1 <= mines.length <= 5000
// 0 <= xi, yi < n
// 每一对 (xi, yi) 都 不重复
func orderOfLargestPlusSign(n int, mines [][]int) int {
	if n == 0 {
		return 0
	}
	dp, matrix := make([][]int, n), make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, n)
		matrix[i] = make([]int, n)
		for j := 0; j < n; j++ {
			matrix[i][j] = 1
		}
	}

	for _, mine := range mines {
		row, col := mine[0], mine[1]
		matrix[row][col] = 0
	}
	count := 0
	// 左右
	for i := 0; i < n; i++ {
		count = 0
		// 左臂
		for j := 0; j < n; j++ {
			if matrix[i][j] == 1 {
				count++
			} else {
				count = 0
			}
			dp[i][j] = count
		}
		// 右臂
		count = 0
		for j := n - 1; j >= 0; j-- {
			if matrix[i][j] == 1 {
				count++
			} else {
				count = 0
			}
			dp[i][j] = min(dp[i][j], count)
		}
	}
	result := 0
	// 上下
	for j := 0; j < n; j++ {
		count = 0
		// 上臂
		for i := 0; i < n; i++ {
			if matrix[i][j] == 1 {
				count++
			} else {
				count = 0
			}
			dp[i][j] = min(dp[i][j], count)
		}
		// 下臂
		count = 0
		for i := n - 1; i >= 0; i-- {
			if matrix[i][j] == 1 {
				count++
			} else {
				count = 0
			}
			dp[i][j] = min(dp[i][j], count)
			result = max(result, dp[i][j])
		}
	}
	return result
}

// 801. 使序列递增的最小交换次数
// 我们有两个长度相等且不为空的整型数组 nums1 和 nums2 。在一次操作中，我们可以交换 nums1[i] 和 nums2[i]的元素。
//
// 例如，如果 nums1 = [1,2,3,8] ， nums2 =[5,6,7,4] ，你可以交换 i = 3 处的元素，得到 nums1 =[1,2,3,4] 和 nums2 =[5,6,7,8] 。
// 返回 使 nums1 和 nums2 严格递增 所需操作的最小次数 。
//
// 数组 arr 严格递增 且  arr[0] < arr[1] < arr[2] < ... < arr[arr.length - 1] 。
//
// 注意：
// 用例保证可以实现操作。
//
// 示例 1:
// 输入: nums1 = [1,3,5,4], nums2 = [1,2,3,7]
// 输出: 1
// 解释:
// 交换 A[3] 和 B[3] 后，两个数组如下:
// A = [1, 3, 5, 7] ， B = [1, 2, 3, 4]
// 两个数组均为严格递增的。
//
// 示例 2:
// 输入: nums1 = [0,3,5,8,9], nums2 = [2,1,4,6,9]
// 输出: 1
//
// 提示:
// 2 <= nums1.length <= 105
// nums2.length == nums1.length
// 0 <= nums1[i], nums2[i] <= 2 * 105
func minSwap(nums1 []int, nums2 []int) int {
	n := len(nums1)
	swap, noswap := 1, 0
	for i := 1; i < n; i++ {
		tmpSwap, tmpNoswap := math.MaxInt32, math.MaxInt32
		if nums1[i-1] < nums1[i] && nums2[i-1] < nums2[i] {
			// 不进行交换
			tmpNoswap = min(tmpNoswap, noswap)
			tmpSwap = min(tmpSwap, swap+1)

		}
		if nums1[i-1] < nums2[i] && nums2[i-1] < nums1[i] {
			// 进行交换
			tmpNoswap = min(tmpNoswap, swap)
			tmpSwap = min(tmpSwap, noswap+1)
		}
		swap = tmpSwap
		noswap = tmpNoswap
	}
	return min(swap, noswap)
}

// 902. 最大为 N 的数字组合
// 给定一个按 非递减顺序 排列的数字数组 digits 。你可以用任意次数 digits[i] 来写的数字。例如，如果 digits = ['1','3','5']，我们可以写数字，如 '13', '551', 和 '1351315'。
//
// 返回 可以生成的小于或等于给定整数 n 的正整数的个数 。
//
// 示例 1：
// 输入：digits = ["1","3","5","7"], n = 100
// 输出：20
// 解释：
// 可写出的 20 个数字是：
// 1, 3, 5, 7, 11, 13, 15, 17, 31, 33, 35, 37, 51, 53, 55, 57, 71, 73, 75, 77.
//
// 示例 2：
// 输入：digits = ["1","4","9"], n = 1000000000
// 输出：29523
// 解释：
// 我们可以写 3 个一位数字，9 个两位数字，27 个三位数字，
// 81 个四位数字，243 个五位数字，729 个六位数字，
// 2187 个七位数字，6561 个八位数字和 19683 个九位数字。
// 总共，可以使用D中的数字写出 29523 个整数。
//
// 示例 3:
// 输入：digits = ["7"], n = 8
// 输出：1
// 提示：
// 1 <= digits.length <= 9
// digits[i].length == 1
// digits[i] 是从 '1' 到 '9' 的数
// digits 中的所有值都 不同
// digits 按 非递减顺序 排列
// 1 <= n <= 109
func atMostNGivenDigitSet(digits []string, n int) int {
	s := strconv.Itoa(n)
	k, m := len(digits), len(s)
	dp := make([]int, m+1)
	dp[0] = 1

	for i := 0; i < m; i++ {
		num := int(s[m-i-1] - '0')
		for _, d := range digits {
			dnum, _ := strconv.Atoi(d)
			if dnum < num {
				// k^i
				dp[i+1] += int(math.Pow(float64(k), float64(i)))
			} else if dnum == num {
				// 前一位
				dp[i+1] += dp[i]
			}
		}

	}
	for i := 1; i < m; i++ {
		dp[m] += int(math.Pow(float64(k), float64(i)))
	}
	return dp[m]
}

// 1235. 规划兼职工作
// 你打算利用空闲时间来做兼职工作赚些零花钱。
//
// 这里有 n 份兼职工作，每份工作预计从 startTime[i] 开始到 endTime[i] 结束，报酬为 profit[i]。
//
// 给你一份兼职工作表，包含开始时间 startTime，结束时间 endTime 和预计报酬 profit 三个数组，请你计算并返回可以获得的最大报酬。
//
// 注意，时间上出现重叠的 2 份工作不能同时进行。
//
// 如果你选择的工作在时间 X 结束，那么你可以立刻进行在时间 X 开始的下一份工作。
//
// 示例 1：
// 输入：startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
// 输出：120
// 解释：
// 我们选出第 1 份和第 4 份工作，
// 时间范围是 [1-3]+[3-6]，共获得报酬 120 = 50 + 70。
//
// 示例 2：
// 输入：startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit = [20,20,100,70,60]
// 输出：150
// 解释：
// 我们选择第 1，4，5 份工作。
// 共获得报酬 150 = 20 + 70 + 60。
//
// 示例 3：
// 输入：startTime = [1,1,1], endTime = [2,3,4], profit = [5,6,4]
// 输出：6
//
// 提示：
// 1 <= startTime.length == endTime.length == profit.length <= 5 * 10^4
// 1 <= startTime[i] < endTime[i] <= 10^9
// 1 <= profit[i] <= 10^4
func jobScheduling(startTime []int, endTime []int, profit []int) int {
	n := len(startTime)
	jobList := make([]*Job, n)
	for i := 0; i < n; i++ {
		jobList[i] = &Job{startTime[i], endTime[i], profit[i]}
	}

	getEndTimeIndex := func(t int) int {
		low, high := 0, n-1
		for low < high {
			mid := (low + high) >> 1
			if jobList[mid].end <= t {
				low = mid + 1
			} else {
				high = mid
			}
		}
		return low
	}

	sort.Slice(jobList, func(i, j int) bool {
		return jobList[i].end < jobList[j].end
	})
	dp := make([]int, n+1)
	result := 0
	for i := 0; i < n; i++ {
		job := jobList[i]
		start := job.start
		// 二分查找
		index := getEndTimeIndex(start)
		dp[i+1] = max(dp[i], dp[index]+job.profit)
		// dp[i + 1] ;
		result = max(result, dp[i+1])
	}

	return result
}

type Job struct {
	start, end, profit int
}

// 790. 多米诺和托米诺平铺
// 有两种形状的瓷砖：一种是 2 x 1 的多米诺形，另一种是形如 "L" 的托米诺形。两种形状都可以旋转。
//
// 给定整数 n ，返回可以平铺 2 x n 的面板的方法的数量。返回对 109 + 7 取模 的值。
// 平铺指的是每个正方形都必须有瓷砖覆盖。两个平铺不同，当且仅当面板上有四个方向上的相邻单元中的两个，使得恰好有一个平铺有一个瓷砖占据两个正方形。
//
// 示例 1:
// 输入: n = 3
// 输出: 5
// 解释: 五种不同的方法如上所示。
//
// 示例 2:
// 输入: n = 1
// 输出: 1
//
// 提示：
// 1 <= n <= 1000
func numTilings(n int) int {
	// dp[n] = 2*dp[n-1] + dp[n-3]
	dp := make([]int, max(4, n+1))
	dp[0] = 1
	dp[1] = 1
	dp[2] = 2
	dp[3] = 5
	for i := 4; i <= n; i++ {
		dp[i] = (2*dp[i-1]%Mod + dp[i-3]) % Mod
	}
	return dp[n]
}

// 808. 分汤
// 有 A 和 B 两种类型 的汤。一开始每种类型的汤有 n 毫升。有四种分配操作：
//
// 提供 100ml 的 汤A 和 0ml 的 汤B 。
// 提供 75ml 的 汤A 和 25ml 的 汤B 。
// 提供 50ml 的 汤A 和 50ml 的 汤B 。
// 提供 25ml 的 汤A 和 75ml 的 汤B 。
// 当我们把汤分配给某人之后，汤就没有了。每个回合，我们将从四种概率同为 0.25 的操作中进行分配选择。如果汤的剩余量不足以完成某次操作，我们将尽可能分配。当两种类型的汤都分配完时，停止操作。
//
// 注意 不存在先分配 100 ml 汤B 的操作。
//
// 需要返回的值： 汤A 先分配完的概率 +  汤A和汤B 同时分配完的概率 / 2。返回值在正确答案 10-5 的范围内将被认为是正确的。
//
// 示例 1:
// 输入: n = 50
// 输出: 0.62500
// 解释:如果我们选择前两个操作，A 首先将变为空。
// 对于第三个操作，A 和 B 会同时变为空。
// 对于第四个操作，B 首先将变为空。
// 所以 A 变为空的总概率加上 A 和 B 同时变为空的概率的一半是 0.25 *(1 + 1 + 0.5 + 0)= 0.625。
//
// 示例 2:
// 输入: n = 100
// 输出: 0.71875
//
// 提示:
// 0 <= n <= 109
func soupServings(n int) float64 {
	tmp := 0
	if n%25 > 0 {
		tmp = 1
	}
	// dp(i, j) = 1/4 * (dp(i - 4, j) + dp(i - 3, j - 1) + dp(i - 2, j - 2) + dp(i - 1, j - 3))
	n = n / 25
	n += tmp
	if n >= 500 {
		return 1.0
	}
	dp := make([][]float64, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = make([]float64, n+1)
	}
	dp[0][0] = 0.5 // 特殊情况，0毫升A、0毫升B（同时分配完 1 * 0.5）
	for i := 1; i <= n; i++ {
		dp[i][0] = 0 // i毫升A，0毫升B，则B必定先分配完，不可能出现A先分配完或者A、B同时分配完
		dp[0][i] = 1 // 0毫升A，i毫升B，则A必定先分配完，概率为1
	}

	for i := 1; i <= n; i++ {
		a1 := max(i-4, 0) // 不足4，按4算（实际上是不足100，按100算，然后分配完了，没有剩余）
		a2 := max(i-3, 0) // 不足3，按3算（实际上是不足75，按75算，然后分配完了，没有剩余）
		a3 := max(i-2, 0) // 不足2，按2算（实际上是不足50，按75算，然后分配完了，没有剩余）
		a4 := max(i-1, 0) // 不足1，按1算（实际上是不足25，按25算，然后分配完了，没有剩余）
		for j := 1; j <= n; j++ {

			b1 := j
			b2 := max(j-1, 0) // 不足1，按1算（实际上是不足25，按25算，然后分配完了，没有剩余）
			b3 := max(j-2, 0) // 不足2，按2算（实际上是不足50，按75算，然后分配完了，没有剩余）
			b4 := max(j-3, 0) // 不足3，按3算（实际上是不足75，按75算，然后分配完了，没有剩余）
			// 状态转移方程：dp[i][j] = 0.25 * (dp[i-100][j] + dp[i-75][j-25] + dp[i-50][j-50] +
			// dp[i-75][j-25])
			// 将N缩小为原来的25分之一的转移方程：dp[i][j] = 0.25 * (dp[i-4][j] + dp[i-3][j-1] + dp[i-2][j-2] +
			// dp[i-3][j-1])
			dp[i][j] = 0.25 * (dp[a1][b1] + dp[a2][b2] + dp[a3][b3] + dp[a4][b4])
		}
	}

	return dp[n][n]
}
