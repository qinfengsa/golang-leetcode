package dp

import (
	"fmt"
	"math"
	"sort"
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
