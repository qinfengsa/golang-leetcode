package dp

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
