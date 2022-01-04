package back

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// 回溯算法

// 二进制手表顶部有 4 个 LED 代表 小时（0-11），底部的 6 个 LED 代表 分钟（0-59）。
//
// 每个 LED 代表一个 0 或 1，最低位在右侧。
//
// 例如，上面的二进制手表读取 “3:25”。
//
// 给定一个非负整数 n代表当前 LED 亮着的数量，返回所有可能的时间。
//
// 示例：
//
// 输入: n = 1
// 返回: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
//
// 提示：
//
// 输出的顺序没有要求。
// 小时不会以零开头，比如 “01:00”是不允许的，应为 “1:00”。
// 分钟必须由两位数组成，可能会以零开头，比如 “10:2”是无效的，应为 “10:02”。
// 超过表示范围（小时 0-11，分钟 0-59）的数据将会被舍弃，也就是说不会出现 "13:00", "0:61" 等时间。
func readBinaryWatch(turnedOn int) []string {
	var result []string

	leds := [10]bool{}

	var getTime = func(leds [10]bool) (int, int) {
		hours, minutes := 0, 0
		// 8 4 2 1
		for i := 0; i < 4; i++ {
			if leds[i] {
				hours += 1 << (3 - i)
			}
		}
		if hours >= 12 {
			return -1, -1
		}

		// 32 16 8 4 2 1
		for i := 4; i < 10; i++ {
			if leds[i] {
				minutes += 1 << (9 - i)
			}
		}
		if minutes > 59 {
			return -1, -1
		}

		return hours, minutes
	}

	var back func(num, start int)
	back = func(num, start int) {
		if num == 0 {
			hours, minutes := getTime(leds)
			if hours != -1 {
				result = append(result, fmt.Sprintf("%d:%02d", hours, minutes))
			}
			return
		}
		for i := start; i < 10; i++ {
			if leds[i] {
				continue
			}
			leds[i] = true
			back(num-1, i+1)
			leds[i] = false
		}
	}
	back(turnedOn, 0)

	return result
}

// 17. 电话号码的字母组合
// 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
//
// 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
//
// 示例 1：
// 输入：digits = "23" 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
//
// 示例 2：
// 输入：digits = "" 输出：[]
//
// 示例 3：
// 输入：digits = "2" 输出：["a","b","c"]
//
// 提示：
// 0 <= digits.length <= 4
// digits[i] 是范围 ['2', '9'] 的一个数字。
func letterCombinations(digits string) []string {
	size := len(digits)
	result := make([]string, 0)
	if size == 0 {
		return result
	}
	digitMap := map[byte][]byte{
		'2': {'a', 'b', 'c'},
		'3': {'d', 'e', 'f'},
		'4': {'g', 'h', 'i'},
		'5': {'j', 'k', 'l'},
		'6': {'m', 'n', 'o'},
		'7': {'p', 'q', 'r', 's'},
		'8': {'t', 'u', 'v'},
		'9': {'w', 'x', 'y', 'z'},
	}

	var back func(idx int, chars []byte)

	back = func(idx int, chars []byte) {
		if idx == size {
			result = append(result, string(chars))
			return
		}
		c := digits[idx]
		for _, b := range digitMap[c] {
			chars[idx] = b
			back(idx+1, chars)
		}
	}
	back(0, make([]byte, size))
	return result
}

// 22. 括号生成
// 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
//
// 示例 1：
// 输入：n = 3 输出：["((()))","(()())","(())()","()(())","()()()"]
//
// 示例 2：
// 输入：n = 1 输出：["()"]
//
// 提示：
// 1 <= n <= 8
func generateParenthesis(n int) []string {
	result := make([]string, 0)

	size := n << 1
	var back func(idx, leftCnt, rightCnt int, chars []byte)
	back = func(idx, leftCnt, rightCnt int, chars []byte) {
		if idx == size {
			result = append(result, string(chars))
			return
		}
		if leftCnt == rightCnt {
			// leftCnt == rightCnt 只能放左括号
			chars[idx] = '('
			back(idx+1, leftCnt+1, rightCnt, chars)
		} else if leftCnt == n {
			// leftCnt == n 只能放右括号
			chars[idx] = ')'
			back(idx+1, leftCnt, rightCnt+1, chars)
		} else {
			chars[idx] = '('
			back(idx+1, leftCnt+1, rightCnt, chars)
			chars[idx] = ')'
			back(idx+1, leftCnt, rightCnt+1, chars)
		}
	}

	back(0, 0, 0, make([]byte, size))

	return result
}

// 37. 解数独
// 编写一个程序，通过填充空格来解决数独问题。
//
// 数独的解法需 遵循如下规则：
//
// 数字 1-9 在每一行只能出现一次。
// 数字 1-9 在每一列只能出现一次。
// 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
// 数独部分空格内已填入了数字，空白格用 '.' 表示。
//
// 示例：
// 输入：board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
// 输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
// 解释：输入的数独如上图所示，唯一有效的解决方案如下所示：
//
// 提示：
// board.length == 9
// board[i].length == 9
// board[i][j] 是一位数字或者 '.'
// 题目数据 保证 输入数独仅有一个解
func solveSudoku(board [][]byte) {
	hashRows, hashCols, hashBoards := make([][]bool, 9), make([][]bool, 9), make([][]bool, 9)
	for i := 0; i < 9; i++ {
		hashRows[i] = make([]bool, 9)
		hashCols[i] = make([]bool, 9)
		hashBoards[i] = make([]bool, 9)
	}

	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				continue
			}
			numIdx := board[i][j] - '1'
			hashRows[i][numIdx] = true
			hashCols[j][numIdx] = true
			// 计算9宫格位置
			idx := i/3*3 + j/3
			hashBoards[idx][numIdx] = true
		}
	}

	// 回溯
	var back func(row, col int) bool

	back = func(i, j int) bool {
		if j == 9 {
			j = 0
			i++
			if i == 9 {
				return true
			}
		}
		if board[i][j] != '.' {
			return back(i, j+1)
		}
		for num := '1'; num <= '9'; num++ {
			// 计算9宫格位置
			idx := i/3*3 + j/3
			numIdx := num - '1'
			if hashRows[i][numIdx] || hashCols[j][numIdx] || hashBoards[idx][numIdx] {
				continue
			}
			hashRows[i][numIdx] = true
			hashCols[j][numIdx] = true
			hashBoards[idx][numIdx] = true
			board[i][j] = byte(num)
			if back(i, j+1) {
				return true
			}
			board[i][j] = '.'
			hashRows[i][numIdx] = false
			hashCols[j][numIdx] = false
			hashBoards[idx][numIdx] = false
		}
		return false
	}
	back(0, 0)
}

// 39. 组合总和
// 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
//
// candidates 中的数字可以无限制重复被选取。
// 说明：
// 所有数字（包括 target）都是正整数。
// 解集不能包含重复的组合。
//
// 示例 1：
// 输入：candidates = [2,3,6,7], target = 7,
// 所求解集为：
// [
//  [7],
//  [2,2,3]
// ]
//
// 示例 2：
// 输入：candidates = [2,3,5], target = 8,
// 所求解集为：
// [
//  [2,2,2,2],
//  [2,3,3],
//  [3,5]
// ]
//
// 提示：
// 1 <= candidates.length <= 30
// 1 <= candidates[i] <= 200
// candidate 中的每个元素都是独一无二的。
// 1 <= target <= 500
func combinationSum(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	size := len(candidates)
	result := make([][]int, 0)
	// 回溯算法
	var back func(idx, num int, nums []int)

	back = func(idx, num int, nums []int) {
		if num == target {
			tmp := make([]int, len(nums))
			copy(tmp, nums)
			result = append(result, tmp)
			return
		}
		for i := idx; i < size; i++ {
			if num+candidates[i] > target {
				break
			}
			l := len(nums)
			nums = append(nums, candidates[i])
			back(i, num+candidates[i], nums)
			nums = nums[0:l]
		}
	}

	back(0, 0, make([]int, 0))

	return result
}

// 40. 组合总和 II
// 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
//
// candidates 中的每个数字在每个组合中只能使用一次。
// 说明：
// 所有数字（包括目标数）都是正整数。
// 解集不能包含重复的组合。
//
// 示例 1:
// 输入: candidates = [10,1,2,7,6,1,5], target = 8,
// 所求解集为:
// [
//  [1, 7],
//  [1, 2, 5],
//  [2, 6],
//  [1, 1, 6]
// ]
//
// 示例 2:
// 输入: candidates = [2,5,2,1,2], target = 5,
// 所求解集为:
// [
//  [1,2,2],
//  [5]
// ]
func combinationSum2(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	size := len(candidates)
	result := make([][]int, 0)
	// 回溯算法
	var back func(idx, num int, nums []int)

	back = func(idx, num int, nums []int) {

		if num == target {
			tmp := make([]int, len(nums))
			copy(tmp, nums)
			result = append(result, tmp)
			return
		}
		for i := idx; i < size; i++ {
			if num+candidates[i] > target {
				break
			}
			// 防止重复
			if i > idx && candidates[i-1] == candidates[i] {
				continue
			}
			l := len(nums)
			nums = append(nums, candidates[i])
			back(i+1, num+candidates[i], nums)
			nums = nums[0:l]
		}
	}

	back(0, 0, make([]int, 0))

	return result
}

// 46. 全排列
// 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
//
// 示例 1：
// 输入：nums = [1,2,3] 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
//
// 示例 2：
// 输入：nums = [0,1] 输出：[[0,1],[1,0]]
//
// 示例 3：
// 输入：nums = [1] 输出：[[1]]
//
// 提示：
// 1 <= nums.length <= 6
// -10 <= nums[i] <= 10
// nums 中的所有整数 互不相同
func permute(nums []int) [][]int {
	result := make([][]int, 0)

	n := len(nums)
	// 回溯
	var back func(visited []bool, resNums []int)
	back = func(visited []bool, resNums []int) {
		if len(resNums) == n {
			tmpNums := make([]int, len(resNums))
			copy(tmpNums, resNums)
			result = append(result, tmpNums)
		}
		l := len(resNums)
		for i, num := range nums {
			if visited[i] {
				continue
			}
			visited[i] = true
			resNums = append(resNums, num)
			back(visited, resNums)
			resNums = resNums[0:l]
			visited[i] = false

		}
	}

	back(make([]bool, n), make([]int, 0))
	return result
}

// 47. 全排列 II
// 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
//
// 示例 1：
// 输入：nums = [1,1,2]  输出： [[1,1,2], [1,2,1], [2,1,1]]
//
// 示例 2：
// 输入：nums = [1,2,3] 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
//
// 提示：
// 1 <= nums.length <= 8
// -10 <= nums[i] <= 10
func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	result := make([][]int, 0)

	n := len(nums)
	// 回溯
	var back func(visited []bool, resNums []int)
	back = func(visited []bool, resNums []int) {
		if len(resNums) == n {
			tmpNums := make([]int, len(resNums))
			copy(tmpNums, resNums)
			result = append(result, tmpNums)
		}
		l := len(resNums)
		for i, num := range nums {
			if visited[i] {
				continue
			}
			last := i - 1
			for last >= 0 && !visited[last] {
				last--
			}
			// 找到上一个
			if last >= 0 && num == nums[last] {
				continue
			}
			visited[i] = true
			resNums = append(resNums, num)
			back(visited, resNums)
			resNums = resNums[0:l]
			visited[i] = false

		}
	}

	back(make([]bool, n), make([]int, 0))
	return result
}

// 51. N 皇后
// n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
//
// 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
// 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
//
// 示例 1：
// 输入：n = 4 输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
// 解释：如上图所示，4 皇后问题存在两个不同的解法。
//
// 示例 2：
// 输入：n = 1 输出：[["Q"]]
//
// 提示：
// 1 <= n <= 9
// 皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。
func solveNQueens(n int) [][]string {
	result := make([][]string, 0)

	cols := make([]bool, n)
	slashs, backslashs := make([]bool, 2*n-1), make([]bool, 2*n-1)
	// 反斜线

	var back func(i int, res [][]byte)

	back = func(i int, res [][]byte) {
		if i == n {
			tmp := make([]string, n)
			for row := 0; row < n; row++ {
				tmp[row] = string(res[row])
			}
			result = append(result, tmp)
			return
		}

		for j := 0; j < n; j++ {
			if cols[j] || slashs[i+j] || backslashs[n-1-i+j] {
				continue
			}
			cols[j] = true
			res[i][j] = 'Q'
			slashs[i+j] = true
			backslashs[n-1-i+j] = true
			back(i+1, res)
			res[i][j] = '.'
			cols[j] = false
			backslashs[n-1-i+j] = false
			slashs[i+j] = false
		}

	}
	res := make([][]byte, n)
	for i := 0; i < n; i++ {
		res[i] = make([]byte, n)
		for j := 0; j < n; j++ {
			res[i][j] = '.'
		}
	}

	back(0, res)

	return result
}

// 52. N皇后 II
// n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
//
// 给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。
//
// 示例 1：
// 输入：n = 4 输出：2
// 解释：如上图所示，4 皇后问题存在两个不同的解法。
//
// 示例 2：
// 输入：n = 1 输出：1
//
// 提示：
// 1 <= n <= 9
// 皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。
func totalNQueens(n int) int {
	result := 0

	cols := make([]bool, n)
	slashs, backslashs := make([]bool, 2*n-1), make([]bool, 2*n-1)
	// 反斜线

	var back func(i int)

	back = func(i int) {
		if i == n {
			result++
			return
		}

		for j := 0; j < n; j++ {
			if cols[j] || slashs[i+j] || backslashs[n-1-i+j] {
				continue
			}
			cols[j] = true
			slashs[i+j] = true
			backslashs[n-1-i+j] = true
			back(i + 1)
			cols[j] = false
			backslashs[n-1-i+j] = false
			slashs[i+j] = false
		}

	}
	back(0)

	return result
}

// 77. 组合
// 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
//
// 你可以按 任何顺序 返回答案。
// 示例 1：
// 输入：n = 4, k = 2
// 输出：
// [
//  [2,4],
//  [3,4],
//  [2,3],
//  [1,2],
//  [1,3],
//  [1,4],
// ]
//
// 示例 2：
// 输入：n = 1, k = 1 输出：[[1]]
//
// 提示：
// 1 <= n <= 20
// 1 <= k <= n
func combine(n int, k int) [][]int {
	result := make([][]int, 0)

	// 回溯
	var back func(visited []bool, nums []int, start, idx int)
	back = func(visited []bool, nums []int, start, idx int) {
		if idx == k {
			tmpNums := make([]int, k)
			copy(tmpNums, nums)
			result = append(result, tmpNums)
			return
		}
		for i := 0; i < n; i++ {
			if visited[i] {
				continue
			}
			visited[i] = true
			nums[idx] = i + 1
			back(visited, nums, i+1, idx+1)
			nums[idx] = 0
			visited[i] = false

		}
	}

	back(make([]bool, n), make([]int, k), 0, 0)

	return result
}

// 78. 子集
// 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
//
// 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
//
// 示例 1：
// 输入：nums = [1,2,3]
// 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
//
// 示例 2：
// 输入：nums = [0] 输出：[[],[0]]
//
// 提示：
// 1 <= nums.length <= 10
// -10 <= nums[i] <= 10
// nums 中的所有元素 互不相同
func subsets(nums []int) [][]int {
	n := len(nums)
	result := make([][]int, 0)
	for i := 0; i < 1<<n; i++ {
		tmp := make([]int, 0)

		for j := 0; j < n; j++ {
			if i&(1<<j) > 0 {
				tmp = append(tmp, nums[j])
			}
		}

		result = append(result, tmp)
	}
	return result
}

// 79. 单词搜索
// 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
//
// 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
//
// 示例 1：
// 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
// 输出：true
//
// 示例 2：
// 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
// 输出：true
//
// 示例 3：
// 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
// 输出：false
//
// 提示：
// m == board.length
// n = board[i].length
// 1 <= m, n <= 6
// 1 <= word.length <= 15
// board 和 word 仅由大小写英文字母组成
func exist(board [][]byte, word string) bool {
	m, n := len(board), len(board[0])
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	// var back func(visited []bool, nums []int, start, idx int)
	var back func(row, col, idx int) bool
	size := len(word)
	back = func(row, col, idx int) bool {
		if idx == size-1 {
			return board[row][col] == word[idx]
		}
		if board[row][col] == word[idx] {
			visited[row][col] = true

			for i := 0; i < 4; i++ {
				nextRow, nextCol := row+DirRow[i], col+DirCol[i]
				if !inArea(nextRow, nextCol, m, n) || visited[nextRow][nextCol] {
					continue
				}
				if back(nextRow, nextCol, idx+1) {
					return true
				}
			}
			visited[row][col] = false
		}
		return false
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if back(i, j, 0) {
				return true
			}
		}
	}

	return false
}

var (
	DirCol = []int{1, -1, 0, 0}
	DirRow = []int{0, 0, 1, -1}
)

func inArea(row, col, rows, cols int) bool {
	return row >= 0 && row < rows && col >= 0 && col < cols
}

// 89. 格雷编码
// 格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
//
// 给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。即使有多个不同答案，你也只需要返回其中一种。
//
// 格雷编码序列必须以 0 开头。
//
// 示例 1:
// 输入: 2 输出: [0,1,3,2]
// 解释:
// 00 - 0
// 01 - 1
// 11 - 3
// 10 - 2
//
// 对于给定的 n，其格雷编码序列并不唯一。
// 例如，[0,2,3,1] 也是一个有效的格雷编码序列。
// 00 - 0
// 10 - 2
// 11 - 3
// 01 - 1
//
// 示例 2:
// 输入: 0 输出: [0]
// 解释: 我们定义格雷编码序列必须以 0 开头。
//     给定编码总位数为 n 的格雷编码序列，其长度为 2n。当 n = 0 时，长度为 20 = 1。
//     因此，当 n = 0 时，其格雷编码序列为 [0]。
func grayCode(n int) []int {
	result := make([]int, 0)

	result = append(result, 0)
	if n == 0 {
		return result
	}
	head := 1
	for i := 0; i < n; i++ {
		// 遍历之前的元素 + head 1 10 100 1000
		for j := len(result) - 1; j >= 0; j-- {
			result = append(result, head+result[j])
		}
		head <<= 1
	}

	return result
}

// 90. 子集 II
// 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
//
// 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
//
// 示例 1：
// 输入：nums = [1,2,2] 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
//
// 示例 2：
// 输入：nums = [0] 输出：[[],[0]]
//
// 提示：
// 1 <= nums.length <= 10
// -10 <= nums[i] <= 10
func subsetsWithDup(nums []int) [][]int {
	n := len(nums)
	sort.Ints(nums)
	result := make([][]int, 0)

	var back func(start int, subset []int)

	back = func(start int, subset []int) {
		size := len(subset)
		tmp := make([]int, size)
		copy(tmp, subset)
		result = append(result, tmp)

		for i := start; i < n; i++ {
			// 去除重复
			if i > start && nums[i] == nums[i-1] {
				continue
			}
			subset = append(subset, nums[i])

			back(i+1, subset)
			subset = subset[0:size]
		}

	}

	back(0, make([]int, 0))
	return result
}

// 93. 复原 IP 地址
// 给定一个只包含数字的字符串，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。
//
// 有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
//
// 例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
//
// 示例 1：
// 输入：s = "25525511135" 输出：["255.255.11.135","255.255.111.35"]
//
// 示例 2：
// 输入：s = "0000" 输出：["0.0.0.0"]
//
// 示例 3：
// 输入：s = "1111" 输出：["1.1.1.1"]
//
// 示例 4：
// 输入：s = "010010" 输出：["0.10.0.10","0.100.1.0"]
//
// 示例 5：
// 输入：s = "101023"
// 输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
//
// 提示：
// 0 <= s.length <= 3000
// s 仅由数字组成
func restoreIpAddresses(s string) []string {

	var checkIp func(str string, count int) []string

	checkIp = func(str string, count int) []string {
		result := make([]string, 0)
		if count == 1 {
			if str[0] == '0' && len(str) > 1 {
				return result
			}
			num, _ := strconv.Atoi(str)
			if num <= 255 {
				result = append(result, str)
			}
			return result
		}
		if len(str) > count*3 {
			return result
		}

		for i := 1; i <= 3 && i < len(str); i++ {
			segment := str[:i]
			// 剪枝条件：不能以0开头，不能大于255
			if segment[0] == '0' && len(segment) > 1 {
				break
			}
			num, _ := strconv.Atoi(segment)
			if i == 3 && num > 255 {
				break
			}
			var builder strings.Builder
			builder.WriteString(strconv.Itoa(num))
			builder.WriteString(".")

			nextIps := checkIp(str[i+1:], count-1)
			for _, ip := range nextIps {
				var nextIp strings.Builder
				nextIp.WriteString(builder.String())
				nextIp.WriteString(ip)
				result = append(result, nextIp.String())
			}
		}
		return result
	}
	return checkIp(s, 4)
}

// 216. 组合总和 III
// 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
//
// 说明：
// 所有数字都是正整数。
// 解集不能包含重复的组合。
//
// 示例 1:
// 输入: k = 3, n = 7
// 输出: [[1,2,4]]
//
// 示例 2:
// 输入: k = 3, n = 9
// 输出: [[1,2,6], [1,3,5], [2,3,4]]
func combinationSum3(k int, n int) [][]int {
	result := make([][]int, 0)

	var back func(sum, start, index int, nums []int)

	back = func(sum, start, index int, nums []int) {
		if index == k && sum == 0 {
			tmp := make([]int, len(nums))
			copy(tmp, nums)
			result = append(result, tmp)
			return
		}
		if index >= k || sum <= 0 {
			return
		}
		for num := start; num <= 9; num++ {
			nums[index] = num
			back(sum-num, num+1, index+1, nums)
		}
	}

	back(n, 1, 0, make([]int, k))
	return result
}

// 282. 给表达式添加运算符
// 给定一个仅包含数字 0-9 的字符串 num 和一个目标值整数 target ，在 num 的数字之间添加 二元 运算符（不是一元）+、- 或 * ，返回所有能够得到目标值的表达式。
//
// 示例 1:
// 输入: num = "123", target = 6
// 输出: ["1+2+3", "1*2*3"]
//
// 示例 2:
// 输入: num = "232", target = 8
// 输出: ["2*3+2", "2+3*2"]
//
// 示例 3:
// 输入: num = "105", target = 5
// 输出: ["1*0+5","10-5"]
//
// 示例 4:
// 输入: num = "00", target = 0
// 输出: ["0+0", "0-0", "0*0"]
//
// 示例 5:
// 输入: num = "3456237490", target = 9191
// 输出: []
//
// 提示：
// 1 <= num.length <= 10
// num 仅含数字
// -231 <= target <= 231 - 1
func addOperators(num string, target int) []string {
	result := make([]string, 0)
	n := len(num)

	var back func(start, total, lastNum int, str string)

	back = func(start, total, lastNum int, str string) {
		if start == n {
			if total == target {
				result = append(result, str)
			}
			return
		}
		for i := start; i < n; i++ {
			if i > start && num[start] == '0' {
				break
			}
			curNum, _ := strconv.Atoi(num[start : i+1])
			if start == 0 {
				back(i+1, curNum, curNum, str+strconv.Itoa(curNum))
			} else {
				// +
				back(i+1, total+curNum, curNum, str+"+"+strconv.Itoa(curNum))
				// -
				back(i+1, total-curNum, -curNum, str+"-"+strconv.Itoa(curNum))
				// *
				back(
					i+1,
					total-lastNum+lastNum*curNum,
					lastNum*curNum,
					str+"*"+strconv.Itoa(curNum))
			}
		}
	}
	back(0, 0, 0, "")

	return result
}

// 301. 删除无效的括号
// 给你一个由若干括号和字母组成的字符串 s ，删除最小数量的无效括号，使得输入的字符串有效。
//
// 返回所有可能的结果。答案可以按 任意顺序 返回。
//
// 示例 1：
// 输入：s = "()())()"
// 输出：["(())()","()()()"]
//
// 示例 2：
// 输入：s = "(a)())()"
// 输出：["(a())()","(a)()()"]
//
// 示例 3：
// 输入：s = ")("
// 输出：[""]
//
// 提示：
// 1 <= s.length <= 25
// s 由小写英文字母以及括号 '(' 和 ')' 组成
// s 中至多含 20 个括号
func removeInvalidParentheses(s string) []string {
	result := make([]string, 0)
	left, right := 0, 0
	n := len(s)
	for i := 0; i < n; i++ {
		if s[i] == '(' {
			left++
		} else if s[i] == ')' {
			if left > 0 {
				left--
			} else {
				right++
			}
		}
	}
	resultMap := make(map[string]bool)
	var dfs func(idx, leftCount, rightCount, leftRemove, rightRemove int, str string)

	dfs = func(idx, leftCount, rightCount, leftRemove, rightRemove int, str string) {
		if idx == n {
			if leftRemove == 0 && rightRemove == 0 {
				resultMap[str] = true
			}
			return
		}
		c := s[idx]
		// 删除当前字符
		if c == '(' && leftRemove > 0 {
			dfs(idx+1, leftCount, rightCount, leftRemove-1, rightRemove, str)
		} else if c == ')' && rightRemove > 0 {
			dfs(idx+1, leftCount, rightCount, leftRemove, rightRemove-1, str)
		}
		// 保留当前字符
		if c != '(' && c != ')' {
			dfs(idx+1, leftCount, rightCount, leftRemove, rightRemove, str+string(c))
		} else if c == '(' {
			// 左括号
			dfs(idx+1, leftCount+1, rightCount, leftRemove, rightRemove, str+string(c))
		} else if leftCount > rightCount {
			// 有效右括号
			dfs(idx+1, leftCount, rightCount+1, leftRemove, rightRemove, str+string(c))
		}
	}
	dfs(0, 0, 0, left, right, "")

	for k, _ := range resultMap {
		result = append(result, k)
	}
	return result
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 306. 累加数
// 累加数是一个字符串，组成它的数字可以形成累加序列。
//
// 一个有效的累加序列必须至少包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。
// 给定一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是累加数。
// 说明: 累加序列里的数不会以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。
//
// 示例 1:
// 输入: "112358"
// 输出: true
// 解释: 累加序列为: 1, 1, 2, 3, 5, 8 。1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8
//
// 示例 2:
// 输入: "199100199"
// 输出: true
// 解释: 累加序列为: 1, 99, 100, 199。1 + 99 = 100, 99 + 100 = 199
// 进阶:
// 你如何处理一个溢出的过大的整数输入?
func isAdditiveNumber(num string) bool {
	n := len(num)
	if n < 3 {
		return false
	}
	// a, b , c 第1,2,3个数字开始的索引
	var additive func(a, b, c int) bool

	additive = func(a, b, c int) bool {
		// 0开头的数字无效
		if (b-a > 1 && num[a] == '0') || (c-b > 1 && num[b] == '0') {
			return false
		}
		numLen := max(b-a, c-b)
		if c+numLen > n {
			return false
		}
		num1, _ := strconv.Atoi(num[a:b])
		num2, _ := strconv.Atoi(num[b:c])
		num3, _ := strconv.Atoi(num[c : c+numLen])
		if num3 > 0 && num[c] == '0' {
			return false
		}
		if num1+num2 == num3 {
			if c+numLen == n {
				return true
			}
			return additive(b, c, c+numLen)
		}
		if c+numLen+1 > n {
			return false
		}
		num4, _ := strconv.Atoi(num[c : c+numLen+1])

		if num4 > 0 && num[c] == '0' {
			return false
		}
		if num1+num2 == num4 {
			if c+numLen+1 == n {
				return true
			}
			return additive(b, c, c+numLen+1)
		}

		return false
	}

	for i := 1; i <= n>>1; i++ {
		for j := i + 1; j < n; j++ {
			if additive(0, i, j) {
				return true
			}
		}
	}

	return false
}

// 473. 火柴拼正方形
// 还记得童话《卖火柴的小女孩》吗？现在，你知道小女孩有多少根火柴，请找出一种能使用所有火柴拼成一个正方形的方法。不能折断火柴，可以把火柴连接起来，并且每根火柴都要用到。
//
// 输入为小女孩拥有火柴的数目，每根火柴用其长度表示。输出即为是否能用所有的火柴拼成正方形。
//
// 示例 1:
// 输入: [1,1,2,2,2]
// 输出: true
// 解释: 能拼成一个边长为2的正方形，每边两根火柴。
//
// 示例 2:
// 输入: [3,3,3,3,4]
// 输出: false
// 解释: 不能用所有火柴拼成一个正方形。
//
// 注意:
// 给定的火柴长度和在 0 到 10^9之间。
// 火柴数组的长度不超过15。
func makesquare(matchsticks []int) bool {
	n := len(matchsticks)
	if n < 4 {
		return false
	}
	maxStick, sum := 0, 0
	for _, stick := range matchsticks {
		sum += stick
		maxStick = max(maxStick, stick)
	}
	if sum%4 != 0 {
		return false
	}
	// 边长
	side := sum >> 2
	if maxStick > side {
		return false
	}
	sort.Ints(matchsticks)
	visited := make([]bool, n)
	var back func(start, count, num int) bool

	back = func(start, count, num int) bool {
		if count == 4 {
			return true
		} else if num > side {
			return false
		} else if num == side {
			return back(0, count+1, 0)
		}
		last := -1
		for i := start; i < n; i++ {
			if visited[i] {
				continue
			}
			// 防止重复回溯
			if matchsticks[i] == last {
				continue
			}
			last = matchsticks[i]
			visited[i] = true
			if back(i+1, count, num+matchsticks[i]) {
				return true
			}
			visited[i] = false
		}

		return false
	}

	return back(0, 0, 0)
}

// 491. 递增子序列
// 给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。
//
// 数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。
//
// 示例 1：
// 输入：nums = [4,6,7,7]
// 输出：[[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]
//
// 示例 2：
// 输入：nums = [4,4,3,2,1]
// 输出：[[4,4]]
//
// 提示：
// 1 <= nums.length <= 15
// -100 <= nums[i] <= 100
func findSubsequences(nums []int) [][]int {
	result := make([][]int, 0)
	n := len(nums)
	if n < 2 {
		return result
	}
	var back func(index int, arrs []int)

	back = func(index int, arrs []int) {
		l := len(arrs)
		if index == n {
			if l >= 2 {
				tmp := make([]int, l)
				copy(tmp, arrs)
				result = append(result, tmp)
			}
			return
		}
		// 把第 index 个元素加进 arrs 中
		if l == 0 || arrs[l-1] <= nums[index] {
			tmp := append(arrs, nums[index])
			back(index+1, tmp)
		}
		if index > 0 && l > 0 && arrs[l-1] == nums[index] {
			return
		}
		// 不加 arrs
		back(index+1, arrs)
	}

	back(0, make([]int, 0))
	return result
}

// 526. 优美的排列
// 假设有从 1 到 n 的 n 个整数。用这些整数构造一个数组 perm（下标从 1 开始），只要满足下述条件 之一 ，该数组就是一个 优美的排列 ：
//
// perm[i] 能够被 i 整除
// i 能够被 perm[i] 整除
// 给你一个整数 n ，返回可以构造的 优美排列 的 数量 。
//
// 示例 1：
// 输入：n = 2 输出：2
// 解释：
// 第 1 个优美的排列是 [1,2]：
//    - perm[1] = 1 能被 i = 1 整除
//    - perm[2] = 2 能被 i = 2 整除
// 第 2 个优美的排列是 [2,1]:
//    - perm[1] = 2 能被 i = 1 整除
//    - i = 2 能被 perm[2] = 1 整除
//
// 示例 2：
// 输入：n = 1 输出：1
//
// 提示：
// 1 <= n <= 15
func countArrangement(n int) int {
	result := 0
	// 预处理
	permList := make([][]int, n+1)

	for i := 1; i <= n; i++ {
		for j := 1; j <= n; j++ {
			if i%j == 0 || j%i == 0 {
				permList[i] = append(permList[i], j)
			}
		}
	}

	visited := make([]bool, n+1)
	var back func(index int)

	back = func(index int) {
		if index > n {
			result++
			return
		}
		for _, num := range permList[index] {
			if visited[num] {
				continue
			}
			visited[num] = true
			back(index + 1)
			visited[num] = false
		}

	}
	back(1)
	return result
}

// 638. 大礼包
// 在 LeetCode 商店中， 有 n 件在售的物品。每件物品都有对应的价格。然而，也有一些大礼包，每个大礼包以优惠的价格捆绑销售一组物品。
//
// 给你一个整数数组 price 表示物品价格，其中 price[i] 是第 i 件物品的价格。另有一个整数数组 needs 表示购物清单，其中 needs[i] 是需要购买第 i 件物品的数量。
//
// 还有一个数组 special 表示大礼包，special[i] 的长度为 n + 1 ，其中 special[i][j] 表示第 i 个大礼包中内含第 j 件物品的数量，且 special[i][n] （也就是数组中的最后一个整数）为第 i 个大礼包的价格。
//
// 返回 确切 满足购物清单所需花费的最低价格，你可以充分利用大礼包的优惠活动。你不能购买超出购物清单指定数量的物品，即使那样会降低整体价格。任意大礼包可无限次购买。
//
// 示例 1：
// 输入：price = [2,5], special = [[3,0,5],[1,2,10]], needs = [3,2]
// 输出：14
// 解释：有 A 和 B 两种物品，价格分别为 ¥2 和 ¥5 。
// 大礼包 1 ，你可以以 ¥5 的价格购买 3A 和 0B 。
// 大礼包 2 ，你可以以 ¥10 的价格购买 1A 和 2B 。
// 需要购买 3 个 A 和 2 个 B ， 所以付 ¥10 购买 1A 和 2B（大礼包 2），以及 ¥4 购买 2A 。
//
// 示例 2：
// 输入：price = [2,3,4], special = [[1,1,0,4],[2,2,1,9]], needs = [1,2,1]
// 输出：11
// 解释：A ，B ，C 的价格分别为 ¥2 ，¥3 ，¥4 。
// 可以用 ¥4 购买 1A 和 1B ，也可以用 ¥9 购买 2A ，2B 和 1C 。
// 需要买 1A ，2B 和 1C ，所以付 ¥4 买 1A 和 1B（大礼包 1），以及 ¥3 购买 1B ， ¥4 购买 1C 。
// 不可以购买超出待购清单的物品，尽管购买大礼包 2 更加便宜。
//
// 提示：
// n == price.length
// n == needs.length
// 1 <= n <= 6
// 0 <= price[i] <= 10
// 0 <= needs[i] <= 10
// 1 <= special.length <= 100
// special[i].length == n + 1
// 0 <= special[i][j] <= 50
func shoppingOffers(price []int, special [][]int, needs []int) int {
	result := math.MaxInt32
	sum := 0
	n := len(needs)
	valid := func() bool {
		for _, need := range needs {
			if need < 0 {
				return false
			}
		}
		return true
	}
	// 计算需要几个特别礼包
	calSpecialNum := func(curSpecial, needs []int) int {
		maxNum := math.MaxInt32
		for i := 0; i < len(needs); i++ {
			if needs[i] < curSpecial[i] {
				return 0
			}
			if curSpecial[i] != 0 {
				num := needs[i] / curSpecial[i]
				maxNum = min(maxNum, num)
			}
		}
		return maxNum
	}

	var back func(start int)

	back = func(start int) {
		if !valid() {
			return
		}
		tmpSum := sum
		for i := 0; i < len(needs); i++ {
			sum += needs[i] * price[i]
		}
		result = min(result, sum)
		sum = tmpSum
		for i := start; i < len(special); i++ {
			curSpec := special[i]
			// 特别礼包数量
			specNum := calSpecialNum(curSpec, needs)

			for j := 1; j <= specNum; j++ {
				tmpNeeds := make([]int, n)
				copy(tmpNeeds, needs)
				for k := 0; k < n; k++ {
					needs[k] -= j * curSpec[k]
				}
				// curSpec[n]
				sum += curSpec[n] * j
				back(i + 1)
				needs = tmpNeeds
				sum = tmpSum
			}

		}
	}

	back(0)

	return result
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
