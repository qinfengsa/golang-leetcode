package back

import (
	"fmt"
	"sort"
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
func readBinaryWatch(num int) []string {
	var result []string

	leds := [10]bool{}
	readBinaryWatchBack(num, 0, &leds, &result)
	return result
}

func readBinaryWatchBack(num int, start int, leds *[10]bool, list *[]string) {
	if num == 0 {
		hours, minutes := getTime(leds)
		if hours != -1 {
			*list = append(*list, fmt.Sprintf("%d:%02d", hours, minutes))
		}
		return
	}
	for i := start; i < 10; i++ {
		if leds[i] {
			continue
		}
		leds[i] = true
		readBinaryWatchBack(num-1, i+1, leds, list)
		leds[i] = false
	}
}

func getTime(leds *[10]bool) (int, int) {
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
