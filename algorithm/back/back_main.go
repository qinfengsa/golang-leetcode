package back

import (
	"fmt"
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
