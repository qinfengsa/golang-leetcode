package stack

import (
	"container/list"
	"fmt"
	"log"
	"strconv"
)

func isValidTest() {
	s := "([)]"
	result := isValid(s)
	fmt.Println(result)
}

// 20. 有效的括号
// 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
//
// 有效字符串需满足：
//
// 左括号必须用相同类型的右括号闭合。
// 左括号必须以正确的顺序闭合。
// 注意空字符串可被认为是有效字符串。
//
// 示例 1: 输入: "()" 输出: true
// 示例 2: 输入: "()[]{}" 输出: true
// 示例 3: 输入: "(]" 输出: false
// 示例 4: 输入: "([)]" 输出: false
// 示例 5: 输入: "{[]}" 输出: true
func isValid(s string) bool {
	stack := list.New()
	for _, c := range s {
		switch c {
		case '(', '[', '{':
			stack.PushBack(c)
		case ')', ']', '}':
			{
				if stack.Len() == 0 {
					return false
				}
				c2 := stack.Back()
				stack.Remove(c2)
				if c2.Value != getLeftBracket(c) {
					return false
				}
			}
		}
	}
	return stack.Len() == 0
}

func getLeftBracket(c int32) int32 {
	result := c
	switch c {
	case ')':
		result = '('
	case ']':
		result = '['
	case '}':
		result = '{'
	}
	return result
}

func nextGreaterElementTest() {
	nums1, nums2 := []int{4, 1, 2}, []int{1, 3, 4, 2}
	result := nextGreaterElement(nums1, nums2)
	log.Print(result)
}

// 496. 下一个更大元素 I
// 给定两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。找到 nums1 中每个元素在 nums2 中的下一个比其大的值。
//
// nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。
//
// 示例 1:
// 输入: nums1 = [4,1,2], nums2 = [1,3,4,2]. 输出: [-1,3,-1]
// 解释:
//    对于num1中的数字4，你无法在第二个数组中找到下一个更大的数字，因此输出 -1。
//    对于num1中的数字1，第二个数组中数字1右边的下一个较大数字是 3。
//    对于num1中的数字2，第二个数组中没有下一个更大的数字，因此输出 -1。
//
// 示例 2:
// 输入: nums1 = [2,4], nums2 = [1,2,3,4]. 输出: [3,-1]
// 解释:
//    对于 num1 中的数字 2 ，第二个数组中的下一个较大数字是 3 。
//    对于 num1 中的数字 4 ，第二个数组中没有下一个更大的数字，因此输出 -1 。
//
// 提示：
// nums1和nums2中所有元素是唯一的。
// nums1和nums2 的数组大小都不超过1000。
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	indexMap := map[int]int{}
	for i, num := range nums1 {
		indexMap[num] = i
	}
	size1 := len(nums1)
	result := make([]int, size1)
	for i := 0; i < size1; i++ {
		result[i] = -1
	}
	stack := list.New()
	for _, num := range nums2 {
		for stack.Len() != 0 && stack.Back().Value.(int) < num {
			back := stack.Back()
			stack.Remove(back)
			if idx, ok := indexMap[back.Value.(int)]; ok {
				result[idx] = num
			}
		}
		stack.PushBack(num)
	}
	return result
}

// 682. 棒球比赛
// 你现在是一场采特殊赛制棒球比赛的记录员。这场比赛由若干回合组成，过去几回合的得分可能会影响以后几回合的得分。
//
// 比赛开始时，记录是空白的。你会得到一个记录操作的字符串列表 ops，其中 ops[i] 是你需要记录的第 i 项操作，ops 遵循下述规则：
//
// 整数 x - 表示本回合新获得分数 x
// "+" - 表示本回合新获得的得分是前两次得分的总和。题目数据保证记录此操作时前面总是存在两个有效的分数。
// "D" - 表示本回合新获得的得分是前一次得分的两倍。题目数据保证记录此操作时前面总是存在一个有效的分数。
// "C" - 表示前一次得分无效，将其从记录中移除。题目数据保证记录此操作时前面总是存在一个有效的分数。
// 请你返回记录中所有得分的总和。
//
//
//
// 示例 1：
// 输入：ops = ["5","2","C","D","+"] 输出：30
// 解释：
// "5" - 记录加 5 ，记录现在是 [5]
// "2" - 记录加 2 ，记录现在是 [5, 2]
// "C" - 使前一次得分的记录无效并将其移除，记录现在是 [5].
// "D" - 记录加 2 * 5 = 10 ，记录现在是 [5, 10].
// "+" - 记录加 5 + 10 = 15 ，记录现在是 [5, 10, 15].
// 所有得分的总和 5 + 10 + 15 = 30
//
// 示例 2：
// 输入：ops = ["5","-2","4","C","D","9","+","+"] 输出：27
// 解释：
// "5" - 记录加 5 ，记录现在是 [5]
// "-2" - 记录加 -2 ，记录现在是 [5, -2]
// "4" - 记录加 4 ，记录现在是 [5, -2, 4]
// "C" - 使前一次得分的记录无效并将其移除，记录现在是 [5, -2]
// "D" - 记录加 2 * -2 = -4 ，记录现在是 [5, -2, -4]
// "9" - 记录加 9 ，记录现在是 [5, -2, -4, 9]
// "+" - 记录加 -4 + 9 = 5 ，记录现在是 [5, -2, -4, 9, 5]
// "+" - 记录加 9 + 5 = 14 ，记录现在是 [5, -2, -4, 9, 5, 14]
// 所有得分的总和 5 + -2 + -4 + 9 + 5 + 14 = 27
//
// 示例 3：
// 输入：ops = ["1"] 输出：1
//
// 提示：
// 1 <= ops.length <= 1000
// ops[i] 为 "C"、"D"、"+"，或者一个表示整数的字符串。整数范围是 [-3 * 104, 3 * 104]
// 对于 "+" 操作，题目数据保证记录此操作时前面总是存在两个有效的分数
// 对于 "C" 和 "D" 操作，题目数据保证记录此操作时前面总是存在一个有效的分数
func calPoints(ops []string) int {
	stack := list.New()

	for _, op := range ops {
		switch op {
		case "+":
			{
				if stack.Len() >= 2 {
					back1 := stack.Back()
					num1 := back1.Value.(int)
					stack.Remove(back1)
					back2 := stack.Back()
					num2 := back2.Value.(int)
					stack.PushBack(num1)
					stack.PushBack(num1 + num2)
				}
			}
		case "D":
			{
				if stack.Len() > 0 {
					back := stack.Back()
					num := back.Value.(int)
					stack.PushBack(num * 2)
				}
			}
		case "C":
			{
				back := stack.Back()
				stack.Remove(back)
			}
		default:
			{
				num, _ := strconv.Atoi(op)
				stack.PushBack(num)
			}

		}
	}
	score := 0
	for stack.Len() > 0 {
		back := stack.Back()
		stack.Remove(back)
		score += back.Value.(int)
	}
	return score
}
