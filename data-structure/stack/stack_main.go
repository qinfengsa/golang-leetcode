package stack

import (
	"container/list"
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

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

// 84. 柱状图中最大的矩形
// 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
//
// 求在该柱状图中，能够勾勒出来的矩形的最大面积。
//
// 示例 1:
//
// 输入：heights = [2,1,5,6,2,3] 输出：10
// 解释：最大的矩形为图中红色区域，面积为 10
//
// 示例 2：
// 输入： heights = [2,4] 输出： 4
//
// 提示：
// 1 <= heights.length <=105
// 0 <= heights[i] <= 104
func largestRectangleArea(heights []int) int {
	stack := list.New()
	stack.PushBack(-1)
	// 如果当前高度比栈顶的高度小, 出栈, 把 大的移走
	maxArea, size := 0, len(heights)

	for i := 0; i < size; i++ {
		back := stack.Back()
		lastIdx := back.Value.(int)
		for lastIdx != -1 && heights[lastIdx] >= heights[i] {
			stack.Remove(back)
			tmpIdx := lastIdx
			back = stack.Back()
			lastIdx = back.Value.(int)
			maxArea = max(maxArea, heights[tmpIdx]*(i-lastIdx-1))
		}

		stack.PushBack(i)
	}
	back := stack.Back()
	lastIdx := back.Value.(int)
	for lastIdx != -1 {
		stack.Remove(back)
		tmpIdx := lastIdx
		back = stack.Back()
		lastIdx = back.Value.(int)
		maxArea = max(maxArea, heights[tmpIdx]*(size-lastIdx-1))
	}
	return maxArea
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 316. 去除重复字母
// 给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
//
// 注意：该题与 1081 https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters 相同
//
// 示例 1：
// 输入：s = "bcabc"
// 输出："abc"
//
// 示例 2：
// 输入：s = "cbacdcbc"
// 输出："acdb"
//
// 提示：
// 1 <= s.length <= 104
// s 由小写英文字母组成
func removeDuplicateLetters(s string) string {

	stack := list.New()
	visited := make([]bool, 26)
	lastIndex := make(map[byte]int)
	n := len(s)
	for i := 0; i < n; i++ {
		c := s[i]
		lastIndex[c] = i
	}
	for i := 0; i < n; i++ {
		c := s[i]
		if visited[c-'a'] {
			continue
		}
		for stack.Len() > 0 {
			back := stack.Back()

			prev := back.Value.(byte)
			// c 的字典序比 前面的 prev 小 并且 prev 不是最后一个（后面还有prev）
			if c < prev && lastIndex[prev] > i {
				stack.Remove(back)
				visited[prev-'a'] = false
			} else {
				break
			}

		}
		visited[c-'a'] = true
		stack.PushBack(c)
	}
	var builder strings.Builder
	for e := stack.Front(); e != nil; e = e.Next() {
		builder.WriteByte(e.Value.(byte))
	}
	return builder.String()
}

// 394. 字符串解码
// 给定一个经过编码的字符串，返回它解码后的字符串。
//
// 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
// 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
// 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
//
// 示例 1：
// 输入：s = "3[a]2[bc]"
// 输出："aaabcbc"
//
// 示例 2：
// 输入：s = "3[a2[c]]"
// 输出："accaccacc"
//
// 示例 3：
// 输入：s = "2[abc]3[cd]ef"
// 输出："abcabccdcdcdef"
//
// 示例 4：
// 输入：s = "abc3[cd]xyz"
// 输出："abccdcdcdxyz"
func decodeString(s string) string {
	strStack, numStack := list.New(), list.New()
	num := 0
	var builder strings.Builder

	for i := 0; i < len(s); i++ {
		if s[i] == '[' {
			numStack.PushBack(num)
			strStack.PushBack(builder.String())
			builder.Reset()
			num = 0
		} else if s[i] == ']' {
			numBack := numStack.Back()
			numStack.Remove(numBack)
			count := numBack.Value.(int)
			tmp := strings.Repeat(builder.String(), count)

			strBack := strStack.Back()
			strStack.Remove(strBack)
			lastStr := strBack.Value.(string)
			builder.Reset()
			builder.WriteString(lastStr)
			builder.WriteString(tmp)
		} else if isNum(s[i]) {
			num = num*10 + int(s[i]-'0')
		} else {
			builder.WriteByte(s[i])
		}
	}
	return builder.String()
}

func isNum(c byte) bool {
	return c >= '0' && c <= '9'
}

// 456. 132 模式
// 给你一个整数数组 nums ，数组中共有 n 个整数。132 模式的子序列 由三个整数 nums[i]、nums[j] 和 nums[k] 组成，并同时满足：i < j < k 和 nums[i] < nums[k] < nums[j] 。
//
// 如果 nums 中存在 132 模式的子序列 ，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：nums = [1,2,3,4]
// 输出：false
// 解释：序列中不存在 132 模式的子序列。
//
// 示例 2：
// 输入：nums = [3,1,4,2]
// 输出：true
// 解释：序列中有 1 个 132 模式的子序列： [1, 4, 2] 。
//
// 示例 3：
// 输入：nums = [-1,3,2,0]
// 输出：true
// 解释：序列中有 3 个 132 模式的的子序列：[-1, 3, 2]、[-1, 3, 0] 和 [-1, 2, 0] 。
//
// 提示：
// n == nums.length
// 1 <= n <= 2 * 105
// -109 <= nums[i] <= 109
func find132pattern(nums []int) bool {
	n := len(nums)
	minNums := make([]int, n)
	minNums[0] = nums[0]
	for i := 1; i < n; i++ {
		minNums[i] = min(minNums[i-1], nums[i])
	}
	stack := list.New()
	stack.PushBack(nums[n-1])

	for i := n - 2; i > 0; i-- {
		if nums[i] > minNums[i] {
			for stack.Len() > 0 {
				back := stack.Back()
				if back.Value.(int) <= minNums[i] {
					stack.Remove(back)
				} else {
					break
				}
			}
			// 当前栈顶元素 top(third) > minNums[i](first)
			if stack.Len() > 0 {
				back := stack.Back()
				// nums[i](second) > top(third)
				if nums[i] > back.Value.(int) {
					return true
				}
			}

			stack.PushBack(nums[i])
		}
	}
	return false
}

// 503. 下一个更大元素 II
// 给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。
//
// 示例 1:
// 输入: [1,2,1]
// 输出: [2,-1,2]
// 解释: 第一个 1 的下一个更大的数是 2；
// 数字 2 找不到下一个更大的数；
// 第二个 1 的下一个最大的数需要循环搜索，结果也是 2。
// 注意: 输入数组的长度不会超过 10000。
func nextGreaterElements(nums []int) []int {
	n := len(nums)
	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = -1
	}
	stack := list.New()
	for i := 0; i < n<<1; i++ {
		num := nums[i%n]
		for stack.Len() > 0 {
			back := stack.Back()
			index := back.Value.(int)
			if nums[index] >= num {
				break
			}
			result[index] = num
			stack.Remove(back)
		}

		stack.PushBack(i % n)
	}

	return result
}

// 591. 标签验证器
// 给定一个表示代码片段的字符串，你需要实现一个验证器来解析这段代码，并返回它是否合法。合法的代码片段需要遵守以下的所有规则：
//
// 代码必须被合法的闭合标签包围。否则，代码是无效的。
// 闭合标签（不一定合法）要严格符合格式：<TAG_NAME>TAG_CONTENT</TAG_NAME>。其中，<TAG_NAME>是起始标签，</TAG_NAME>是结束标签。起始和结束标签中的 TAG_NAME 应当相同。当且仅当 TAG_NAME 和 TAG_CONTENT 都是合法的，闭合标签才是合法的。
// 合法的 TAG_NAME 仅含有大写字母，长度在范围 [1,9] 之间。否则，该 TAG_NAME 是不合法的。
// 合法的 TAG_CONTENT 可以包含其他合法的闭合标签，cdata （请参考规则7）和任意字符（注意参考规则1）除了不匹配的<、不匹配的起始和结束标签、不匹配的或带有不合法 TAG_NAME 的闭合标签。否则，TAG_CONTENT 是不合法的。
// 一个起始标签，如果没有具有相同 TAG_NAME 的结束标签与之匹配，是不合法的。反之亦然。不过，你也需要考虑标签嵌套的问题。
// 一个<，如果你找不到一个后续的>与之匹配，是不合法的。并且当你找到一个<或</时，所有直到下一个>的前的字符，都应当被解析为 TAG_NAME（不一定合法）。
// cdata 有如下格式：<![CDATA[CDATA_CONTENT]]>。CDATA_CONTENT 的范围被定义成 <![CDATA[ 和后续的第一个 ]]>之间的字符。
// CDATA_CONTENT 可以包含任意字符。cdata 的功能是阻止验证器解析CDATA_CONTENT，所以即使其中有一些字符可以被解析为标签（无论合法还是不合法），也应该将它们视为常规字符。
// 合法代码的例子:
//
// 输入: "<DIV>This is the first line <![CDATA[<div>]]></DIV>"
// 输出: True
// 解释:
// 代码被包含在了闭合的标签内： <DIV> 和 </DIV> 。
// TAG_NAME 是合法的，TAG_CONTENT 包含了一些字符和 cdata 。
// 即使 CDATA_CONTENT 含有不匹配的起始标签和不合法的 TAG_NAME，它应该被视为普通的文本，而不是标签。
// 所以 TAG_CONTENT 是合法的，因此代码是合法的。最终返回True。
//
// 输入: "<DIV>>>  ![cdata[]] <![CDATA[<div>]>]]>]]>>]</DIV>"
// 输出: True
// 解释:
// 我们首先将代码分割为： start_tag|tag_content|end_tag 。
// start_tag -> "<DIV>"
// end_tag -> "</DIV>"
// tag_content 也可被分割为： text1|cdata|text2 。
// text1 -> ">>  ![cdata[]] "
// cdata -> "<![CDATA[<div>]>]]>" ，其中 CDATA_CONTENT 为 "<div>]>"
// text2 -> "]]>>]"
// start_tag 不是 "<DIV>>>" 的原因参照规则 6 。
// cdata 不是 "<![CDATA[<div>]>]]>]]>" 的原因参照规则 7 。
//
// 不合法代码的例子:
// 输入: "<A>  <B> </A>   </B>"
// 输出: False
// 解释: 不合法。如果 "<A>" 是闭合的，那么 "<B>" 一定是不匹配的，反之亦然。
//
// 输入: "<DIV>  div tag is not closed  <DIV>"
// 输出: False
//
// 输入: "<DIV>  unmatched <  </DIV>"
// 输出: False
//
// 输入: "<DIV> closed tags with invalid tag name  <b>123</b> </DIV>"
// 输出: False
//
// 输入: "<DIV> unmatched tags with invalid tag name  </1234567890> and <CDATA[[]]>  </DIV>"
// 输出: False
//
// 输入: "<DIV>  unmatched start tag <B>  and unmatched end tag </C>  </DIV>"
// 输出: False
//
// 注意:
// 为简明起见，你可以假设输入的代码（包括提到的任意字符）只包含数字, 字母, '<','>','/','!','[',']'和' '。
func isValidII(code string) bool {
	n := len(code)
	if n <= 5 {
		return false
	}
	if code[0] != '<' || code[n-1] != '>' {
		return false
	}
	stack := list.New()
	containsTag := false

	isValidTagName := func(tag string, end bool) bool {
		if len(tag) < 1 || len(tag) > 9 {
			return false
		}
		for _, c := range tag {
			if !unicode.IsUpper(c) {
				return false
			}
		}
		if end {
			if stack.Len() == 0 {
				return false
			}
			back := stack.Back()
			stack.Remove(back)
			backVal := back.Value.(string)
			if backVal != tag {
				return false
			}

		} else {
			stack.PushBack(tag)
			containsTag = true
		}
		return true
	}

	for i := 0; i < n; i++ {
		end := false
		if stack.Len() == 0 && containsTag {
			return false
		}
		var closeIdx int
		if code[i] == '<' {
			// !
			if stack.Len() > 0 && code[i+1] == '!' {
				closeIdx = strings.Index(code[i+1:], "]]>")
				if closeIdx < 0 {
					return false
				}
				closeIdx += i + 1
				cdata := code[i+2 : closeIdx]
				if len(cdata) < 7 || cdata[:7] != "[CDATA[" {
					return false
				}

			} else {
				if code[i+1] == '/' {
					i++
					end = true
				}
				closeIdx = strings.Index(code[i+1:], ">")
				if closeIdx < 0 {
					return false
				}
				closeIdx += i + 1
				tag := code[i+1 : closeIdx]
				if !isValidTagName(tag, end) {
					return false
				}
			}
			i = closeIdx
		}

	}

	return stack.Len() == 0 && containsTag
}

// 636. 函数的独占时间
// 有一个 单线程 CPU 正在运行一个含有 n 道函数的程序。每道函数都有一个位于  0 和 n-1 之间的唯一标识符。
//
// 函数调用 存储在一个 调用栈 上 ：当一个函数调用开始时，它的标识符将会推入栈中。而当一个函数调用结束时，它的标识符将会从栈中弹出。标识符位于栈顶的函数是 当前正在执行的函数 。每当一个函数开始或者结束时，将会记录一条日志，包括函数标识符、是开始还是结束、以及相应的时间戳。
//
// 给你一个由日志组成的列表 logs ，其中 logs[i] 表示第 i 条日志消息，该消息是一个按 "{function_id}:{"start" | "end"}:{timestamp}" 进行格式化的字符串。例如，"0:start:3" 意味着标识符为 0 的函数调用在时间戳 3 的 起始开始执行 ；而 "1:end:2" 意味着标识符为 1 的函数调用在时间戳 2 的 末尾结束执行。注意，函数可以 调用多次，可能存在递归调用 。
// 函数的 独占时间 定义是在这个函数在程序所有函数调用中执行时间的总和，调用其他函数花费的时间不算该函数的独占时间。例如，如果一个函数被调用两次，一次调用执行 2 单位时间，另一次调用执行 1 单位时间，那么该函数的 独占时间 为 2 + 1 = 3 。
// 以数组形式返回每个函数的 独占时间 ，其中第 i 个下标对应的值表示标识符 i 的函数的独占时间。
//
// 示例 1：
// 输入：n = 2, logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
// 输出：[3,4]
// 解释：
// 函数 0 在时间戳 0 的起始开始执行，执行 2 个单位时间，于时间戳 1 的末尾结束执行。
// 函数 1 在时间戳 2 的起始开始执行，执行 4 个单位时间，于时间戳 5 的末尾结束执行。
// 函数 0 在时间戳 6 的开始恢复执行，执行 1 个单位时间。
// 所以函数 0 总共执行 2 + 1 = 3 个单位时间，函数 1 总共执行 4 个单位时间。
//
// 示例 2：
// 输入：n = 1, logs = ["0:start:0","0:start:2","0:end:5","0:start:6","0:end:6","0:end:7"]
// 输出：[8]
// 解释：
// 函数 0 在时间戳 0 的起始开始执行，执行 2 个单位时间，并递归调用它自身。
// 函数 0（递归调用）在时间戳 2 的起始开始执行，执行 4 个单位时间。
// 函数 0（初始调用）恢复执行，并立刻再次调用它自身。
// 函数 0（第二次递归调用）在时间戳 6 的起始开始执行，执行 1 个单位时间。
// 函数 0（初始调用）在时间戳 7 的起始恢复执行，执行 1 个单位时间。
// 所以函数 0 总共执行 2 + 4 + 1 + 1 = 8 个单位时间。
//
// 示例 3：
// 输入：n = 2, logs = ["0:start:0","0:start:2","0:end:5","1:start:6","1:end:6","0:end:7"]
// 输出：[7,1]
// 解释：
// 函数 0 在时间戳 0 的起始开始执行，执行 2 个单位时间，并递归调用它自身。
// 函数 0（递归调用）在时间戳 2 的起始开始执行，执行 4 个单位时间。
// 函数 0（初始调用）恢复执行，并立刻调用函数 1 。
// 函数 1在时间戳 6 的起始开始执行，执行 1 个单位时间，于时间戳 6 的末尾结束执行。
// 函数 0（初始调用）在时间戳 7 的起始恢复执行，执行 1 个单位时间，于时间戳 7 的末尾结束执行。
// 所以函数 0 总共执行 2 + 4 + 1 = 7 个单位时间，函数 1 总共执行 1 个单位时间。
//
// 示例 4：
// 输入：n = 2, logs = ["0:start:0","0:start:2","0:end:5","1:start:7","1:end:7","0:end:8"]
// 输出：[8,1]
//
// 示例 5：
// 输入：n = 1, logs = ["0:start:0","0:end:0"]
// 输出：[1]
//
// 提示：
// 1 <= n <= 100
// 1 <= logs.length <= 500
// 0 <= function_id < n
// 0 <= timestamp <= 109
// 两个开始事件不会在同一时间戳发生
// 两个结束事件不会在同一时间戳发生
// 每道函数都有一个对应 "start" 日志的 "end" 日志
func exclusiveTime(n int, logs []string) []int {
	result := make([]int, n)

	stack := list.New()
	lastTime := 0

	for _, curlog := range logs {
		fu := strings.Split(curlog, ":")
		time, _ := strconv.Atoi(fu[2])
		idx, _ := strconv.Atoi(fu[0])
		if fu[1] == "start" {
			if stack.Len() > 0 {
				back := stack.Back()
				lastIdx := back.Value.(int)
				result[lastIdx] += time - lastTime
			}
			stack.PushBack(idx)
			lastTime = time
		} else {
			if stack.Len() > 0 {
				back := stack.Back()
				lastIdx := back.Value.(int)
				stack.Remove(back)
				result[lastIdx] += time - lastTime + 1
			}
			lastTime = time + 1
		}
	}

	return result
}

// 2104. 子数组范围和
// 给你一个整数数组 nums 。nums 中，子数组的 范围 是子数组中最大元素和最小元素的差值。
//
// 返回 nums 中 所有 子数组范围的 和 。
//
// 子数组是数组中一个连续 非空 的元素序列。
//
// 示例 1：
// 输入：nums = [1,2,3]
// 输出：4
// 解释：nums 的 6 个子数组如下所示：
// [1]，范围 = 最大 - 最小 = 1 - 1 = 0
// [2]，范围 = 2 - 2 = 0
// [3]，范围 = 3 - 3 = 0
// [1,2]，范围 = 2 - 1 = 1
// [2,3]，范围 = 3 - 2 = 1
// [1,2,3]，范围 = 3 - 1 = 2
// 所有范围的和是 0 + 0 + 0 + 1 + 1 + 2 = 4
//
// 示例 2：
// 输入：nums = [1,3,3]
// 输出：4
// 解释：nums 的 6 个子数组如下所示：
// [1]，范围 = 最大 - 最小 = 1 - 1 = 0
// [3]，范围 = 3 - 3 = 0
// [3]，范围 = 3 - 3 = 0
// [1,3]，范围 = 3 - 1 = 2
// [3,3]，范围 = 3 - 3 = 0
// [1,3,3]，范围 = 3 - 1 = 2
// 所有范围的和是 0 + 0 + 0 + 2 + 0 + 2 = 4
//
// 示例 3：
// 输入：nums = [4,-2,-3,4,1]
// 输出：59
// 解释：nums 中所有子数组范围的和是 59
//
// 提示：
// 1 <= nums.length <= 1000
// -109 <= nums[i] <= 109
func subArrayRanges(nums []int) int64 {
	n := len(nums)
	minLeft, maxLeft := make([]int, n), make([]int, n)

	maxStack, minStack := list.New(), list.New()
	for i, num := range nums {
		// min
		for minStack.Len() > 0 {
			back := minStack.Back()
			minIdx := back.Value.(int)
			minVal := nums[minIdx]
			if minVal <= num {
				break
			}
			minStack.Remove(back)
		}
		if minStack.Len() > 0 {
			back := minStack.Back()
			minIdx := back.Value.(int)
			minLeft[i] = minIdx
		} else {
			minLeft[i] = -1
		}
		minStack.PushBack(i)

		// max
		for maxStack.Len() > 0 {
			back := maxStack.Back()
			maxIdx := back.Value.(int)
			maxVal := nums[maxIdx]
			if maxVal > num {
				break
			}
			maxStack.Remove(back)
		}
		if maxStack.Len() > 0 {
			back := maxStack.Back()
			maxIdx := back.Value.(int)
			maxLeft[i] = maxIdx
		} else {
			maxLeft[i] = -1
		}
		maxStack.PushBack(i)

	}

	minRight, maxRight := make([]int, n), make([]int, n)
	minStack.Init()
	maxStack.Init()
	for i := n - 1; i >= 0; i-- {
		num := nums[i]

		// min
		for minStack.Len() > 0 {
			back := minStack.Back()
			minIdx := back.Value.(int)
			minVal := nums[minIdx]
			if minVal < num {
				break
			}
			minStack.Remove(back)
		}
		if minStack.Len() > 0 {
			back := minStack.Back()
			minIdx := back.Value.(int)
			minRight[i] = minIdx
		} else {
			minRight[i] = n
		}
		minStack.PushBack(i)

		// max
		for maxStack.Len() > 0 {
			back := maxStack.Back()
			maxIdx := back.Value.(int)
			maxVal := nums[maxIdx]
			if maxVal >= num {
				break
			}

			maxStack.Remove(back)
		}
		if maxStack.Len() > 0 {
			back := maxStack.Back()
			maxIdx := back.Value.(int)
			maxRight[i] = maxIdx
		} else {
			maxRight[i] = n
		}
		maxStack.PushBack(i)
	}
	fmt.Println(minLeft)
	fmt.Println(maxLeft)
	fmt.Println(minRight)
	fmt.Println(maxRight)
	var sumMax, sumMin int64
	for i, num := range nums {
		sumMax += int64(maxRight[i]-i) * int64(i-maxLeft[i]) * int64(num)
		sumMin += int64(minRight[i]-i) * int64(i-minLeft[i]) * int64(num)
	}
	return sumMax - sumMin
}

// 726. 原子的数量
// 给你一个字符串化学式 formula ，返回 每种原子的数量 。
//
// 原子总是以一个大写字母开始，接着跟随 0 个或任意个小写字母，表示原子的名字。
//
// 如果数量大于 1，原子后会跟着数字表示原子的数量。如果数量等于 1 则不会跟数字。
//
// 例如，"H2O" 和 "H2O2" 是可行的，但 "H1O2" 这个表达是不可行的。
// 两个化学式连在一起可以构成新的化学式。
//
// 例如 "H2O2He3Mg4" 也是化学式。
// 由括号括起的化学式并佐以数字（可选择性添加）也是化学式。
//
// 例如 "(H2O2)" 和 "(H2O2)3" 是化学式。
// 返回所有原子的数量，格式为：第一个（按字典序）原子的名字，跟着它的数量（如果数量大于 1），然后是第二个原子的名字（按字典序），跟着它的数量（如果数量大于 1），以此类推。
//
// 示例 1：
// 输入：formula = "H2O"
// 输出："H2O"
// 解释：原子的数量是 {'H': 2, 'O': 1}。
//
// 示例 2：
// 输入：formula = "Mg(OH)2"
// 输出："H2MgO2"
// 解释：原子的数量是 {'H': 2, 'Mg': 1, 'O': 2}。
//
// 示例 3：
// 输入：formula = "K4(ON(SO3)2)2"
// 输出："K4N2O14S4"
// 解释：原子的数量是 {'K': 4, 'N': 2, 'O': 14, 'S': 4}。
//
// 提示：
// 1 <= formula.length <= 1000
// formula 由英文字母、数字、'(' 和 ')' 组成
// formula 总是有效的化学式
func countOfAtoms(formula string) string {
	stack := list.New()
	n := len(formula)
	stack.PushBack(make(map[string]int))
	for i := 0; i < n; {
		c := formula[i]
		if c == '(' {
			stack.PushBack(make(map[string]int))
			i++
		} else if c == ')' {
			back := stack.Back()
			stack.Remove(back)
			i++
			// 括号后面是数字
			count := 0
			for i < n && unicode.IsDigit(rune(formula[i])) {
				count = count*10 + int(formula[i]-'0')
				i++
			}
			if count == 0 {
				count = 1
			}
			// 当前括号的原子放入前一个 map中
			curAtomMap := back.Value.(map[string]int)
			back = stack.Back()
			lastAtomMap := back.Value.(map[string]int)
			for name, v := range curAtomMap {
				lastAtomMap[name] += v * count
			}

		} else {
			start := i
			i++
			for i < n && unicode.IsLower(rune(formula[i])) {
				i++
			}
			name := formula[start:i]
			count := 0
			for i < n && unicode.IsDigit(rune(formula[i])) {
				count = count*10 + int(formula[i]-'0')
				i++
			}
			if count == 0 {
				count = 1
			}
			back := stack.Back()
			lastAtomMap := back.Value.(map[string]int)

			lastAtomMap[name] += count

		}

	}

	var builder strings.Builder

	back := stack.Back()
	lastAtomMap := back.Value.(map[string]int)

	atomList := make([]*Atom, 0)
	for name, cnt := range lastAtomMap {
		count := ""
		if cnt > 1 {
			count = strconv.Itoa(cnt)
		}
		atomList = append(atomList, &Atom{name: name, count: count})
	}
	sort.Slice(atomList, func(i, j int) bool {
		return atomList[i].name < atomList[j].name
	})

	for _, atom := range atomList {
		builder.WriteString(atom.name)
		builder.WriteString(atom.count)
	}
	return builder.String()
}

type Atom struct {
	name, count string
}

// 735. 行星碰撞
// 给定一个整数数组 asteroids，表示在同一行的行星。
//
// 对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。
//
// 找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。
//
// 示例 1：
// 输入：asteroids = [5,10,-5]
// 输出：[5,10]
// 解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
//
// 示例 2：
// 输入：asteroids = [8,-8]
// 输出：[]
// 解释：8 和 -8 碰撞后，两者都发生爆炸。
//
// 示例 3：
// 输入：asteroids = [10,2,-5]
// 输出：[10]
//
// 解释：2 和 -5 发生碰撞后剩下 -5 。10 和 -5 发生碰撞后剩下 10 。
//
// 提示：
// 2 <= asteroids.length <= 104
// -1000 <= asteroids[i] <= 1000
// asteroids[i] != 0
func asteroidCollision(asteroids []int) []int {
	stack := list.New()
collision:
	for _, asteroid := range asteroids {
		for stack.Len() > 0 {
			back := stack.Back()
			if asteroid < 0 && 0 < back.Value.(int) {
				if back.Value.(int) < -asteroid {
					stack.Remove(back)
					continue
				} else if back.Value.(int) == -asteroid {
					stack.Remove(back)
				}
				continue collision
			} else {
				break
			}
		}
		stack.PushBack(asteroid)
	}
	result := make([]int, stack.Len())
	n := stack.Len()
	for i := 0; i < n; i++ {
		front := stack.Front()
		stack.Remove(front)
		num := front.Value.(int)
		result[i] = num
	}
	return result
}
