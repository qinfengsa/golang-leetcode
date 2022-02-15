package math

import (
	"container/list"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// 150. 逆波兰表达式求值
// 根据 逆波兰表示法，求表达式的值。
//
// 有效的算符包括 +、-、*、/ 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
//
// 说明：
// 整数除法只保留整数部分。
// 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
//
// 示例 1：
// 输入：tokens = ["2","1","+","3","*"] 输出：9
// 解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
//
// 示例 2：
// 输入：tokens = ["4","13","5","/","+"] 输出：6
// 解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
//
// 示例 3：
// 输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"] 输出：22
// 解释：
// 该算式转化为常见的中缀算术表达式为：
//   ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
// = ((10 * (6 / (12 * -11))) + 17) + 5
// = ((10 * (6 / -132)) + 17) + 5
// = ((10 * 0) + 17) + 5
// = (0 + 17) + 5
// = 17 + 5
// = 22
//
// 提示：
// 1 <= tokens.length <= 104
// tokens[i] 要么是一个算符（"+"、"-"、"*" 或 "/"），要么是一个在范围 [-200, 200] 内的整数
//
// 逆波兰表达式：
// 逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。
//
// 平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 。
// 该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
// 逆波兰表达式主要有以下两个优点：
//
// 去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。
// 适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。
func evalRPN(tokens []string) int {
	result := 0
	stack := list.New()
	for _, token := range tokens {
		if len(token) == 0 {
			break
		}
		if len(token) == 1 && isCalculate(token[0]) {
			back := stack.Back()
			stack.Remove(back)
			b := back.Value.(int)

			back = stack.Back()
			stack.Remove(back)
			a := back.Value.(int)
			num := calculateNum(a, b, token[0])
			stack.PushBack(num)
		} else {
			v, _ := strconv.Atoi(token)
			stack.PushBack(v)
		}
	}
	if stack.Len() != 0 {
		back := stack.Back()
		result = back.Value.(int)
	}

	return result
}

func isCalculate(c byte) bool {
	switch c {
	case '+', '-', '*', '/':
		return true
	default:
		return false
	}
}

func calculateNum(a, b int, c byte) int {
	result := 0
	switch c {
	case '+':
		result = a + b
	case '-':
		result = a - b
	case '*':
		result = a * b
	case '/':
		result = a / b
	default:
		break

	}
	return result
}

type operation struct {
	op       byte
	priority int
}

// 224. 基本计算器
// 给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。
//
// 示例 1：
// 输入：s = "1 + 1" 输出：2
//
// 示例 2：
// 输入：s = " 2-1 + 2 " 输出：3
//
// 示例 3：
// 输入：s = "(1+(4+5+2)-3)+(6+8)" 输出：23
//
// 提示：
// 1 <= s.length <= 3 * 105
// s 由数字、'+'、'-'、'('、')'、和 ' ' 组成
// s 表示一个有效的表达式
//
// 227. 基本计算器 II
// 给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。
// 整数除法仅保留整数部分。
//
// 示例 1：
// 输入：s = "3+2*2" 输出：7
//
// 示例 2：
// 输入：s = " 3/2 " 输出：1
//
// 示例 3：
// 输入：s = " 3+5 / 2 " 输出：5
//
// 提示：
// 1 <= s.length <= 3 * 105
// s 由整数和算符 ('+', '-', '*', '/') 组成，中间由一些空格隔开
// s 表示一个 有效表达式
// 表达式中的所有整数都是非负整数，且在范围 [0, 231 - 1] 内
// 题目数据保证答案是一个 32-bit 整数
func calculate(s string) int {
	numStack, opStack := list.New(), list.New()
	n := len(s)
	opHash := map[byte]int{
		'+': 1,
		'-': 1,
		'*': 2,
		'/': 2,
	}
	opPriority := 0
	for i := 0; i < n; i++ {
		if s[i] == ' ' {
			continue
		} else if isCalculate(s[i]) {
			if s[i] == '-' {

				if numStack.Len() == 0 {
					numStack.PushBack(0)
				}
				if i > 0 && s[i-1] == '(' {
					numStack.PushBack(0)
				}
			}

			oper := &operation{
				op:       s[i],
				priority: opPriority + opHash[s[i]],
			}
			for opStack.Len() > 0 {
				back := opStack.Back()
				tmpOper := back.Value.(*operation)
				if tmpOper.priority < oper.priority {
					break
				}
				// 栈中运算符的优先级更高（相同）先计算
				opStack.Remove(back)
				op := tmpOper.op
				numBack := numStack.Back()
				numStack.Remove(numBack)
				b := numBack.Value.(int)
				numBack = numStack.Back()
				numStack.Remove(numBack)
				a := numBack.Value.(int)
				num := calculateNum(a, b, op)
				numStack.PushBack(num)
			}

			opStack.PushBack(oper)

		} else if s[i] == '(' {
			opPriority += 10
		} else if s[i] == ')' {
			opPriority -= 10
		} else if isNum(s[i]) {
			num := int(s[i] - '0')

			for i+1 < n && isNum(s[i+1]) {
				i++
				num = num*10 + int(s[i]-'0')
			}
			numStack.PushBack(num)
		}
	}
	for opStack.Len() > 0 && numStack.Len() >= 2 {
		back := opStack.Back()
		opStack.Remove(back)
		tmpOper := back.Value.(*operation)
		op := tmpOper.op

		numBack := numStack.Back()
		numStack.Remove(numBack)
		b := numBack.Value.(int)

		numBack = numStack.Back()
		numStack.Remove(numBack)
		a := numBack.Value.(int)
		num := calculateNum(a, b, op)
		numStack.PushBack(num)
		back = opStack.Back()
	}

	numBack := numStack.Back()
	numStack.Remove(numBack)
	result := numBack.Value.(int)
	return result
}

func isMulOp(c byte) bool {
	return c == '*' || c == '/'
}

func isNum(c byte) bool {
	return c >= '0' && c <= '9'
}

// 241. 为运算表达式设计优先级
// 给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。
//
// 示例 1:
// 输入: "2-1-1"
// 输出: [0, 2]
// 解释:
// ((2-1)-1) = 0
// (2-(1-1)) = 2
//
// 示例 2:
// 输入: "2*3-4*5"
// 输出: [-34, -14, -10, -10, 10]
// 解释:
// (2*(3-(4*5))) = -34
// ((2*3)-(4*5)) = -14
// ((2*(3-4))*5) = -10
// (2*((3-4)*5)) = -10
// (((2*3)-4)*5) = 10
func diffWaysToCompute(expression string) []int {
	result := make([]int, 0)
	n := len(expression)
	if isDigit(expression) {
		num, _ := strconv.Atoi(expression)
		return []int{num}
	}

	for i := 1; i < n-1; i++ {
		c := expression[i]
		if !isCalculate(c) {
			continue
		}
		left, right := diffWaysToCompute(expression[:i]), diffWaysToCompute(expression[i+1:])
		for _, leftNum := range left {
			for _, rightNum := range right {
				result = append(result, calculateNum(leftNum, rightNum, c))
			}
		}
	}

	return result
}

// 判断是否为全数字
func isDigit(expression string) bool {
	_, err := strconv.Atoi(expression)
	if err != nil {
		return false
	}
	return true
}

// 556. 下一个更大元素 III
// 给你一个正整数 n ，请你找出符合条件的最小整数，其由重新排列 n 中存在的每位数字组成，并且其值大于 n 。如果不存在这样的正整数，则返回 -1 。
//
// 注意 ，返回的整数应当是一个 32 位整数 ，如果存在满足题意的答案，但不是 32 位整数 ，同样返回 -1 。
//
// 示例 1：
// 输入：n = 12 输出：21
//
// 示例 2：
// 输入：n = 21 输出：-1
//
// 提示：
// 1 <= n <= 231 - 1
func nextGreaterElement(n int) int {
	nums := [32]int{}
	for i := 0; i < 32; i++ {
		nums[i] = -1
	}
	index := 31
	for n > 0 {
		nums[index] = n % 10
		n /= 10
		index--
	}
	// 1 从后往前,找到第一个递增（从后往前递增,从前往后递减）
	// 2 然后 获取前一位数字a, 如果前一位是0或不存在,返回-1
	// 3 在 后面的数字中找到比a大的最小值, 放到a的位置
	// 4 对后面的数字排序,(逆序即可)
	index = -1
	for i := 30; i >= 0; i-- {
		if nums[i] != -1 && nums[i] < nums[i+1] {
			index = i
			break
		}
	}
	if index == -1 {
		return -1
	}
	left, right := index+1, len(nums)-1
	for left < right {
		nums[left], nums[right] = nums[right], nums[left]
		left++
		right--
	}
	for i := index + 1; i < len(nums); i++ {
		if nums[i] > nums[index] {
			nums[index], nums[i] = nums[i], nums[index]
			break
		}
	}
	maxVal := math.MaxInt32 / 10

	result := 0
	for _, num := range nums {
		if num == -1 {
			continue
		}
		if result > maxVal || (result == maxVal && num > 7) {
			return -1
		}
		result = result*10 + num
	}
	return result

}

// 592. 分数加减运算
// 给定一个表示分数加减运算表达式的字符串，你需要返回一个字符串形式的计算结果。 这个结果应该是不可约分的分数，即最简分数。 如果最终结果是一个整数，例如 2，你需要将它转换成分数形式，其分母为 1。所以在上述例子中, 2 应该被转换为 2/1。
//
// 示例 1:
// 输入:"-1/2+1/2" 输出: "0/1"
//
// 示例 2:
// 输入:"-1/2+1/2+1/3" 输出: "1/3"
//
// 示例 3:
// 输入:"1/3-1/2" 输出: "-1/6"
//
// 示例 4:
// 输入:"5/3+1/3" 输出: "2/1"
//
// 说明:
// 输入和输出字符串只包含 '0' 到 '9' 的数字，以及 '/', '+' 和 '-'。
// 输入和输出分数格式均为 ±分子/分母。如果输入的第一个分数或者输出的分数是正数，则 '+' 会被省略掉。
// 输入只包含合法的最简分数，每个分数的分子与分母的范围是  [1,10]。 如果分母是1，意味着这个分数实际上是一个整数。
// 输入的分数个数范围是 [1,10]。
// 最终结果的分子与分母保证是 32 位整数范围内的有效整数。
func fractionAddition(expression string) string {
	calReg, _ := regexp.Compile("[+-]")

	cals := make([]bool, 0)
	for i := 1; i < len(expression); i++ {
		if expression[i] == '+' {
			cals = append(cals, true)
		} else if expression[i] == '-' {
			cals = append(cals, false)
		}
	}
	var nodeList []*FractionNode
	for _, num := range calReg.Split(expression, -1) {
		if len(num) > 0 {
			facts := strings.Split(num, "/")
			numerator, _ := strconv.Atoi(facts[0])
			denominator, _ := strconv.Atoi(facts[1])
			nodeList = append(nodeList, &FractionNode{numerator, denominator})
		}
	}
	node := nodeList[0]
	if expression[0] == '-' {
		node.numerator *= -1
	}

	for i, c := range cals {
		curNode := nodeList[i+1]
		if !c {
			curNode.numerator *= -1
		}
		node = addFraction(node, curNode)
	}

	if node.numerator == 0 {
		return "0/1"
	}
	var builder strings.Builder
	gcd := getGcd(abs(node.numerator), node.denominator)
	builder.WriteString(strconv.Itoa(node.numerator / gcd))
	builder.WriteString("/")
	builder.WriteString(strconv.Itoa(node.denominator / gcd))
	return builder.String()
}

type FractionNode struct {
	// 分子 分母
	numerator, denominator int
}

func addFraction(node1, node2 *FractionNode) *FractionNode {
	if node1 == nil {
		return node2
	}
	gcd := getGcd(node1.denominator, node2.denominator)
	denominator := node1.denominator * node2.denominator / gcd

	numerator := (node1.numerator*node2.denominator + node2.numerator*node1.denominator) / gcd

	return &FractionNode{numerator, denominator}

}

// 640. 求解方程
// 求解一个给定的方程，将x以字符串"x=#value"的形式返回。该方程仅包含'+'，' - '操作，变量 x 和其对应系数。
//
// 如果方程没有解，请返回“No solution”。
// 如果方程有无限解，则返回“Infinite solutions”。
// 如果方程中只有一个解，要保证返回值 x 是一个整数。
//
// 示例 1：
// 输入: "x+5-3+x=6+x-2"
// 输出: "x=2"
//
// 示例 2:
// 输入: "x=x"
// 输出: "Infinite solutions"
//
// 示例 3:
// 输入: "2x=x"
// 输出: "x=0"
//
// 示例 4:
// 输入: "2x+3x-6x=x+2"
// 输出: "x=-1"
//
// 示例 5:
// 输入: "x=x+2"
// 输出: "No solution"
func solveEquation(equation string) string {
	strs := strings.Split(equation, "=")
	left, right := createEquation(strs[0]), createEquation(strs[1])

	xNum := left.xNum - right.xNum
	value := right.value - left.value
	if xNum == 0 {
		if value == 0 {
			return "Infinite solutions"
		} else {
			return "No solution"
		}
	}
	return "x=" + strconv.Itoa(value/xNum)
}

type equation struct {
	xNum, value int
}

func createEquation(expression string) *equation {
	calReg, _ := regexp.Compile("[+-]")
	flags := make([]bool, 0)
	nums := make([]string, 0)
	flags = append(flags, expression[0] != '-')
	for i := 1; i < len(expression); i++ {
		c := expression[i]
		if c == '+' {
			flags = append(flags, true)
		} else if c == '-' {
			flags = append(flags, false)
		}
	}

	for _, num := range calReg.Split(expression, -1) {
		if len(num) > 0 {
			nums = append(nums, num)
		}
	}
	fmt.Println(expression)
	xNum, value := 0, 0
	for i := 0; i < len(nums); i++ {
		flag, num := flags[i], nums[i]
		n := len(num)
		if num[n-1] == 'x' {

			val := 1
			if n > 1 {
				val, _ = strconv.Atoi(num[:n-1])
			}
			if !flag {
				val *= -1
			}
			fmt.Println("x = ", val)
			xNum += val
		} else {
			val, _ := strconv.Atoi(num)
			if !flag {
				val *= -1
			}
			value += val
		}

	}
	return &equation{
		value: value, xNum: xNum,
	}
}

// 679. 24 点游戏
// 给定一个长度为4的整数数组 cards 。你有 4 张卡片，每张卡片上都包含一个范围在 [1,9] 的数字。您应该使用运算符 ['+', '-', '*', '/'] 和括号 '(' 和 ')' 将这些卡片上的数字排列成数学表达式，以获得值24。
//
// 你须遵守以下规则:
//
// 除法运算符 '/' 表示实数除法，而不是整数除法。
// 例如， 4 /(1 - 2 / 3)= 4 /(1 / 3)= 12 。
// 每个运算都在两个数字之间。特别是，不能使用 “-” 作为一元运算符。
// 例如，如果 cards =[1,1,1,1] ，则表达式 “-1 -1 -1 -1” 是 不允许 的。
// 你不能把数字串在一起
// 例如，如果 cards =[1,2,1,2] ，则表达式 “12 + 12” 无效。
// 如果可以得到这样的表达式，其计算结果为 24 ，则返回 true ，否则返回 false 。
//
// 示例 1:
// 输入: cards = [4, 1, 8, 7]
// 输出: true
// 解释: (8-4) * (7-1) = 24
//
// 示例 2:
// 输入: cards = [1, 2, 1, 2]
// 输出: false
//
// 提示:
// cards.length == 4
// 1 <= cards[i] <= 9
func judgePoint24(cards []int) bool {

	var cal func(nums []float64) bool

	isAnswer := func(num float64) bool {
		return math.Abs(num-24.0) < 1e-6
	}

	cal = func(nums []float64) bool {
		n := len(nums)
		if n == 0 {
			return false
		}
		if n == 1 {
			return isAnswer(float64(nums[0]))
		}
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				nums2 := make([]float64, 0)
				for k := 0; k < n; k++ {
					if k == i || k == j {
						continue
					}
					nums2 = append(nums2, float64(nums[k]))
				}
				// + - * /
				for m := 0; m < 6; m++ {
					switch m {
					case 0:
						{
							nums2 = append(nums2, float64(nums[i]+nums[j]))
						}
					case 1:
						{
							nums2 = append(nums2, float64(nums[i]*nums[j]))
						}
					case 2:
						{
							nums2 = append(nums2, float64(nums[i]-nums[j]))
						}
					case 3:
						{
							nums2 = append(nums2, float64(nums[j]-nums[i]))
						}
					case 4:
						{
							if nums[j] == 0 {
								continue
							}
							nums2 = append(nums2, float64(nums[i]/nums[j]))
						}
					case 5:
						{
							if nums[i] == 0 {
								continue
							}
							nums2 = append(nums2, float64(nums[j]/nums[i]))
						}

					}
					if cal(nums2) {
						return true
					}
					nums2 = nums2[:len(nums2)-1]
				}

			}
		}
		return false
	}
	nums := make([]float64, len(cards))
	for i, num := range cards {
		nums[i] = float64(num)
	}
	return cal(nums)
}
