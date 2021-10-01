package math

import (
	"container/list"
	"strconv"
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
