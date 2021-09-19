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
