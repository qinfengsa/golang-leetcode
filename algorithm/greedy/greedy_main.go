package greedy

import "sort"

// 贪心算法

// 122. 买卖股票的最佳时机 II
// 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
//
// 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
//
// 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
// 示例 1: 输入: [7,1,5,3,6,4] 输出: 7
// 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
//     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
// 示例 2: 输入: [1,2,3,4,5] 输出: 4
// 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
//     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
//     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
// 示例 3: 输入: [7,6,4,3,1] 输出: 0
// 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
// 提示：
// 1 <= prices.length <= 3 * 10 ^ 4
// 0 <= prices[i] <= 10 ^ 4
func maxProfit2(prices []int) int {
	l := len(prices)

	result := 0
	for i := 1; i < l; i++ {
		if prices[i] > prices[i-1] {
			result += prices[i] - prices[i-1]
		}
	}

	return result
}

// 455. 分发饼干
// 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
//
// 对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
//
// 示例 1:
//
// 输入: g = [1,2,3], s = [1,1] 输出: 1
// 解释:
// 你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
// 虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
// 所以你应该输出1。
// 示例 2:
//
// 输入: g = [1,2], s = [1,2,3]
// 输出: 2
// 解释:
// 你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
// 你拥有的饼干数量和尺寸都足以让所有孩子满足。
// 所以你应该输出2.
// 提示：
//
// 1 <= g.length <= 3 * 104
// 0 <= s.length <= 3 * 104
// 1 <= g[i], s[j] <= 231 - 1
func findContentChildren(g []int, s []int) int {
	m, n := len(g), len(s)
	if n == 0 {
		return 0
	}
	sort.Ints(g)
	sort.Ints(s)
	i, j, result := 0, 0, 0
	for i < m && j < n {
		if g[i] <= s[j] {
			i++
			j++
			result++
		} else {
			j++
		}
	}

	return result
}

// 860. 柠檬水找零
// 在柠檬水摊上，每一杯柠檬水的售价为 5 美元。
// 顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
// 每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。
//
// 注意，一开始你手头没有任何零钱。
//
// 如果你能给每位顾客正确找零，返回 true ，否则返回 false 。
//
// 示例 1： 输入：[5,5,5,10,20] 输出：true
// 解释：
// 前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
// 第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
// 第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
// 由于所有客户都得到了正确的找零，所以我们输出 true。
//
// 示例 2：输入：[5,5,10] 输出：true
//
// 示例 3：输入：[10,10] 输出：false
//
// 示例 4：输入：[5,5,10,10,20] 输出：false
// 解释：
// 前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
// 对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
// 对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
// 由于不是每位顾客都得到了正确的找零，所以答案是 false。
//
// 提示：
// 0 <= bills.length <= 10000
// bills[i] 不是 5 就是 10 或是 20
func lemonadeChange(bills []int) bool {
	coins := [2]int{0, 0}
	for _, bill := range bills {
		if bill == 5 {
			coins[0]++
		} else if bill == 10 {
			if coins[0] == 0 {
				return false
			}
			coins[0]--
			coins[1]++
		} else if bill == 20 {
			bill -= 5
			if coins[1] > 0 {
				coins[1]--
				bill -= 10
			}
			for bill > 0 && coins[0] > 0 {
				coins[0]--
				bill -= 5
			}
			if bill > 0 {
				return false
			}
		}

	}
	return true
}
