package greedy

import (
	"container/list"
	"math"
	"sort"
	"strings"
)

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

// 134. 加油站
// 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
// 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
// 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
//
// 说明:
// 如果题目有解，该答案即为唯一答案。
// 输入数组均为非空数组，且长度相同。
// 输入数组中的元素均为非负数。
//
// 示例 1:
// 输入:  gas  = [1,2,3,4,5] cost = [3,4,5,1,2]
// 输出: 3
// 解释:
// 从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
// 开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
// 开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
// 开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
// 开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
// 开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
// 因此，3 可为起始索引。
//
// 示例 2:
// 输入:  gas  = [2,3,4] cost = [3,4,3]
// 输出: -1
// 解释:
// 你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
// 我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
// 开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
// 开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
// 你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
// 因此，无论怎样，你都不可能绕环路行驶一周。
func canCompleteCircuit(gas []int, cost []int) int {

	left, size := 0, len(gas)
	// 最小剩余的油量 最小剩余油量的idx
	minLeft, minIdx := math.MaxInt32, 0

	for i := 0; i < size; i++ {
		left += gas[i] - cost[i]
		if left < minLeft {
			minLeft = left
			minIdx = i
		}
	}

	if left < 0 {
		return -1
	}

	return (minIdx + 1) % size
}

// 135. 分发糖果
// 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
//
// 你需要按照以下要求，帮助老师给这些孩子分发糖果：
// 每个孩子至少分配到 1 个糖果。
// 评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
// 那么这样下来，老师至少需要准备多少颗糖果呢？
//
// 示例 1：
// 输入：[1,0,2] 输出：5
// 解释：你可以分别给这三个孩子分发 2、1、2 颗糖果。
//
// 示例 2：
// 输入：[1,2,2] 输出：4
// 解释：你可以分别给这三个孩子分发 1、2、1 颗糖果。
//     第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
func candy(ratings []int) int {
	// 找谷底 最低的孩子只得1颗糖果
	size := len(ratings)
	leftNums, rightNums := make([]int, size), make([]int, size)
	leftNums[0], rightNums[size-1] = 1, 1
	for i := 1; i < size; i++ {
		if ratings[i] > ratings[i-1] {
			leftNums[i] = leftNums[i-1] + 1
		} else {
			leftNums[i] = 1
		}
	}
	for i := size - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] {
			rightNums[i] = rightNums[i+1] + 1
		} else {
			rightNums[i] = 1
		}
	}
	result := 0
	for i := 0; i < size; i++ {
		result += max(leftNums[i], rightNums[i])
	}
	return result
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

// 330. 按要求补齐数组
// 给定一个已排序的正整数数组 nums，和一个正整数 n 。从 [1, n] 区间内选取任意个数字补充到 nums 中，使得 [1, n] 区间内的任何数字都可以用 nums 中某几个数字的和来表示。请输出满足上述要求的最少需要补充的数字个数。
//
// 示例 1:
// 输入: nums = [1,3], n = 6
// 输出: 1
// 解释:
// 根据 nums 里现有的组合 [1], [3], [1,3]，可以得出 1, 3, 4。
// 现在如果我们将 2 添加到 nums 中， 组合变为: [1], [2], [3], [1,3], [2,3], [1,2,3]。
// 其和可以表示数字 1, 2, 3, 4, 5, 6，能够覆盖 [1, 6] 区间里所有的数。
// 所以我们最少需要添加一个数字。
//
// 示例 2:
// 输入: nums = [1,5,10], n = 20
// 输出: 2
// 解释: 我们需要添加 [2, 4]。
//
// 示例 3:
// 输入: nums = [1,2,2], n = 5
// 输出: 0
func minPatches(nums []int, n int) int {
	count, idx := 0, 0
	num := 1

	for num <= n {
		if idx < len(nums) && nums[idx] <= num {
			num += nums[idx]
			idx++
		} else {
			// 当前 nums[idx] 无法 通过 num 组合
			// 新增 num
			num += num
			count++
		}
	}

	return count
}

// 402. 移掉 K 位数字
// 给你一个以字符串表示的非负整数 num 和一个整数 k ，移除这个数中的 k 位数字，使得剩下的数字最小。请你以字符串形式返回这个最小的数字。
//
// 示例 1 ：
// 输入：num = "1432219", k = 3 输出："1219"
// 解释：移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219 。
//
// 示例 2 ：
// 输入：num = "10200", k = 1
// 输出："200"
// 解释：移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。
//
// 示例 3 ：
// 输入：num = "10", k = 2
// 输出："0"
// 解释：从原数字移除所有的数字，剩余为空就是 0 。
//
// 提示：
// 1 <= k <= num.length <= 105
// num 仅由若干位数字（0 - 9）组成
// 除了 0 本身之外，num 不含任何前导零
func removeKdigits(num string, k int) string {
	n := len(num)
	if n <= k {
		return "0"
	}
	stack := list.New()
	for i := 0; i < n; i++ {
		for stack.Len() > 0 {
			back := stack.Back()
			backNum := back.Value.(byte)
			if num[i] >= backNum || k <= 0 {
				break
			}
			k--
			stack.Remove(back)
		}
		stack.PushBack(num[i])
	}
	for i := 0; i < k; i++ {
		back := stack.Back()
		stack.Remove(back)
	}
	var builder strings.Builder
	leadingZero := true
	for e := stack.Front(); e != nil; e = e.Next() {
		val := e.Value.(byte)
		if leadingZero && val == '0' {
			continue
		}
		leadingZero = false
		builder.WriteByte(val)
	}
	if builder.Len() == 0 {
		return "0"
	}
	return builder.String()
}

// 452. 用最少数量的箭引爆气球
// 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
//
// 一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
//
// 给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
//
// 示例 1：
// 输入：points = [[10,16],[2,8],[1,6],[7,12]]
// 输出：2
// 解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
//
// 示例 2：
// 输入：points = [[1,2],[3,4],[5,6],[7,8]]
// 输出：4
//
// 示例 3：
// 输入：points = [[1,2],[2,3],[3,4],[4,5]]
// 输出：2
//
// 示例 4：
// 输入：points = [[1,2]]
// 输出：1
//
// 示例 5：
// 输入：points = [[2,3],[2,3]]
// 输出：1
//
// 提示：
// 1 <= points.length <= 104
// points[i].length == 2
// -2^31 <= xstart < xend <= 2^31 - 1
func findMinArrowShots(points [][]int) int {

	// end 从小到大 排列
	sort.Slice(points, func(i, j int) bool {
		point1, point2 := points[i], points[j]
		if point1[1] == point2[1] {
			return point1[0] < point2[0]
		}
		return point1[1] < point2[1]
	})
	result := 1
	preEnd := points[0][1]
	// 求弓箭数量
	for _, point := range points {
		start, end := point[0], point[1]
		// 没有交集, 需要额外的弓箭
		if preEnd < start {
			result++
			preEnd = end
		}
	}

	return result
}
