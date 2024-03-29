package greedy

import (
	"container/list"
	"fmt"
	"math"
	"sort"
	"strconv"
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
//
//	随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
//
// 示例 2: 输入: [1,2,3,4,5] 输出: 4
// 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
//
//	注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
//	因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
//
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
//
//	第三个孩子只得到 1 颗糖果，这已满足上述两个条件。
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
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
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

// 517. 超级洗衣机
// 假设有 n 台超级洗衣机放在同一排上。开始的时候，每台洗衣机内可能有一定量的衣服，也可能是空的。
// 在每一步操作中，你可以选择任意 m (1 <= m <= n) 台洗衣机，与此同时将每台洗衣机的一件衣服送到相邻的一台洗衣机。
// 给定一个整数数组 machines 代表从左至右每台洗衣机中的衣物数量，请给出能让所有洗衣机中剩下的衣物的数量相等的 最少的操作步数 。如果不能使每台洗衣机中衣物的数量相等，则返回 -1 。
//
// 示例 1：
// 输入：machines = [1,0,5] 输出：3
// 解释：
// 第一步:    1     0 <-- 5    =>    1     1     4
// 第二步:    1 <-- 1 <-- 4    =>    2     1     3
// 第三步:    2     1 <-- 3    =>    2     2     2
//
// 示例 2：
// 输入：machines = [0,3,0] 输出：2
// 解释：
// 第一步:    0 <-- 3     0    =>    1     2     0
// 第二步:    1     2 --> 0    =>    1     1     1
//
// 示例 3：
// 输入：machines = [0,2,0] 输出：-1
// 解释：
// 不可能让所有三个洗衣机同时剩下相同数量的衣物。
//
// 提示：
// n == machines.length
// 1 <= n <= 104
// 0 <= machines[i] <= 105
func findMinMoves(machines []int) int {
	n := len(machines)
	sum := 0
	for _, num := range machines {
		sum += num
	}
	if sum%n != 0 {
		return -1
	}
	avg := sum / n

	for i := 0; i < n; i++ {
		machines[i] -= avg
	}
	result := 0
	// 移动次数 最大移动次数
	move, maxMove := 0, 0
	for _, num := range machines {
		move += num
		// 最大移动次数
		maxMove = max(maxMove, abs(move))

		result = max(result, max(maxMove, num))
	}

	return result
}

// 1005. K 次取反后最大化的数组和
// 给你一个整数数组 nums 和一个整数 k ，按以下方法修改该数组：
//
// 选择某个下标 i 并将 nums[i] 替换为 -nums[i] 。
// 重复这个过程恰好 k 次。可以多次选择同一个下标 i 。
//
// 以这种方式修改数组后，返回数组 可能的最大和 。
//
// 示例 1：
// 输入：nums = [4,2,3], k = 1
// 输出：5
// 解释：选择下标 1 ，nums 变为 [4,-2,3] 。
//
// 示例 2：
// 输入：nums = [3,-1,0,2], k = 3
// 输出：6
// 解释：选择下标 (1, 2, 2) ，nums 变为 [3,1,0,2] 。
//
// 示例 3：
// 输入：nums = [2,-3,-1,5,-4], k = 2
// 输出：13
// 解释：选择下标 (1, 4) ，nums 变为 [2,3,-1,5,4] 。
//
// 提示：
// 1 <= nums.length <= 104
// -100 <= nums[i] <= 100
// 1 <= k <= 104
func largestSumAfterKNegations(nums []int, k int) int {
	sort.Ints(nums)
	n := len(nums)

	for i, num := range nums {
		if k == 0 {
			break
		}
		if num >= 0 {
			break
		}
		nums[i] = -num
		k--
	}
	sort.Ints(nums)
	result := 0
	for i := 1; i < n; i++ {
		result += nums[i]
	}
	if k&1 == 0 {
		result += nums[0]
	} else {
		result -= nums[0]
	}
	return result
}

// 2029. 石子游戏 IX
// Alice 和 Bob 再次设计了一款新的石子游戏。现有一行 n 个石子，每个石子都有一个关联的数字表示它的价值。给你一个整数数组 stones ，其中 stones[i] 是第 i 个石子的价值。
//
// Alice 和 Bob 轮流进行自己的回合，Alice 先手。每一回合，玩家需要从 stones 中移除任一石子。
//
// 如果玩家移除石子后，导致 所有已移除石子 的价值 总和 可以被 3 整除，那么该玩家就 输掉游戏 。
// 如果不满足上一条，且移除后没有任何剩余的石子，那么 Bob 将会直接获胜（即便是在 Alice 的回合）。
// 假设两位玩家均采用 最佳 决策。如果 Alice 获胜，返回 true ；如果 Bob 获胜，返回 false 。
//
// 示例 1：
// 输入：stones = [2,1]
// 输出：true
// 解释：游戏进行如下：
// - 回合 1：Alice 可以移除任意一个石子。
// - 回合 2：Bob 移除剩下的石子。
// 已移除的石子的值总和为 1 + 2 = 3 且可以被 3 整除。因此，Bob 输，Alice 获胜。
//
// 示例 2：
// 输入：stones = [2]
// 输出：false
// 解释：Alice 会移除唯一一个石子，已移除石子的值总和为 2 。
// 由于所有石子都已移除，且值总和无法被 3 整除，Bob 获胜。
//
// 示例 3：
// 输入：stones = [5,1,2,4,3]
// 输出：false
// 解释：Bob 总会获胜。其中一种可能的游戏进行方式如下：
// - 回合 1：Alice 可以移除值为 1 的第 2 个石子。已移除石子值总和为 1 。
// - 回合 2：Bob 可以移除值为 3 的第 5 个石子。已移除石子值总和为 = 1 + 3 = 4 。
// - 回合 3：Alices 可以移除值为 4 的第 4 个石子。已移除石子值总和为 = 1 + 3 + 4 = 8 。
// - 回合 4：Bob 可以移除值为 2 的第 3 个石子。已移除石子值总和为 = 1 + 3 + 4 + 2 = 10.
// - 回合 5：Alice 可以移除值为 5 的第 1 个石子。已移除石子值总和为 = 1 + 3 + 4 + 2 + 5 = 15.
// Alice 输掉游戏，因为已移除石子值总和（15）可以被 3 整除，Bob 获胜。
//
// 提示：
// 1 <= stones.length <= 105
// 1 <= stones[i] <= 104
func stoneGameIX(stones []int) bool {
	count0, count1, count2 := 0, 0, 0
	for _, stone := range stones {
		stone %= 3
		if stone == 0 {
			count0++
		} else if stone == 1 {
			count1++
		} else {
			count2++
		}
	}
	if count0%2 == 0 {
		return count1 >= 1 && count2 >= 1
	}
	return count1-count2 > 2 || count2-count1 > 2
}

// 678. 有效的括号字符串
// 给定一个只包含三种字符的字符串：（ ，） 和 *，写一个函数来检验这个字符串是否为有效字符串。有效字符串具有如下规则：
//
// 任何左括号 ( 必须有相应的右括号 )。
// 任何右括号 ) 必须有相应的左括号 ( 。
// 左括号 ( 必须在对应的右括号之前 )。
// * 可以被视为单个右括号 ) ，或单个左括号 ( ，或一个空字符串。
// 一个空字符串也被视为有效字符串。
//
// 示例 1:
// 输入: "()" 输出: True
//
// 示例 2:
// 输入: "(*)" 输出: True
//
// 示例 3:
// 输入: "(*))" 输出: True
//
// 注意:
// 字符串大小将在 [1，100] 范围内。
func checkValidString(s string) bool {
	left1, left2 := 0, 0
	for _, c := range s {
		switch c {
		case '(':
			{
				left1++
				left2++
			}
		case ')':
			{
				if left1 > 0 {
					left1--
				}
				if left2 == 0 {
					return false
				}
				left2--
			}
		case '*':
			{
				if left1 > 0 {
					// 右括号
					left1--
				}
				// 左括号
				left2++
			}
		}

	}
	return left1 == 0
}

// 738. 单调递增的数字
// 当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。
//
// 给定一个整数 n ，返回 小于或等于 n 的最大数字，且数字呈 单调递增 。
//
// 示例 1:
// 输入: n = 10 输出: 9
//
// 示例 2:
// 输入: n = 1234 输出: 1234
//
// 示例 3:
// 输入: n = 332 输出: 299
//
// 提示:
// 0 <= n <= 109
func monotoneIncreasingDigits(n int) int {
	s := strconv.Itoa(n)
	bytes := []byte(s)
	l := len(bytes)
	i := 1
	// 原来的数字 递增
	for i < l && bytes[i-1] <= bytes[i] {
		i++
	}
	// 把递减开始的地方的前一位 -1
	for 0 < i && i < l && bytes[i-1] > bytes[i] {
		i--
		bytes[i]--
	}
	// 补9
	for j := i + 1; j < l; j++ {
		bytes[j] = '9'
	}
	num, _ := strconv.Atoi(string(bytes))
	return num
}

// 1282. 用户分组
// 有 n 个人被分成数量未知的组。每个人都被标记为一个从 0 到 n - 1 的唯一ID 。
//
// 给定一个整数数组 groupSizes ，其中 groupSizes[i] 是第 i 个人所在的组的大小。例如，如果 groupSizes[1] = 3 ，则第 1 个人必须位于大小为 3 的组中。
//
// 返回一个组列表，使每个人 i 都在一个大小为 groupSizes[i] 的组中。
//
// 每个人应该 恰好只 出现在 一个组 中，并且每个人必须在一个组中。如果有多个答案，返回其中 任何 一个。可以 保证 给定输入 至少有一个 有效的解。
//
// 示例 1：
// 输入：groupSizes = [3,3,3,3,3,1,3]
// 输出：[[5],[0,1,2],[3,4,6]]
// 解释：
// 第一组是 [5]，大小为 1，groupSizes[5] = 1。
// 第二组是 [0,1,2]，大小为 3，groupSizes[0] = groupSizes[1] = groupSizes[2] = 3。
// 第三组是 [3,4,6]，大小为 3，groupSizes[3] = groupSizes[4] = groupSizes[6] = 3。
// 其他可能的解决方案有 [[2,1,6],[5],[0,4,3]] 和 [[5],[0,6,2],[4,3,1]]。
//
// 示例 2：
// 输入：groupSizes = [2,1,3,3,3,2]
// 输出：[[1],[0,5],[2,3,4]]
//
// 提示：
// groupSizes.length == n
// 1 <= n <= 500
// 1 <= groupSizes[i] <= n
func groupThePeople(groupSizes []int) [][]int {
	result := make([][]int, 0)
	groupMap := make(map[int][]int)
	for i, size := range groupSizes {
		groupMap[size] = append(groupMap[size], i)
		if len(groupMap[size]) == size {
			result = append(result, groupMap[size])
			delete(groupMap, size)
		}

	}

	return result
}

// 763. 划分字母区间
// 字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
//
// 示例：
// 输入：S = "ababcbacadefegdehijhklij"
// 输出：[9,7,8]
// 解释：
// 划分结果为 "ababcbaca", "defegde", "hijhklij"。
// 每个字母最多出现在一个片段中。
// 像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
//
// 提示：
// S的长度在[1, 500]之间。
// S只包含小写字母 'a' 到 'z' 。
func partitionLabels(s string) []int {
	lastIndex := make([]int, 26)
	for i, c := range s {
		lastIndex[c-'a'] = i
	}
	result := make([]int, 0)
	last, start := 0, -1
	for i, c := range s {
		last = max(lastIndex[c-'a'], last)
		if last == i {
			result = append(result, i-start)
			start = i
		}
	}
	return result
}

// 757. 设置交集大小至少为2
// 一个整数区间 [a, b]  ( a < b ) 代表着从 a 到 b 的所有连续整数，包括 a 和 b。
//
// 给你一组整数区间intervals，请找到一个最小的集合 S，使得 S 里的元素与区间intervals中的每一个整数区间都至少有2个元素相交。
//
// 输出这个最小集合S的大小。
//
// 示例 1:
// 输入: intervals = [[1, 3], [1, 4], [2, 5], [3, 5]]
// 输出: 3
// 解释:
// 考虑集合 S = {2, 3, 4}. S与intervals中的四个区间都有至少2个相交的元素。
// 且这是S最小的情况，故我们输出3。
//
// 示例 2:
// 输入: intervals = [[1, 2], [2, 3], [2, 4], [4, 5]]
// 输出: 5
// 解释:
// 最小的集合S = {1, 2, 3, 4, 5}.
//
// 注意:
// intervals 的长度范围为[1, 3000]。
// intervals[i] 长度为 2，分别代表左、右边界。
// intervals[i][j] 的值是 [0, 10^8]范围内的整数。
func intersectionSizeTwo(intervals [][]int) int {

	// [a,b] a降序 b 升序
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] == intervals[j][0] {
			return intervals[i][1] <= intervals[j][1]
		}
		return intervals[i][0] > intervals[j][0]
	})
	fmt.Println(intervals)
	result := 0
	// 假设前一个区间为[a1,b1]，与集合的交集是[left,right]，保证left < right,接下来要处理的区间是[a2,b2]。
	// 有以下几种情况：
	// 1、a2 <= left < right <= b2,left、right都在[a2,b2]内部，不需要更新。
	// 2、a2 <= left <= b2 < right,left在[a2,b2]内部，更新right。
	// 3、left < a2 <= right <= b2,,right在[a2,b2]内部，更新left。
	// 4、left、right不在[a2,b2]内部，同时更新left、right。
	// 关键点：更新的时候需要保证left < right

	left, right := math.MaxInt32, math.MaxInt32
	for _, interval := range intervals {
		if interval[0] <= left && left <= interval[1] && interval[0] <= right && right <= interval[1] {
			continue
		} else if interval[0] <= left && left <= interval[1] {
			result++
			if left == interval[0] {
				right = interval[0] + 1
			} else {
				right = left
				left = interval[0]
			}
		} else if interval[0] <= right && right <= interval[1] {
			result++
			if right == interval[0] {
				left = interval[0]
				right = interval[0] + 1
			} else {
				left = interval[0]
			}
		} else {
			result += 2
			left = interval[0]
			right = interval[0] + 1
		}

	}

	return result
}

// 921. 使括号有效的最少添加
// 只有满足下面几点之一，括号字符串才是有效的：
//
// 它是一个空字符串，或者
// 它可以被写成 AB （A 与 B 连接）, 其中 A 和 B 都是有效字符串，或者
// 它可以被写作 (A)，其中 A 是有效字符串。
// 给定一个括号字符串 s ，移动N次，你就可以在字符串的任何位置插入一个括号。
//
// 例如，如果 s = "()))" ，你可以插入一个开始括号为 "(()))" 或结束括号为 "())))" 。
// 返回 为使结果字符串 s 有效而必须添加的最少括号数。
//
// 示例 1：
// 输入：s = "())"
// 输出：1
//
// 示例 2：
// 输入：s = "((("
// 输出：3
//
// 提示：
// 1 <= s.length <= 1000
// s 只包含 '(' 和 ')' 字符。
func minAddToMakeValid(s string) int {
	leftCount, result := 0, 0
	for _, c := range s {
		if c == '(' {
			leftCount++
		} else if c == ')' {
			if leftCount == 0 {
				result++
			} else {
				leftCount--
			}
		}
	}
	result += leftCount
	return result
}

// 1710. 卡车上的最大单元数
// 请你将一些箱子装在 一辆卡车 上。给你一个二维数组 boxTypes ，其中 boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi] ：
//
// numberOfBoxesi 是类型 i 的箱子的数量。
// numberOfUnitsPerBoxi 是类型 i 每个箱子可以装载的单元数量。
// 整数 truckSize 表示卡车上可以装载 箱子 的 最大数量 。只要箱子数量不超过 truckSize ，你就可以选择任意箱子装到卡车上。
//
// 返回卡车可以装载 单元 的 最大 总数。
//
// 示例 1：
// 输入：boxTypes = [[1,3],[2,2],[3,1]], truckSize = 4
// 输出：8
// 解释：箱子的情况如下：
// - 1 个第一类的箱子，里面含 3 个单元。
// - 2 个第二类的箱子，每个里面含 2 个单元。
// - 3 个第三类的箱子，每个里面含 1 个单元。
// 可以选择第一类和第二类的所有箱子，以及第三类的一个箱子。
// 单元总数 = (1 * 3) + (2 * 2) + (1 * 1) = 8
//
// 示例 2：
// 输入：boxTypes = [[5,10],[2,5],[4,7],[3,9]], truckSize = 10
// 输出：91
//
// 提示：
// 1 <= boxTypes.length <= 1000
// 1 <= numberOfBoxesi, numberOfUnitsPerBoxi <= 1000
// 1 <= truckSize <= 106
func maximumUnits(boxTypes [][]int, truckSize int) int {
	result := 0
	sort.Slice(boxTypes, func(i, j int) bool {
		return boxTypes[i][1] > boxTypes[j][1]
	})
	index, n := 0, len(boxTypes)

	for index < n && truckSize > 0 {
		box := boxTypes[index]
		result += min(box[0], truckSize) * box[1]
		truckSize -= box[0]
		index++
	}
	return result
}

// 2027. 转换字符串的最少操作次数
// 给你一个字符串 s ，由 n 个字符组成，每个字符不是 'X' 就是 'O' 。
//
// 一次 操作 定义为从 s 中选出 三个连续字符 并将选中的每个字符都转换为 'O' 。注意，如果字符已经是 'O' ，只需要保持 不变 。
//
// 返回将 s 中所有字符均转换为 'O' 需要执行的 最少 操作次数。
//
// 示例 1：
// 输入：s = "XXX"
// 输出：1
// 解释：XXX -> OOO
// 一次操作，选中全部 3 个字符，并将它们转换为 'O' 。
//
// 示例 2：
// 输入：s = "XXOX"
// 输出：2
// 解释：XXOX -> OOOX -> OOOO
// 第一次操作，选择前 3 个字符，并将这些字符转换为 'O' 。
// 然后，选中后 3 个字符，并执行转换。最终得到的字符串全由字符 'O' 组成。
//
// 示例 3：
// 输入：s = "OOOO"
// 输出：0
// 解释：s 中不存在需要转换的 'X' 。
//
// 提示：
// 3 <= s.length <= 1000
// s[i] 为 'X' 或 'O'
func minimumMoves(s string) int {
	n := len(s)
	result := 0
	// 找到第一个X
	i := 0
	for i < n {
		for i < n && s[i] == 'O' {
			i++
		}
		b := false
		for j := 0; j < 3; j++ {
			if i >= n {
				break
			}
			if s[i] == 'X' {
				b = true
			}
			i++
		}
		if b {
			result++
		}
	}

	return result

}

// 2591. 将钱分给最多的儿童
// 给你一个整数 money ，表示你总共有的钱数（单位为美元）和另一个整数 children ，表示你要将钱分配给多少个儿童。
// 你需要按照如下规则分配：
// 所有的钱都必须被分配。
// 每个儿童至少获得 1 美元。
// 没有人获得 4 美元。
// 请你按照上述规则分配金钱，并返回 最多 有多少个儿童获得 恰好 8 美元。如果没有任何分配方案，返回 -1 。
//
// 示例 1：
// 输入：money = 20, children = 3
// 输出：1
// 解释：
// 最多获得 8 美元的儿童数为 1 。一种分配方案为：
// - 给第一个儿童分配 8 美元。
// - 给第二个儿童分配 9 美元。
// - 给第三个儿童分配 3 美元。
// 没有分配方案能让获得 8 美元的儿童数超过 1 。
//
// 示例 2：
// 输入：money = 16, children = 2
// 输出：2
// 解释：每个儿童都可以获得 8 美元。
//
// 提示：
// 1 <= money <= 200
// 2 <= children <= 30
func distMoney(money int, children int) int {
	if money < children {
		return -1
	}
	money -= children
	count := min(money/7, children)
	// 剩余的钱
	money -= 7 * count
	children -= count
	if (children == 0 && money > 0) || (children == 1 && money == 3) {
		count--
	}
	return count
}

// 2578. 最小和分割
// 给你一个正整数 num ，请你将它分割成两个非负整数 num1 和 num2 ，满足：
// num1 和 num2 直接连起来，得到 num 各数位的一个排列。
// 换句话说，num1 和 num2 中所有数字出现的次数之和等于 num 中所有数字出现的次数。
// num1 和 num2 可以包含前导 0 。
// 请你返回 num1 和 num2 可以得到的和的 最小 值。
//
// 注意：
// num 保证没有前导 0 。
// num1 和 num2 中数位顺序可以与 num 中数位顺序不同。
//
// 示例 1：
// 输入：num = 4325
// 输出：59
// 解释：我们可以将 4325 分割成 num1 = 24 和 num2 = 35 ，和为 59 ，59 是最小和。
//
// 示例 2：
// 输入：num = 687
// 输出：75
// 解释：我们可以将 687 分割成 num1 = 68 和 num2 = 7 ，和为最优值 75 。
//
// 提示：
// 10 <= num <= 109
func splitNum(num int) int {
	nums := make([]int, 0)
	for num > 0 {
		nums = append(nums, num%10)
		num = num / 10
	}
	sort.Ints(nums)
	n := len(nums)
	num1, num2 := 0, 0
	for i := 0; i < n; i += 2 {
		num1 = num1*10 + nums[i]
	}
	for i := 1; i < n; i += 2 {
		num2 = num2*10 + nums[i]
	}

	return num1 + num2
}

// 1465. 切割后面积最大的蛋糕
// 矩形蛋糕的高度为 h 且宽度为 w，给你两个整数数组 horizontalCuts 和 verticalCuts，其中：
//
// horizontalCuts[i] 是从矩形蛋糕顶部到第  i 个水平切口的距离
// verticalCuts[j] 是从矩形蛋糕的左侧到第 j 个竖直切口的距离
// 请你按数组 horizontalCuts 和 verticalCuts 中提供的水平和竖直位置切割后，请你找出 面积最大 的那份蛋糕，并返回其 面积 。由于答案可能是一个很大的数字，因此需要将结果 对 109 + 7 取余 后返回。
//
// 示例 1：
// 输入：h = 5, w = 4, horizontalCuts = [1,2,4], verticalCuts = [1,3]
// 输出：4
// 解释：上图所示的矩阵蛋糕中，红色线表示水平和竖直方向上的切口。切割蛋糕后，绿色的那份蛋糕面积最大。
//
// 示例 2：
// 输入：h = 5, w = 4, horizontalCuts = [3,1], verticalCuts = [1]
// 输出：6
// 解释：上图所示的矩阵蛋糕中，红色线表示水平和竖直方向上的切口。切割蛋糕后，绿色和黄色的两份蛋糕面积最大。
//
// 示例 3：
// 输入：h = 5, w = 4, horizontalCuts = [3], verticalCuts = [3]
// 输出：9
//
// 提示：
// 2 <= h, w <= 109
// 1 <= horizontalCuts.length <= min(h - 1, 105)
// 1 <= verticalCuts.length <= min(w - 1, 105)
// 1 <= horizontalCuts[i] < h
// 1 <= verticalCuts[i] < w
// 题目数据保证 horizontalCuts 中的所有元素各不相同
// 题目数据保证 verticalCuts 中的所有元素各不相同
func maxArea(h int, w int, horizontalCuts []int, verticalCuts []int) int {
	sort.Ints(horizontalCuts)
	sort.Ints(verticalCuts)
	maxHeight, maxWidth := 0, 0

	for i := 0; i < len(horizontalCuts); i++ {
		var height int
		if i == 0 {
			height = horizontalCuts[i]
		} else {
			height = horizontalCuts[i] - horizontalCuts[i-1]
		}
		maxHeight = max(maxHeight, height)
		if i == len(horizontalCuts)-1 {
			height = h - horizontalCuts[i]
			maxHeight = max(maxHeight, height)
		}
	}
	for i := 0; i < len(verticalCuts); i++ {
		var width int
		if i == 0 {
			width = verticalCuts[i]
		} else {
			width = verticalCuts[i] - verticalCuts[i-1]
		}
		maxWidth = max(maxWidth, width)
		if i == len(verticalCuts)-1 {
			width = w - verticalCuts[i]
			maxWidth = max(maxWidth, width)
		}
	}
	mod := 1000000007
	return int(int64(maxHeight) * int64(maxWidth) % int64(mod))
}
