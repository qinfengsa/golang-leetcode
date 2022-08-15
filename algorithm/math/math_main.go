package math

import (
	"container/list"
	"fmt"
	"log"
	"math"
	"math/bits"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

func reverseTest() {
	x := 0
	result := reverse(x)
	fmt.Println(result)
}

// 7. 整数反转
// 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
//
// 示例 1: 输入: 123 输出: 321
// 示例 2: 输入: -123 输出: -321
// 示例 3: 输入: 120 输出: 21
// 注意: 假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。
func reverse(x int) int {
	MIN := -(1 << 31)
	MAX := (1 << 31) - 1
	num := 0
	for x != 0 {
		num = num*10 + x%10
		if num < MIN || num > MAX {
			return 0
		}
		x /= 10
	}
	return num
}

// 9. 回文数
// 判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
//
// 示例 1: 输入: 121 输出: true
// 示例 2: 输入: -121 输出: false
// 解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
// 示例 3: 输入: 10 输出: false
// 解释: 从右向左读, 为 01 。因此它不是一个回文数。
// 进阶:
//
// 你能不将整数转为字符串来解决这个问题吗？
func isPalindrome(x int) bool {
	if x < 0 || (x > 0 && x%10 == 0) {
		return false
	}
	if x < 10 {
		return true
	}
	num, tmp := 0, x
	for tmp > 0 {
		num = num*10 + tmp%10
		tmp /= 10
	}
	return num == x
}

// 69. x 的平方根
// 实现 int sqrt(int x) 函数。
//
// 计算并返回 x 的平方根，其中 x 是非负整数。
//
// 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
//
// 示例 1: 输入: 4 输出: 2
// 示例 2: 输入: 8 输出: 2
// 说明: 8 的平方根是 2.82842...,
//     由于返回类型是整数，小数部分将被舍去。
func mySqrt(x int) int {
	/*left, right, result := 1, x, 0
	for left <= right {
		mid := (left + right) >> 1
		if mid * mid <= x {
			result = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	return result*/

	// 牛顿迭代法
	// Xn+1 = (Xn)/2 + a/(2*Xn) (a就是x, Xn就是要求的平方根)
	// Xn+1 = (Xn + a/Xn)/2
	// 当Xn*Xn不大于x时，Xn为最接近的平方跟
	r := x
	for r*r > x {
		r = (r + x/r) >> 1
	}
	return r
}

// 136. 只出现一次的数字
// 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
//
// 说明：
//
// 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
//
// 示例 1: 输入: [2,2,1] 输出: 1
// 示例 2: 输入: [4,1,2,1,2] 输出: 4
func singleNumber(nums []int) int {
	result := 0
	for _, num := range nums {
		result ^= num
	}
	return result
}

func convertToTitleTest() {
	n := 10
	result := convertToTitle(n)
	log.Println(result)
}

// 168. Excel表列名称
// 给定一个正整数，返回它在 Excel 表中相对应的列名称。
//
// 例如，
//
//    1 -> A
//    2 -> B
//    3 -> C
//    ...
//    26 -> Z
//    27 -> AA
//    28 -> AB
//    ...
// 示例 1:
// 输入: 1 输出: "A"
// 示例 2:
// 输入: 28 输出: "AB"
// 示例 3:
// 输入: 701 输出: "ZY"
func convertToTitle(n int) string {
	result := ""
	for n > 0 {
		num := n % 26
		if num == 0 {
			num = 26
			n -= 26
		}
		result = fmt.Sprintf("%c", num-1+'A') + result
		n /= 26
	}
	return result
}

// 171. Excel表列序号
// 给定一个Excel表格中的列名称，返回其相应的列序号。
//
// 例如，
//    A -> 1
//    B -> 2
//    C -> 3
//    ...
//    Z -> 26
//    AA -> 27
//    AB -> 28
//    ...
// 示例 1:
//
// 输入: "A" 输出: 1
// 示例 2:
// 输入: "AB" 输出: 28
// 示例 3:
// 输入: "ZY" 输出: 701
func titleToNumber(columnTitle string) int {
	num := 0
	for _, c := range columnTitle {
		v := c - 'A' + 1
		num = num*26 + int(v)
	}

	return num
}

// 172. 阶乘后的零
// 给定一个整数 n，返回 n! 结果尾数中零的数量。
//
// 示例 1:
//
// 输入: 3 输出: 0
// 解释: 3! = 6, 尾数中没有零。
// 示例 2:
//
// 输入: 5 输出: 1
// 解释: 5! = 120, 尾数中有 1 个零.
// 说明: 你算法的时间复杂度应为 O(log n) 。
func trailingZeroes(n int) int {
	if n < 5 {
		return 0
	}
	// 0的个数和5,10 有关
	result := 0
	for n > 0 {
		n /= 5
		result += n
	}
	return result
}

// 190. 颠倒二进制位
// 颠倒给定的 32 位无符号整数的二进制位。
// 示例 1：
//
// 输入: 00000010100101000001111010011100
// 输出: 00111001011110000010100101000000
// 解释: 输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
//     因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
// 示例 2：
//
// 输入：11111111111111111111111111111101
// 输出：10111111111111111111111111111111
// 解释：输入的二进制串 11111111111111111111111111111101 表示无符号整数 4294967293，
//     因此返回 3221225471 其二进制表示形式为 10111111111111111111111111111111 。
// 提示：
//
// 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
// 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 2 中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。
//
// 进阶: 如果多次调用这个函数，你将如何优化你的算法？
func reverseBits(num uint32) uint32 {
	// 前16位 和 后16位 交换
	num = (num >> 16) | (num << 16)
	num = ((num & 0xff00ff00) >> 8) | ((num & 0x00ff00ff) << 8)
	num = ((num & 0xf0f0f0f0) >> 4) | ((num & 0x0f0f0f0f) << 4)
	num = ((num & 0xcccccccc) >> 2) | ((num & 0x33333333) << 2)
	num = ((num & 0xaaaaaaaa) >> 1) | ((num & 0x55555555) << 1)
	return num
}

// 191. 位1的个数
// 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。
// 示例 1：
// 输入：00000000000000000000000000001011 输出：3
// 解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
// 示例 2：
// 输入：00000000000000000000000010000000 输出：1
// 解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
// 示例 3：
// 输入：11111111111111111111111111111101 输出：31
// 解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
// 提示：
// 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
// 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。
// 进阶:
// 如果多次调用这个函数，你将如何优化你的算法？
func hammingWeight(num uint32) int {
	count := 0

	for num > 0 {
		if num&1 == 1 {
			count++
		}
		num >>= 1
	}

	return count
}

// 202. 快乐数
// 编写一个算法来判断一个数 n 是不是快乐数。
//
//「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。
//
// 如果 n 是快乐数就返回 True ；不是，则返回 False 。
// 示例： 输入：19 输出：true
// 解释：
// 12 + 92 = 82
// 82 + 22 = 68
// 62 + 82 = 100
// 12 + 02 + 02 = 1
func isHappy(n int) bool {
	numMap := make(map[int]bool)

	for n != 1 {
		_, ok := numMap[n]
		if ok {
			return false
		}
		numMap[n] = true
		num := 0
		for n > 0 {
			a := n % 10
			n /= 10
			num += a * a
		}
		n = num
	}

	return true
}

// 204. 计数质数
// 统计所有小于非负整数 n 的质数的数量。
// 示例 1： 输入：n = 10 输出：4
// 解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
// 示例 2：输入：n = 0 输出：0
// 示例 3：输入：n = 1 输出：0
// 提示：0 <= n <= 5 * 106
func countPrimes(n int) int {
	nums := make([]bool, n)
	if n <= 1 {
		return 0
	}

	for i := 2; i*i < n; i++ {
		if nums[i] {
			continue
		}
		for j := 2; i*j < n; j++ {
			nums[i*j] = true
		}
	}
	count := 0
	for i := 2; i < n; i++ {
		if !nums[i] {
			count++
		}
	}
	return count
}

// 231. 2的幂
// 给定一个整数，编写一个函数来判断它是否是 2 的幂次方。
//
// 示例 1:
//
// 输入: 1 输出: true
// 解释: 20 = 1
// 示例 2:
//
// 输入: 16 输出: true
// 解释: 24 = 16
// 示例 3:
//
// 输入: 218 输出: false
func isPowerOfTwo(n int) bool {
	if n <= 0 {
		return false
	}
	return n&(n-1) == 0
}

// 263. 丑数
// 编写一个程序判断给定的数是否为丑数。
//
// 丑数就是只包含质因数 2, 3, 5 的正整数。
//
// 示例 1:
//
// 输入: 6 输出: true
// 解释: 6 = 2 × 3
// 示例 2:
//
// 输入: 8 输出: true
// 解释: 8 = 2 × 2 × 2
// 示例 3:
//
// 输入: 14 输出: false
// 解释: 14 不是丑数，因为它包含了另外一个质因数 7。
// 说明：
//
// 1 是丑数。
// 输入不会超过 32 位有符号整数的范围: [−231,  231 − 1]。
func isUgly(num int) bool {
	if num < 1 {
		return false
	}
	for num%2 == 0 {
		num /= 2
	}
	for num%3 == 0 {
		num /= 3
	}
	for num%5 == 0 {
		num /= 5
	}
	return num == 1
}

// 268. 丢失的数字
// 给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
// 进阶：
//
// 你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题?
//
//
// 示例 1：输入：nums = [3,0,1] 输出：2
// 解释：n = 3，因为有 3 个数字，所以所有的数字都在范围 [0,3] 内。2 是丢失的数字，因为它没有出现在 nums 中。
// 示例 2：输入：nums = [0,1] 输出：2
// 解释：n = 2，因为有 2 个数字，所以所有的数字都在范围 [0,2] 内。2 是丢失的数字，因为它没有出现在 nums 中。
// 示例 3：输入：nums = [9,6,4,2,3,5,7,0,1] 输出：8
// 解释：n = 9，因为有 9 个数字，所以所有的数字都在范围 [0,9] 内。8 是丢失的数字，因为它没有出现在 nums 中。
// 示例 4：输入：nums = [0] 输出：1
// 解释：n = 1，因为有 1 个数字，所以所有的数字都在范围 [0,1] 内。1 是丢失的数字，因为它没有出现在 nums 中。
// 提示：
// n == nums.length
// 1 <= n <= 104
// 0 <= nums[i] <= n
// nums 中的所有数字都 独一无二
func missingNumber(nums []int) int {
	result := 0
	for i, num := range nums {
		result ^= i ^ num
	}
	result ^= len(nums)
	return result
}

// 371. 两整数之和
// 不使用运算符 + 和 - ，计算两整数 a 、b 之和。
//
// 示例 1:
//
// 输入: a = 1, b = 2 输出: 3
// 示例 2:
//
// 输入: a = -2, b = 3 输出: 1
func getSum(a int, b int) int {
	for b != 0 {
		tmp := a ^ b
		b = (a & b) << 1
		a = tmp
	}
	return a
}

// 405. 数字转换为十六进制数
// 给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 补码运算 方法。
//
// 注意:
//
// 十六进制中所有字母(a-f)都必须是小写。
// 十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符'0'来表示；对于其他情况，十六进制字符串中的第一个字符将不会是0字符。
// 给定的数确保在32位有符号整数范围内。
// 不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法。
// 示例 1：
// 输入: 26 输出: "1a"
//
// 示例 2：
// 输入: -1 输出: "ffffffff"
func toHex(num int) string {
	if num < 0 {
		num += 4294967296
	}
	if num == 0 {
		return "0"
	}
	hash := [16]string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"}
	result := ""
	for num > 0 {
		val := num & 15
		result = hash[val] + result
		num >>= 4
	}

	return result
}

// 441. 排列硬币
// 你总共有 n 枚硬币，你需要将它们摆成一个阶梯形状，第 k 行就必须正好有 k 枚硬币。
//
// 给定一个数字 n，找出可形成完整阶梯行的总行数。
//
// n 是一个非负整数，并且在32位有符号整型的范围内。
//
// 示例 1:
// n = 5
// 硬币可排列成以下几行:
// ¤
// ¤ ¤
// ¤ ¤
//
// 因为第三行不完整，所以返回2.
// 示例 2:
// n = 8
// 硬币可排列成以下几行:
// ¤
// ¤ ¤
// ¤ ¤ ¤
// ¤ ¤
//
// 因为第四行不完整，所以返回3.
func arrangeCoins(n int) int {
	low, high := 1, n
	for low < high {
		mid := (low + high) >> 1
		sum := mid * (mid + 1) >> 1
		if sum > n {
			high = mid
		} else { //判断是最后一个元素
			if sum == n || (1+mid+1)*(mid+1)>>1 > n {
				return mid
			}
			low = mid + 1
		}

	}

	return low
}

// 461. 汉明距离
// 两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
//
// 给出两个整数 x 和 y，计算它们之间的汉明距离。
//
// 注意： 0 ≤ x, y < 231.
//
// 示例:
//
// 输入: x = 1, y = 4 输出: 2
//
// 解释:
// 1   (0 0 0 1)
// 4   (0 1 0 0)
//        ↑   ↑
//
// 上面的箭头指出了对应二进制位不同的位置。
func hammingDistance(x int, y int) int {
	num := x ^ y
	/*count := 0
	for num != 0 {
		if num&1 == 1 {
			count++
		}
		num >>= 1
	}*/

	return bits.OnesCount(uint(num))
}

// 476. 数字的补数
// 给定一个正整数，输出它的补数。补数是对该数的二进制表示取反。
//
// 示例 1:
// 输入: 5 输出: 2
// 解释: 5 的二进制表示为 101（没有前导零位），其补数为 010。所以你需要输出 2 。
//
// 示例 2:
// 输入: 1 输出: 0
// 解释: 1 的二进制表示为 1（没有前导零位），其补数为 0。所以你需要输出 0 。
//
// 注意:
// 给定的整数保证在 32 位带符号整数的范围内。
// 你可以假定二进制数不包含前导零位。
// 本题与 1009 https://leetcode-cn.com/problems/complement-of-base-10-integer/ 相同
func findComplement(num int) int {
	count, tmp := 0, num
	for tmp != 0 {
		count++
		tmp >>= 1
	}
	return num ^ ((1 << count) - 1)
}

// 492. 构造矩形
// 作为一位web开发者， 懂得怎样去规划一个页面的尺寸是很重要的。 现给定一个具体的矩形页面面积，你的任务是设计一个长度为 L 和宽度为 W 且满足以下要求的矩形的页面。要求：
//
// 1. 你设计的矩形页面必须等于给定的目标面积。
// 2. 宽度 W 不应大于长度 L，换言之，要求 L >= W 。
// 3. 长度 L 和宽度 W 之间的差距应当尽可能小。
// 你需要按顺序输出你设计的页面的长度 L 和宽度 W。
//
// 示例：
// 输入: 4 输出: [2, 2]
// 解释: 目标面积是 4， 所有可能的构造方案有 [1,4], [2,2], [4,1]。
// 但是根据要求2，[1,4] 不符合要求; 根据要求3，[2,2] 比 [4,1] 更能符合要求. 所以输出长度 L 为 2， 宽度 W 为 2。
// 说明:
//
// 给定的面积不大于 10,000,000 且为正整数。
// 你设计的页面的长度和宽度必须都是正整数。
func constructRectangle(area int) []int {
	w := int(math.Sqrt(float64(area)))
	result := make([]int, 2)
	for area%w != 0 {
		w--
	}
	result[0] = area / w
	result[1] = w

	return result
}

// 504. 七进制数
// 给定一个整数，将其转化为7进制，并以字符串形式输出。
//
// 示例 1:
//
// 输入: 100
// 输出: "202"
// 示例 2:
//
// 输入: -7
// 输出: "-10"
// 注意: 输入范围是 [-1e7, 1e7] 。
func convertToBase7(num int) string {
	if num == 0 {
		return "0"
	}
	result := ""
	minus := num < 0
	if minus {
		num = -num
	}
	for num > 0 {
		result = strconv.Itoa(num%7) + result
		num /= 7
	}
	if minus {
		result = "-" + result
	}
	return result
}

// 507. 完美数
// 对于一个 正整数，如果它和除了它自身以外的所有 正因子 之和相等，我们称它为 「完美数」。
//
// 给定一个 整数 n， 如果是完美数，返回 true，否则返回 false
//
// 示例 1：
// 输入：28 输出：True
// 解释：28 = 1 + 2 + 4 + 7 + 14
// 1, 2, 4, 7, 和 14 是 28 的所有正因子。
//
// 示例 2：
// 输入：num = 6 输出：true
//
// 示例 3：
// 输入：num = 496 输出：true
//
// 示例 4：
// 输入：num = 8128 输出：true
//
// 示例 5：
// 输入：num = 2 输出：false
//
// 提示： 1 <= num <= 108
func checkPerfectNumber(num int) bool {
	if num == 1 {
		return false
	}
	sum := 1
	for i := 2; i*i <= num; i++ {
		if num%i != 0 {
			continue
		}
		if i*i == num {
			sum += i
		} else {
			sum += i + num/i
		}

	}

	return sum == num
}

// 509. 斐波那契数
// 斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
//
// F(0) = 0,   F(1) = 1
// F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
// 给定 N，计算 F(N)。
//
// 示例 1：
// 输入：2 输出：1
// 解释：F(2) = F(1) + F(0) = 1 + 0 = 1.
//
// 示例 2：
// 输入：3 输出：2
// 解释：F(3) = F(2) + F(1) = 1 + 1 = 2.
//
// 示例 3：
// 输入：4 输出：3
// 解释：F(4) = F(3) + F(2) = 2 + 1 = 3.
//
// 提示： 0 ≤ N ≤ 30
func fib(n int) int {
	if n == 0 {
		return 0
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// 598. 范围求和 II
// 给定一个初始元素全部为 0，大小为 m*n 的矩阵 M 以及在 M 上的一系列更新操作。
//
// 操作用二维数组表示，其中的每个操作用一个含有两个正整数 a 和 b 的数组表示，含义是将所有符合 0 <= i < a 以及 0 <= j < b 的元素 M[i][j] 的值都增加 1。
//
// 在执行给定的一系列操作后，你需要返回矩阵中含有最大整数的元素个数。
//
// 示例 1:
// 输入: m = 3, n = 3 operations = [[2,2],[3,3]] 输出: 4
// 解释:
// 初始状态, M =
// [[0, 0, 0],
//  [0, 0, 0],
//  [0, 0, 0]]
//
// 执行完操作 [2,2] 后, M =
// [[1, 1, 0],
//  [1, 1, 0],
//  [0, 0, 0]]
//
// 执行完操作 [3,3] 后, M =
// [[2, 2, 1],
//  [2, 2, 1],
//  [1, 1, 1]]
//
// M 中最大的整数是 2, 而且 M 中有4个值为2的元素。因此返回 4。
// 注意:
// m 和 n 的范围是 [1,40000]。
// a 的范围是 [1,m]，b 的范围是 [1,n]。
// 操作数目不超过 10000。
func maxCount(m int, n int, ops [][]int) int {
	for _, op := range ops {
		if op[0] < m {
			m = op[0]
		}
		if op[1] < n {
			n = op[1]
		}
	}
	return m * n
}

// 628. 三个数的最大乘积
// 给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。
//
// 示例 1:
//
// 输入: [1,2,3]
// 输出: 6
// 示例 2:
//
// 输入: [1,2,3,4]
// 输出: 24
// 注意:
//
// 给定的整型数组长度范围是[3,104]，数组中所有的元素范围是[-1000, 1000]。
// 输入的数组中任意三个数的乘积不会超出32位有符号整数的范围。
func maximumProduct(nums []int) int {
	/*sort.Ints(nums)
	max = nums[size-1] * nums[size-2] * nums[size-3]
	if nums[1] < 0 {
		num := nums[0] * nums[1] * nums[size-1]
		if num > max {
			max = num
		}
	}*/
	maxVal := 0
	a := -1 << 32
	max1, max2, max3 := a, a, a
	min1, min2 := 0, 0
	for _, num := range nums {
		if num > max1 {
			max1, max2, max3 = num, max1, max2
		} else if num > max2 {
			max2, max3 = num, max2
		} else if num > max3 {
			max3 = num
		}

		if num < min1 {
			min1, min2 = num, min1
		} else if num < min2 {
			min2 = num
		}
	}
	maxVal = max1 * max2 * max3
	if min2 < 0 {
		num := min1 * min2 * max1
		if num > maxVal {
			maxVal = num
		}
	}
	return maxVal
}

// 693. 交替位二进制数
// 给定一个正整数，检查它的二进制表示是否总是 0、1 交替出现：换句话说，就是二进制表示中相邻两位的数字永不相同。
//
// 示例 1：
// 输入：n = 5 输出：true
// 解释：5 的二进制表示是：101
//
// 示例 2：
// 输入：n = 7 输出：false
// 解释：7 的二进制表示是：111.
//
// 示例 3：
// 输入：n = 11 输出：false
// 解释：11 的二进制表示是：1011.
//
// 示例 4：
// 输入：n = 10 输出：true
// 解释：10 的二进制表示是：1010.
//
// 示例 5：
// 输入：n = 3 输出：false
//
// 提示：
// 1 <= n <= 231 - 1
func hasAlternatingBits(n int) bool {
	/*last := n & 1
	n >>= 1
	for n > 0 {
		tmp := n & 1
		if tmp == last {
			return false
		}
		n >>= 1
		last = tmp
	}
	return true*/
	num := n ^ (n >> 1)
	return num&(num+1) == 0
}

// 1720. 解码异或后的数组
// 未知 整数数组 arr 由 n 个非负整数组成。
//
// 经编码后变为长度为 n - 1 的另一个整数数组 encoded ，其中 encoded[i] = arr[i] XOR arr[i + 1] 。例如，arr = [1,0,2,1] 经编码后得到 encoded = [1,2,3] 。
//
// 给你编码后的数组 encoded 和原数组 arr 的第一个元素 first（arr[0]）。
//
// 请解码返回原数组 arr 。可以证明答案存在并且是唯一的。
//
// 示例 1：
// 输入：encoded = [1,2,3], first = 1 输出：[1,0,2,1]
// 解释：若 arr = [1,0,2,1] ，那么 first = 1 且 encoded = [1 XOR 0, 0 XOR 2, 2 XOR 1] = [1,2,3]
//
// 示例 2：
// 输入：encoded = [6,2,7,3], first = 4 输出：[4,2,0,7,4]
//
// 提示：
// 2 <= n <= 104
// encoded.length == n - 1
// 0 <= encoded[i] <= 105
// 0 <= first <= 105
func decode(encoded []int, first int) []int {
	n := len(encoded) + 1
	result := make([]int, n)
	result[0] = first
	for i := 1; i < n; i++ {
		result[i] = result[i-1] ^ encoded[i-1]
	}
	return result
}

// 264. 丑数 II
// 给你一个整数 n ，请你找出并返回第 n 个 丑数 。
//
// 丑数 就是只包含质因数 2、3 和/或 5 的正整数。
//
// 示例 1：
//
// 输入：n = 10 输出：12
// 解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
// 示例 2：
//
// 输入：n = 1 输出：1
// 解释：1 通常被视为丑数。
//
//
// 提示： 1 <= n <= 1690
func nthUglyNumber(n int) int {
	a, b, c := 0, 0, 0
	dp := make([]int, n)
	dp[0] = 1

	for i := 0; i < n-1; i++ {
		num2, num3, num5 := dp[a]*2, dp[b]*3, dp[c]*5

		minNum := min(min(num2, num3), num5)
		dp[i+1] = minNum
		if num2 == minNum {
			a++
		}
		if num3 == minNum {
			b++
		}
		if num5 == minNum {
			c++
		}
	}
	return dp[n-1]
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

// 1486. 数组异或操作
// 给你两个整数，n 和 start 。
// 数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）且 n == nums.length 。
//
// 请返回 nums 中所有元素按位异或（XOR）后得到的结果。
//
// 示例 1：
// 输入：n = 5, start = 0 输出：8
// 解释：数组 nums 为 [0, 2, 4, 6, 8]，其中 (0 ^ 2 ^ 4 ^ 6 ^ 8) = 8 。
//     "^" 为按位异或 XOR 运算符。
//
// 示例 2：
// 输入：n = 4, start = 3 输出：8
// 解释：数组 nums 为 [3, 5, 7, 9]，其中 (3 ^ 5 ^ 7 ^ 9) = 8.
//
// 示例 3：
// 输入：n = 1, start = 7 输出：7
//
// 示例 4：
// 输入：n = 10, start = 5 输出：2
//
// 提示：
// 1 <= n <= 1000
// 0 <= start <= 1000
// n == nums.length
func xorOperation(n int, start int) int {
	result := start
	for i := 1; i < n; i++ {
		start += 2
		result ^= start
	}

	return result
}

// 29. 两数相除
// 给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
// 返回被除数 dividend 除以除数 divisor 得到的商。
// 整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
//
// 示例 1:
// 输入: dividend = 10, divisor = 3 输出: 3
// 解释: 10/3 = truncate(3.33333..) = truncate(3) = 3
//
// 示例 2:
// 输入: dividend = 7, divisor = -3 输出: -2
// 解释: 7/-3 = truncate(-2.33333..) = -2
//
// 提示：
// 被除数和除数均为 32 位有符号整数。
// 除数不为 0。
// 假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231,  231 − 1]。本题中，如果除法结果溢出，则返回 231 − 1。
func divide(dividend int, divisor int) int {
	if dividend == 0 {
		return 0
	}
	flag, result := true, 0
	if dividend == math.MinInt32 && divisor == -1 {
		return math.MaxInt32
	}
	if divisor == -1 {
		return -dividend
	}

	var lDividend, lDivisor = int64(dividend), int64(divisor)
	if lDividend < 0 {
		lDividend = -lDividend
		flag = !flag
	}
	if lDivisor < 0 {
		lDivisor = -lDivisor
		flag = !flag
	}
	count := 0
	for lDividend-lDivisor >= 0 {
		count++
		lDivisor <<= 1
	}
	for count >= 0 {
		if lDividend >= lDivisor {
			result += 1 << count
			lDividend -= lDivisor
		}
		count--
		lDivisor >>= 1
	}

	if flag {
		return result
	}
	return -result
}

// 50. Pow(x, n)
// 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。
//
// 示例 1：
// 输入：x = 2.00000, n = 10 输出：1024.00000
//
// 示例 2：
// 输入：x = 2.10000, n = 3 输出：9.26100
//
// 示例 3：
// 输入：x = 2.00000, n = -2 输出：0.25000
// 解释：2-2 = 1/22 = 1/4 = 0.25
//
// 提示：
// -100.0 < x < 100.0
// -231 <= n <= 231-1
// -104 <= xn <= 104
func myPow(x float64, n int) float64 {

	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	if n == -1 {
		return 1.0 / x
	}
	result := 1.0
	flag := false
	if n < 0 {
		flag = true
		n = -n
	}

	for i := n; i != 0; i >>= 1 {
		if i&1 == 1 {
			result *= x
		}
		x *= x
	}

	if flag {
		return 1.0 / result
	}
	return result
}

// 60. 排列序列
// 给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。
//
// 按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
//
// "123"
// "132"
// "213"
// "231"
// "312"
// "321"
// 给定 n 和 k，返回第 k 个排列。
//
// 示例 1：
// 输入：n = 3, k = 3 输出："213"
//
// 示例 2：
// 输入：n = 4, k = 9 输出："2314"
//
// 示例 3：
// 输入：n = 3, k = 1 输出："123"
//
// 提示：
// 1 <= n <= 9
// 1 <= k <= n!
func getPermutation(n int, k int) string {
	nums := make([]int, n)
	for i := 0; i < n; i++ {
		nums[i] = i + 1
	}

	var getLen = func(num int) int {
		result := 1
		for i := num; i > 1; i-- {
			result *= i
		}
		return result
	}

	var builder strings.Builder

	for i := n - 1; i > 0; i-- {
		idx := 0
		if k > 1 {
			// 剩余 i 个 元素共有 cnt 种 排列
			cnt := getLen(i)
			//  移除 若干 cnt 个排列  idx 表示轮到 第 idx个 元素开头
			idx = (k - 1) / cnt
			k -= idx * cnt
		}
		builder.WriteString(strconv.Itoa(nums[idx]))
		tmp := make([]int, n)
		for j := 0; j < idx; j++ {
			tmp[j] = nums[j]
		}
		for j := idx + 1; j < n; j++ {
			tmp[j-1] = nums[j]
		}
		nums = tmp
	}
	builder.WriteString(strconv.Itoa(nums[0]))
	return builder.String()
}

// 1137. 第 N 个泰波那契数
// 泰波那契序列 Tn 定义如下：
//
// T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2
// 给你整数 n，请返回第 n 个泰波那契数 Tn 的值。
//
// 示例 1：
// 输入：n = 4 输出：4
// 解释：
// T_3 = 0 + 1 + 1 = 2
// T_4 = 1 + 1 + 2 = 4
//
// 示例 2：
// 输入：n = 25 输出：1389537
//
// 提示：
//
// 0 <= n <= 37
// 答案保证是一个 32 位整数，即 answer <= 2^31 - 1。
func tribonacci(n int) int {
	if n == 0 {
		return 0
	}
	if n <= 2 {
		return 1
	}
	a, b, c := 0, 1, 1
	for i := 3; i <= n; i++ {
		c, b, a = a+b+c, c, b
	}
	return c
}

// 292. Nim 游戏
// 你和你的朋友，两个人一起玩 Nim 游戏：
//
// 桌子上有一堆石头。
// 你们轮流进行自己的回合，你作为先手。
// 每一回合，轮到的人拿掉 1 - 3 块石头。
// 拿掉最后一块石头的人就是获胜者。
// 假设你们每一步都是最优解。请编写一个函数，来判断你是否可以在给定石头数量为 n 的情况下赢得游戏。如果可以赢，返回 true；否则，返回 false 。
//
// 示例 1：
// 输入：n = 4
// 输出：false
// 解释：如果堆中有 4 块石头，那么你永远不会赢得比赛；
//     因为无论你拿走 1 块、2 块 还是 3 块石头，最后一块石头总是会被你的朋友拿走。
//
// 示例 2：
// 输入：n = 1 输出：true
//
// 示例 3：
// 输入：n = 2 输出：true
//
// 提示：
// 1 <= n <= 231 - 1
func canWinNim(n int) bool {
	return n%4 != 0
}

// 149. 直线上最多的点数
// 给你一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点。求最多有多少个点在同一条直线上。
//
// 示例 1：
// 输入：points = [[1,1],[2,2],[3,3]] 输出：3
//
// 示例 2：
// 输入：points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]] 输出：4
//
// 提示：
// 1 <= points.length <= 300
// points[i].length == 2
// -104 <= xi, yi <= 104
// points 中的所有点 互不相同
func maxPoints(points [][]int) int {
	size := len(points)
	if size <= 2 {
		return size
	}

	maxCnt := 0
	for i := 0; i < size; i++ {
		same := 1
		for j := i + 1; j < size; j++ {
			count := 0
			if points[i][0] == points[j][0] && points[i][1] == points[j][1] {
				// 重复的点
				same++
				continue
			} else {
				count++
				for k := j + 1; k < size; k++ {
					if inLine(points[i], points[j], points[k]) {
						count++
					}
				}
			}
			maxCnt = max(maxCnt, same+count)
		}
	}

	return maxCnt
}

// 166. 分数到小数
// 给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。
// 如果小数部分为循环小数，则将循环的部分括在括号内。
// 如果存在多个答案，只需返回 任意一个 。
// 对于所有给定的输入，保证 答案字符串的长度小于 104 。
//
// 示例 1：
// 输入：numerator = 1, denominator = 2 输出："0.5"
//
// 示例 2：
// 输入：numerator = 2, denominator = 1 输出："2"
//
// 示例 3：
// 输入：numerator = 2, denominator = 3 输出："0.(6)"
//
// 示例 4：
// 输入：numerator = 4, denominator = 333 输出："0.(012)"
//
// 示例 5：
// 输入：numerator = 1, denominator = 5 输出："0.2"
//
// 提示：
// -231 <= numerator, denominator <= 231 - 1
// denominator != 0
func fractionToDecimal(numerator int, denominator int) string {
	if numerator == 0 {
		return "0"
	}
	if denominator == 1 {
		return fmt.Sprintf("%d", numerator)
	}
	if denominator == -1 {
		if numerator >= 0 {
			return strconv.Itoa(-numerator)
		} else {
			result := strconv.Itoa(numerator)
			return result[1:]
		}
	}
	result := ""
	if numerator*denominator < 0 {
		result += "-"
	}
	numerator, denominator = abs(numerator), abs(denominator)
	// 商
	quotient := numerator / denominator
	remainder := numerator % denominator
	result += strconv.Itoa(quotient)
	if remainder == 0 {
		return result
	}
	result += "."
	hash := make(map[int]int)

	for remainder != 0 {
		hash[remainder] = len(result)
		remainder *= 10
		result += strconv.Itoa(remainder / denominator)
		remainder %= denominator
		if i, ok := hash[remainder]; ok {
			result = result[0:i] + "(" + result[i:] + ")"
			break
		}
	}

	return result
}

func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

// 201. 数字范围按位与
// 给你两个整数 left 和 right ，表示区间 [left, right] ，返回此区间内所有数字 按位与 的结果（包含 left 、right 端点）。
//
// 示例 1：
// 输入：left = 5, right = 7 输出：4
//
// 示例 2：
// 输入：left = 0, right = 0 输出：0
//
// 示例 3：
// 输入：left = 1, right = 2147483647 输出：0
//
// 提示：
// 0 <= left <= right <= 231 - 1
func rangeBitwiseAnd(left int, right int) int {
	count := 0

	// right > left 表示最低位经过&运算一定是0（只要有两个连续的数 最低位一定是0和1）
	for right > left && left > 0 {
		count++
		left >>= 1
		right >>= 1
	}
	return right << count
}

// 223. 矩形面积
// 给你 二维 平面上两个 由直线构成的 矩形，请你计算并返回两个矩形覆盖的总面积。
//
// 每个矩形由其 左下 顶点和 右上 顶点坐标表示：
// 第一个矩形由其左下顶点 (ax1, ay1) 和右上顶点 (ax2, ay2) 定义。
// 第二个矩形由其左下顶点 (bx1, by1) 和右上顶点 (bx2, by2) 定义。
//
// 示例 1：
// Rectangle Area
// 输入：ax1 = -3, ay1 = 0, ax2 = 3, ay2 = 4, bx1 = 0, by1 = -1, bx2 = 9, by2 = 2
// 输出：45
//
// 示例 2：
// 输入：ax1 = -2, ay1 = -2, ax2 = 2, ay2 = 2, bx1 = -2, by1 = -2, bx2 = 2, by2 = 2
// 输出：16
//
// 提示：
// -104 <= ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 <= 104
func computeArea(ax1 int, ay1 int, ax2 int, ay2 int, bx1 int, by1 int, bx2 int, by2 int) int {

	area1, area2, area3 := (ax2-ax1)*(ay2-ay1), (bx2-bx1)*(by2-by1), 0
	// x 错开  左 >= 右
	if ax1 >= bx2 || bx1 >= ax2 {
		// y 错开 下 >= 上
	} else if ay1 >= by2 || by1 >= ay2 {

	} else {
		cx1, cy1, cx2, cy2 := max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
		area3 = (cx2 - cx1) * (cy2 - cy1)
	}

	return area1 + area2 - area3
}

// 233. 数字 1 的个数
// 给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。
//
// 示例 1：
// 输入：n = 13
// 输出：6
//
// 示例 2：
// 输入：n = 0
// 输出：0
//
// 提示：
// 0 <= n <= 109
func countDigitOne(n int) int {
	if n == 0 {
		return 0
	}
	count := 0
	for i := 1; i <= n; i *= 10 {
		// 除数 区间 10 100 1000
		div := i * 10
		// 倒数第 i位 是 1 的 个数
		// 23 = 20 + 3 -> 20 中 个位有两个 1  ->  1, 11
		count += n / div * i
		// 查看余数中 倒数第 i位 是 1 的 个数
		// 23 = 20 + 3 -> 3 中 个位有一个 1  ->  21
		// 余数 小于 i -> 0 个; >= 2*i(20)  -> i 个; (10 ~19) ->
		count += min(max(n%div-i+1, 0), i)
	}

	return count
}

// 258. 各位相加
// 给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。
//
// 示例:
// 输入: 38
// 输出: 2
// 解释: 各位相加的过程为：3 + 8 = 11, 1 + 1 = 2。 由于 2 是一位数，所以返回 2。
// 进阶:
// 你可以不使用循环或者递归，且在 O(1) 时间复杂度内解决这个问题吗？
func addDigits(num int) int {
	result := num % 9
	// 找到规律，多次各位相加的结果相当于原数num对9取模，其中考虑到“0”和“9的倍数”这两种特殊情况即可。
	if result == 0 && num != 0 {
		return 9
	}
	return result
}

// 273. 整数转换英文表示
// 将非负整数 num 转换为其对应的英文表示。
//
// 示例 1：
// 输入：num = 123
// 输出："One Hundred Twenty Three"
//
// 示例 2：
// 输入：num = 12345
// 输出："Twelve Thousand Three Hundred Forty Five"
//
// 示例 3：
// 输入：num = 1234567
// 输出："One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
//
// 示例 4：
// 输入：num = 1234567891
// 输出："One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
//
// 提示：
// 0 <= num <= 231 - 1
func numberToWords(num int) string {
	if num == 0 {
		return "Zero"
	}
	Num019 := []string{
		"",
		"One",
		"Two",
		"Three",
		"Four",
		"Five",
		"Six",
		"Seven",
		"Eight",
		"Nine",
		"Ten",
		"Eleven",
		"Twelve",
		"Thirteen",
		"Fourteen",
		"Fifteen",
		"Sixteen",
		"Seventeen",
		"Eighteen",
		"Nineteen"}
	Num090 := []string{
		"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"}
	GENS := []string{"Billion", "Million", "Thousand", ""}
	factors := []int{1000000000, 1000000, 1000}

	var numToWords func(n int)
	numList := make([]string, 0)
	numToWords = func(n int) {
		for i := 0; i < 3; i++ {
			if n >= factors[i] {
				bigNum := n / factors[i]
				n %= factors[i]
				numToWords(bigNum)
				numList = append(numList, GENS[i])
			}
		}
		if n >= 100 {
			hundredNum := n / 100
			n %= 100
			numToWords(hundredNum)
			numList = append(numList, "Hundred")
		}
		if n >= 20 {
			tenIdx := n / 10
			n %= 10
			numList = append(numList, Num090[tenIdx])
		}
		numList = append(numList, Num019[n])
	}
	numToWords(num)
	var builder strings.Builder
	for _, str := range numList {
		if len(str) == 0 {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(str)
	}

	return builder.String()
}

// 313. 超级丑数
// 超级丑数 是一个正整数，并满足其所有质因数都出现在质数数组 primes 中。
//
// 给你一个整数 n 和一个整数数组 primes ，返回第 n 个 超级丑数 。
// 题目数据保证第 n 个 超级丑数 在 32-bit 带符号整数范围内。
//
// 示例 1：
// 输入：n = 12, primes = [2,7,13,19]
// 输出：32
// 解释：给定长度为 4 的质数数组 primes = [2,7,13,19]，前 12 个超级丑数序列为：[1,2,4,7,8,13,14,16,19,26,28,32] 。
//
// 示例 2：
// 输入：n = 1, primes = [2,3,5]
// 输出：1
// 解释：1 不含质因数，因此它的所有质因数都在质数数组 primes = [2,3,5] 中。
//
// 提示：
// 1 <= n <= 106
// 1 <= primes.length <= 100
// 2 <= primes[i] <= 1000
// 题目数据 保证 primes[i] 是一个质数
// primes 中的所有值都 互不相同 ，且按 递增顺序 排列
func nthSuperUglyNumber(n int, primes []int) int {
	m := len(primes)
	dp, indexs := make([]int, n), make([]int, m)

	for i := 0; i < n; i++ {
		dp[i] = math.MaxInt32
	}
	dp[0] = 1
	for i := 1; i < n; i++ {
		for j := 0; j < m; j++ {
			dp[i] = min(dp[i], primes[j]*dp[indexs[j]])
		}
		for j := 0; j < m; j++ {
			if primes[j]*dp[indexs[j]] == dp[i] {
				indexs[j]++
			}
		}
	}

	// primeIndexs[i] 表示 primes[i] 作为 因子的与 nums[j] 得到的最小值 的 j

	return dp[n-1]
}

// 318. 最大单词长度乘积
// 给定一个字符串数组 words，找到 length(word[i]) * length(word[j]) 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。
//
// 示例 1:
// 输入: ["abcw","baz","foo","bar","xtfn","abcdef"]
// 输出: 16
// 解释: 这两个单词为 "abcw", "xtfn"。
//
// 示例 2:
// 输入: ["a","ab","abc","d","cd","bcd","abcd"]
// 输出: 4
// 解释: 这两个单词为 "ab", "cd"。
//
// 示例 3:
// 输入: ["a","aa","aaa","aaaa"]
// 输出: 0
// 解释: 不存在这样的两个单词。
//
// 提示：
// 2 <= words.length <= 1000
// 1 <= words[i].length <= 1000
// words[i] 仅包含小写字母
func maxProduct(words []string) int {
	result, n := 0, len(words)
	wordBits := make([]int, n)
	for i, word := range words {
		wordBit := 0
		for j := 0; j < len(word); j++ {
			wordBit |= 1 << (word[j] - 'a')
		}
		wordBits[i] = wordBit
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if wordBits[i]&wordBits[j] == 0 {
				result = max(result, len(words[i])*len(words[j]))
			}
		}
	}

	return result
}

// 319. 灯泡开关
// 初始时有 n 个灯泡处于关闭状态。
// 对某个灯泡切换开关意味着：如果灯泡状态为关闭，那该灯泡就会被开启；而灯泡状态为开启，那该灯泡就会被关闭。
//
// 第 1 轮，每个灯泡切换一次开关。即，打开所有的灯泡。
// 第 2 轮，每两个灯泡切换一次开关。 即，每两个灯泡关闭一个。
// 第 3 轮，每三个灯泡切换一次开关。
// 第 i 轮，每 i 个灯泡切换一次开关。 而第 n 轮，你只切换最后一个灯泡的开关。
//
// 找出 n 轮后有多少个亮着的灯泡。
//
// 示例 1：
// 输入：n = 3
// 输出：1
// 解释：
// 初始时, 灯泡状态 [关闭, 关闭, 关闭].
// 第一轮后, 灯泡状态 [开启, 开启, 开启].
// 第二轮后, 灯泡状态 [开启, 关闭, 开启].
// 第三轮后, 灯泡状态 [开启, 关闭, 关闭].
//
// 你应该返回 1，因为只有一个灯泡还亮着。
//
// 示例 2：
// 输入：n = 0 输出：0
//
// 示例 3：
// 输入：n = 1 输出：1
//
// 提示：
// 0 <= n <= 109
func bulbSwitch(n int) int {
	// 除了完全平方数，因数都是成对出现的，这意味着实际起到翻转作用(0->1)的，只有 完全平方数而已。
	return int(math.Sqrt(float64(n)))
}

// 326. 3的幂
// 给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 true ；否则，返回 false 。
//
// 整数 n 是 3 的幂次方需满足：存在整数 x 使得 n == 3x
//
// 示例 1：
// 输入：n = 27
// 输出：true
//
// 示例 2：
// 输入：n = 0
// 输出：false
//
// 示例 3：
// 输入：n = 9
// 输出：true
//
// 示例 4：
// 输入：n = 45
// 输出：false
//
// 提示：
// -231 <= n <= 231 - 1
//
// 进阶：
// 你能不使用循环或者递归来完成本题吗？
func isPowerOfThree(n int) bool {
	// return n > 0 && 1162261467%n == 0
	for n >= 3 {
		if n%3 != 0 {
			return false
		}
		n /= 3
	}
	return n == 1
}

// 335. 路径交叉
// 给定一个含有 n 个正数的数组 x。从点 (0,0) 开始，先向北移动 x[0] 米，然后向西移动 x[1] 米，向南移动 x[2] 米，向东移动 x[3] 米，持续移动。也就是说，每次移动后你的方位会发生逆时针变化。
//
// 编写一个 O(1) 空间复杂度的一趟扫描算法，判断你所经过的路径是否相交。
//
// 示例 1:
// ┌───┐
// │   │
// └───┼──>
//     │
// 输入: [2,1,1,2]
// 输出: true
//
// 示例 2:
// ┌──────┐
// │      │
// │
// │
// └────────────>
// 输入: [1,2,3,4]
// 输出: false
//
// 示例 3:
// ┌───┐
// │   │
// └───┼>
// 输入: [1,1,1,1]
// 输出: true
func isSelfCrossing(distance []int) bool {
	n := len(distance)
	if n < 4 {
		return false
	}
	// 交叉只有三种可能：
	//
	// 第一种 条件为 i>=3 && x[i] >= x[i-2] && x[i-1] <= x[i-3]
	//
	// 第二种 条件为 i> 3 && x[i-1] == x[i-3] && x[i-4] + x[i] >= x[i-2]

	// 第三种 条件为 i> 4 && x[i-3]-x[i-5] <= x[i-1] <= x[i-3] && x[i-2]-x[i-4] <= x[i] <= x[i-2] &&
	// x[i-2] > x[i-4]
	for i := 3; i < n; i++ {
		if distance[i] >= distance[i-2] && distance[i-1] <= distance[i-3] {
			return true
		}
		if i > 3 && distance[i-1] == distance[i-3] && distance[i-4]+distance[i] >= distance[i-2] {
			return true
		}
		if i > 4 && distance[i-3]-distance[i-5] <= distance[i-1] && distance[i-1] <= distance[i-3] &&
			distance[i-2]-distance[i-4] <= distance[i] && distance[i] <= distance[i-2] && distance[i-2] > distance[i-4] {
			return true
		}
	}

	return false
}

// 338. 比特位计数
// 给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。
//
// 示例 1：
// 输入：n = 2
// 输出：[0,1,1]
// 解释：
// 0 --> 0
// 1 --> 1
// 2 --> 10
//
// 示例 2：
// 输入：n = 5
// 输出：[0,1,1,2,1,2]
// 解释：
// 0 --> 0
// 1 --> 1
// 2 --> 10
// 3 --> 11
// 4 --> 100
// 5 --> 101
//
// 提示：
// 0 <= n <= 105
//
// 进阶：
// 很容易就能实现时间复杂度为 O(n log n) 的解决方案，你可以在线性时间复杂度 O(n) 内用一趟扫描解决此问题吗？
// 你能不使用任何内置函数解决此问题吗？（如，C++ 中的 __builtin_popcount ）
func countBits(n int) []int {
	result := make([]int, n+1)

	for i := 1; i <= n; i++ {
		result[i] = result[i>>1] + (i & 1)
	}

	return result
}

// 342. 4的幂
// 给定一个整数，写一个函数来判断它是否是 4 的幂次方。如果是，返回 true ；否则，返回 false 。
//
// 整数 n 是 4 的幂次方需满足：存在整数 x 使得 n == 4x
//
// 示例 1：
// 输入：n = 16 输出：true
//
// 示例 2：
// 输入：n = 5 输出：false
//
// 示例 3：
// 输入：n = 1 输出：true
//
// 提示：
// -231 <= n <= 231 - 1
//
// 进阶：
// 你能不使用循环或者递归来完成本题吗？
func isPowerOfFour(n int) bool {
	// 2的次幂

	return n > 0 && n&(n-1) == 0 && n%3 == 1
}

// 343. 整数拆分
// 给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
//
// 示例 1:
// 输入: 2 输出: 1
// 解释: 2 = 1 + 1, 1 × 1 = 1。
//
// 示例 2:
// 输入: 10 输出: 36
// 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
func integerBreak(n int) int {
	if n <= 3 {
		return n - 1
	}
	if n == 4 {
		return 4
	}
	result := 1
	for n > 3 {
		if n == 4 {
			break
		}
		result *= 3
		n -= 3
	}
	if n > 0 {
		result *= n
	}

	return result
}

// 365. 水壶问题
// 有两个容量分别为 x升 和 y升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？
//
// 如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。
//
// 你允许：
// 装满任意一个水壶
// 清空任意一个水壶
// 从一个水壶向另外一个水壶倒水，直到装满或者倒空
//
// 示例 1: (From the famous "Die Hard" example)
// 输入: x = 3, y = 5, z = 4 输出: True
//
// 示例 2:
// 输入: x = 2, y = 6, z = 5 输出: False
func canMeasureWater(jug1Capacity int, jug2Capacity int, targetCapacity int) bool {
	// 贝祖定理告诉 ax+by=z 有解当且仅当 z 是 x, y 的最大公约数的倍数。
	// 因此我们只需要找到 x, y 的最大公约数并判断 z 是否是它的倍数即可。
	if jug1Capacity+jug2Capacity < targetCapacity {
		return false
	}
	if jug1Capacity == 0 || jug2Capacity == 0 {
		return targetCapacity == 0 || targetCapacity == jug1Capacity+jug2Capacity
	}
	return targetCapacity%getGcd(jug1Capacity, jug2Capacity) == 0
}

func getGcd(num1, num2 int) int {
	num1, num2 = max(num1, num2), min(num1, num2)
	if num2 == 0 {
		return num1
	}
	if num1%num2 != 0 {
		return getGcd(num2, num1%num2)
	}

	return num2
}

// 372. 超级次方
// 你的任务是计算 a^b 对 1337 取模，a 是一个正整数，b 是一个非常大的正整数且会以数组形式给出。
//
// 示例 1：
// 输入：a = 2, b = [3] 输出：8
//
// 示例 2：
// 输入：a = 2, b = [1,0] 输出：1024
//
// 示例 3：
// 输入：a = 1, b = [4,3,3,8,5,2] 输出：1
//
// 示例 4：
// 输入：a = 2147483647, b = [2,0,0] 输出：1198
//
// 提示：
// 1 <= a <= 231 - 1
// 1 <= b.length <= 2000
// 0 <= b[i] <= 9
// b 不含前导 0
func superPow(a int, b []int) int {
	if a == 1 {
		return 1
	}
	base := 1337
	// x^n
	mypow := func(x, n int) int {
		if n == 0 {
			return 1
		}
		result := 1
		for i := n; i != 0; i >>= 1 {
			// 奇数
			if i&1 == 1 {
				result *= x
				result %= base
			}
			x *= x
			x %= base
		}
		return result
	}
	a %= base

	result := mypow(a, b[0])

	for i := 1; i < len(b); i++ {
		result = mypow(result, 10) * mypow(a, b[i]) % base
	}

	return result
}

// 386. 字典序排数
// 给你一个整数 n ，按字典序返回范围 [1, n] 内所有整数。
//
// 你必须设计一个时间复杂度为 O(n) 且使用 O(1) 额外空间的算法。
//
// 示例 1：
// 输入：n = 13
// 输出：[1,10,11,12,13,2,3,4,5,6,7,8,9]
//
// 示例 2：
// 输入：n = 2
// 输出：[1,2]
//
// 提示：
// 1 <= n <= 5 * 104
func lexicalOrder(n int) []int {
	result := make([]int, 0)

	var dfs func(num int)

	dfs = func(num int) {
		for i := 0; i <= 9; i++ {
			nextNum := num*10 + i
			if nextNum > n || nextNum <= 0 {
				continue
			}
			result = append(result, nextNum)
			dfs(nextNum)
		}
	}
	dfs(0)
	return result
}

// 390. 消除游戏
// 给定一个从1 到 n 排序的整数列表。
// 首先，从左到右，从第一个数字开始，每隔一个数字进行删除，直到列表的末尾。
// 第二步，在剩下的数字中，从右到左，从倒数第一个数字开始，每隔一个数字进行删除，直到列表开头。
// 我们不断重复这两步，从左到右和从右到左交替进行，直到只剩下一个数字。
// 返回长度为 n 的列表中，最后剩下的数字。
//
// 示例：
// 输入:
// n = 9,
// 1 2 3 4 5 6 7 8 9
// 2 4 6 8
// 2 6
// 6
//
// 输出: 6
func lastRemaining(n int) int {
	if n == 1 {
		return 1
	}
	helf := n >> 1
	return 2 * (helf + 1 - lastRemaining(helf))
}

// 393. UTF-8 编码验证
// UTF-8 中的一个字符可能的长度为 1 到 4 字节，遵循以下的规则：
//
// 对于 1 字节的字符，字节的第一位设为 0 ，后面 7 位为这个符号的 unicode 码。
// 对于 n 字节的字符 (n > 1)，第一个字节的前 n 位都设为1，第 n+1 位设为 0 ，后面字节的前两位一律设为 10 。剩下的没有提及的二进制位，全部为这个符号的 unicode 码。
// 这是 UTF-8 编码的工作方式：
//
//   Char. number range  |        UTF-8 octet sequence
//      (hexadecimal)    |              (binary)
//   --------------------+---------------------------------------------
//   0000 0000-0000 007F | 0xxxxxxx
//   0000 0080-0000 07FF | 110xxxxx 10xxxxxx
//   0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx
//   0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
// 给定一个表示数据的整数数组，返回它是否为有效的 utf-8 编码。
//
// 注意：
// 输入是整数数组。只有每个整数的 最低 8 个有效位 用来存储数据。这意味着每个整数只表示 1 字节的数据。
//
// 示例 1：
// data = [197, 130, 1], 表示 8 位的序列: 11000101 10000010 00000001.
// 返回 true 。
// 这是有效的 utf-8 编码，为一个2字节字符，跟着一个1字节字符。
//
// 示例 2：
// data = [235, 140, 4], 表示 8 位的序列: 11101011 10001100 00000100.
// 返回 false 。
// 前 3 位都是 1 ，第 4 位为 0 表示它是一个3字节字符。
// 下一个字节是开头为 10 的延续字节，这是正确的。
// 但第二个延续字节不以 10 开头，所以是不符合规则的。
func validUtf8(data []int) bool {
	n := len(data)
	if n == 0 {
		return false
	}
	for _, num := range data {
		if num >= (1<<8) || num < 0 {
			return false
		}
	}
	mask1, mask2 := 1<<7, 1<<6
	// 获取字节数
	getByteCount := func(num int) int {
		// 字节的第一位是0 1个字节
		if num&mask1 == 0 {
			return 1
		}
		count := 0

		mask := mask1
		for mask&num != 0 {
			count++
			if count > 4 {
				return 0
			}
			mask >>= 1
		}
		if count == 1 {
			return 0
		}
		return count
	}
	idx := 0
	for idx < n {
		byteCount := getByteCount(data[idx])
		if byteCount == 0 {
			return false
		}
		if idx+byteCount > n {
			return false
		}
		// 10 开头
		for i := 1; i < byteCount; i++ {
			if data[idx+i]&mask1 == 0 || data[idx+i]&mask2 != 0 {
				return false
			}
		}
		idx += byteCount
	}

	return true
}

// 397. 整数替换
// 给定一个正整数 n ，你可以做如下操作：
//
// 如果 n 是偶数，则用 n / 2替换 n 。
// 如果 n 是奇数，则可以用 n + 1或n - 1替换 n 。
// n 变为 1 所需的最小替换次数是多少？
//
// 示例 1：
// 输入：n = 8
// 输出：3
// 解释：8 -> 4 -> 2 -> 1
//
// 示例 2：
// 输入：n = 7
// 输出：4
// 解释：7 -> 8 -> 4 -> 2 -> 1
// 或 7 -> 6 -> 3 -> 2 -> 1
//
// 示例 3：
// 输入：n = 4
// 输出：2
//
// 提示：
// 1 <= n <= 231 - 1
func integerReplacement(n int) int {
	if n == 1 {
		return 0
	}
	if n == math.MaxInt32 {
		return 32
	}
	dp := make(map[int]int)

	var getReplacement func(num int) int

	getReplacement = func(num int) int {
		if num == 1 {
			return 0
		}
		if v, ok := dp[num]; ok {
			return v
		}
		var result int

		if num&1 == 0 {
			result = 1 + getReplacement(num>>1)
		} else {
			result = 1 + min(getReplacement(num+1), getReplacement(num-1))
		}
		dp[num] = result
		return result
	}
	return getReplacement(n)
}

// 396. 旋转函数
// 给定一个长度为 n 的整数数组 A 。
//
// 假设 Bk 是数组 A 顺时针旋转 k 个位置后的数组，我们定义 A 的“旋转函数” F 为：
// F(k) = 0 * Bk[0] + 1 * Bk[1] + ... + (n-1) * Bk[n-1]。
//
// 计算F(0), F(1), ..., F(n-1)中的最大值。
//
// 注意:
// 可以认为 n 的值小于 105。
//
// 示例:
//
// A = [4, 3, 2, 6]
// F(0) = (0 * 4) + (1 * 3) + (2 * 2) + (3 * 6) = 0 + 3 + 4 + 18 = 25
// F(1) = (0 * 6) + (1 * 4) + (2 * 3) + (3 * 2) = 0 + 4 + 6 + 6 = 16
// F(2) = (0 * 2) + (1 * 6) + (2 * 4) + (3 * 3) = 0 + 6 + 8 + 9 = 23
// F(3) = (0 * 3) + (1 * 2) + (2 * 6) + (3 * 4) = 0 + 2 + 12 + 12 = 26
//
// 所以 F(0), F(1), F(2), F(3) 中的最大值是 F(3) = 26 。
func maxRotateFunction(nums []int) int {
	n := len(nums)
	rotateSum, sum := 0, 0
	for i, num := range nums {
		rotateSum += i * num
		sum += num
	}
	result := rotateSum

	for i := n - 1; i > 0; i-- {
		rotateSum += sum - n*nums[i]
		result = max(result, rotateSum)
	}

	return result
}

// 400. 第 N 位数字
// 在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 n 位数字。
//
// 注意：n 是正数且在 32 位整数范围内（n < 231）。
//
// 示例 1：
// 输入：3 输出：3
//
// 示例 2：
// 输入：11
// 输出：0
// 解释：第 11 位数字在序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是 0 ，它是 10 的一部分。
func findNthDigit(n int) int {
	// 位数
	digit := 1
	// start 开始数字 1 10 100
	// count n位数的 个数
	start, count := 1, 9
	for n > count {
		n -= count
		digit++
		start *= 10
		count = digit * start * 9
	}
	s := strconv.Itoa(start + (n-1)/digit)
	idx := (n - 1) % digit
	return int(s[idx] - '0')
}

// 869. 重新排序得到 2 的幂
// 给定正整数 N ，我们按任何顺序（包括原始顺序）将数字重新排序，注意其前导数字不能为零。
//
// 如果我们可以通过上述方式得到 2 的幂，返回 true；否则，返回 false。
//
// 示例 1：
// 输入：1 输出：true
//
// 示例 2：
// 输入：10 输出：false
//
// 示例 3：
// 输入：16 输出：true
//
// 示例 4：
// 输入：24 输出：false
//
// 示例 5：
// 输入：46 输出：true
//
// 提示：
// 1 <= N <= 10^9
func reorderedPowerOf2(n int) bool {
	if isPowerOfTwo(n) {
		return true
	}
	compareNum := func(num1, num2 [10]int) bool {
		for i := 0; i < 10; i++ {
			if num1[i] != num2[i] {
				return false
			}
		}
		return true
	}
	getNums := func(num int) [10]int {
		var nums [10]int

		for num > 0 {
			nums[num%10]++
			num /= 10
		}
		return nums
	}

	nNums := getNums(n)
	for i := 0; i <= 31; i++ {
		if compareNum(nNums, getNums(1<<i)) {
			return true
		}
	}

	return false
}

// 423. 从英文中重建数字
// 给定一个非空字符串，其中包含字母顺序打乱的英文单词表示的数字0-9。按升序输出原始的数字。
//
// 注意:
// 输入只包含小写英文字母。
// 输入保证合法并可以转换为原始的数字，这意味着像 "abc" 或 "zerone" 的输入是不允许的。
// 输入字符串的长度小于 50,000。
//
// 示例 1:
// 输入: "owoztneoer" 输出: "012" (zeroonetwo)
//
// 示例 2:
// 输入: "fviefuro" 输出: "45" (fourfive)
func originalDigits(s string) string {
	// 建立字符 到 数字的映射
	// z => 0 zero
	// w => 2 two
	// u => 4 four
	// x => 6 six
	// g => 8 eight
	// h => 3 - 8  three eight
	// f => 5 - 4  five four
	// s => 7 - 6  seven six
	// i => 9 - 5 - 6 - 8  nine five six eight
	// n => 1 - 7 - 2 * 9  one seven nine
	letters := [26]int{}
	for i := 0; i < len(s); i++ {
		letters[s[i]-'a']++
	}
	counts := [10]int{}
	counts[0] = letters['z'-'a']
	counts[2] = letters['w'-'a']
	counts[4] = letters['u'-'a']
	counts[6] = letters['x'-'a']
	counts[8] = letters['g'-'a']
	counts[3] = letters['h'-'a'] - counts[8]
	counts[5] = letters['f'-'a'] - counts[4]
	counts[7] = letters['s'-'a'] - counts[6]

	counts[9] = letters['i'-'a'] - counts[5] - counts[6] - counts[8]
	counts[1] = letters['n'-'a'] - counts[7] - 2*counts[9]

	var builder strings.Builder
	for i := 0; i < 10; i++ {
		num := strconv.Itoa(i)
		builder.WriteString(strings.Repeat(num, counts[i]))
	}
	return builder.String()
}

// 421. 数组中两个数的最大异或值
// 给你一个整数数组 nums ，返回 nums[i] XOR nums[j] 的最大运算结果，其中 0 ≤ i ≤ j < n 。
//
// 进阶：你可以在 O(n) 的时间解决这个问题吗？
//
// 示例 1：
// 输入：nums = [3,10,5,25,2,8]
// 输出：28
// 解释：最大运算结果是 5 XOR 25 = 28.
//
// 示例 2：
// 输入：nums = [0]
// 输出：0
//
// 示例 3：
// 输入：nums = [2,4]
// 输出：6
//
// 示例 4：
// 输入：nums = [8,10,2]
// 输出：10
//
// 示例 5：
// 输入：nums = [14,70,53,83,49,91,36,80,92,51,66,70]
// 输出：127
//
// 提示：
// 1 <= nums.length <= 2 * 104
// 0 <= nums[i] <= 231 - 1
func findMaximumXOR(nums []int) int {
	x := 0

	for k := 30; k >= 0; k-- {
		seen := make(map[int]bool)
		// 判断 第 k 为有没有1
		for _, num := range nums {
			seen[num>>k] = true
		}
		next := (x << 1) + 1

		found := false
		for _, num := range nums {
			if seen[num>>k^next] {
				found = true
				break
			}
		}
		if found {
			x = next
		} else {
			//
			x = next - 1
		}

	}

	return x
}

// 440. 字典序的第K小数字
// 给定整数 n 和 k，找到 1 到 n 中字典序第 k 小的数字。
//
// 注意：1 ≤ k ≤ n ≤ 109。
//
// 示例 :
// 输入: n: 13   k: 2
// 输出: 10
// 解释:
// 字典序的排列是 [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]，所以第二小的数字是 10。
func findKthNumber(n int, k int) int {

	getCount := func(first, last int) int {
		count := 0
		for first <= n {
			count += min(n+1, last) - first
			first *= 10
			last *= 10
		}
		return count
	}

	num := 1
	k--
	for k > 0 {
		// 计算 num 和 num + 1 直接有多少 个元素
		count := getCount(num, num+1)
		if count <= k {
			// 正常序的下一位 2 -> 3
			num++
			k -= count
		} else {
			// 字典序 下一位 2 -> 20
			num *= 10
			k--
		}
	}

	return num
}

// 458. 可怜的小猪
// 有 buckets 桶液体，其中 正好 有一桶含有毒药，其余装的都是水。它们从外观看起来都一样。为了弄清楚哪只水桶含有毒药，你可以喂一些猪喝，通过观察猪是否会死进行判断。不幸的是，你只有 minutesToTest 分钟时间来确定哪桶液体是有毒的。
//
// 喂猪的规则如下：
// 选择若干活猪进行喂养
// 可以允许小猪同时饮用任意数量的桶中的水，并且该过程不需要时间。
// 小猪喝完水后，必须有 minutesToDie 分钟的冷却时间。在这段时间里，你只能观察，而不允许继续喂猪。
// 过了 minutesToDie 分钟后，所有喝到毒药的猪都会死去，其他所有猪都会活下来。
// 重复这一过程，直到时间用完。
// 给你桶的数目 buckets ，minutesToDie 和 minutesToTest ，返回在规定时间内判断哪个桶有毒所需的 最小 猪数。
//
// 示例 1：
// 输入：buckets = 1000, minutesToDie = 15, minutesToTest = 60
// 输出：5
//
// 示例 2：
// 输入：buckets = 4, minutesToDie = 15, minutesToTest = 15
// 输出：2
//
// 示例 3：
// 输入：buckets = 4, minutesToDie = 15, minutesToTest = 30
// 输出：2
//
// 提示：
// 1 <= buckets <= 1000
// 1 <= minutesToDie <= minutesToTest <= 100
func poorPigs(buckets int, minutesToDie int, minutesToTest int) int {
	states := minutesToTest/minutesToDie + 1
	return int(math.Ceil(math.Log(float64(buckets)) / math.Log(float64(states))))
}

// 470. 用 Rand7() 实现 Rand10()
// 已有方法 rand7 可生成 1 到 7 范围内的均匀随机整数，试写一个方法 rand10 生成 1 到 10 范围内的均匀随机整数。
//
// 不要使用系统的 Math.random() 方法。
//
// 示例 1:
// 输入: 1 输出: [7]
//
// 示例 2:
// 输入: 2 输出: [8,4]
//
// 示例 3:
// 输入: 3 输出: [8,1,10]
//
// 提示:
// rand7 已定义。
// 传入参数: n 表示 rand10 的调用次数。
//
// 进阶:
// rand7()调用次数的 期望值 是多少 ?
// 你能否尽量少调用 rand7() ?
func rand10() int {
	var rand7 func() int
	//  (randX() - 1)*Y + randY() 可以等概率的生成[1, X * Y]
	result := 1
	for true {
		// 1 ~ 49
		num := (rand7()-1)*7 + rand7()
		if num <= 40 {
			result = 1 + num%10
			break
		}
		// 41 ~ 49  63
		num = (num-41)*7 + rand7()
		if num <= 60 {
			result = 1 + num%10
			break
		}
		// 61 ~63  21
		num = (num-61)*7 + rand7()
		if num <= 20 {
			result = 1 + num%10
			break
		}
	}

	return result
}

// 477. 汉明距离总和
// 两个整数的 汉明距离 指的是这两个数字的二进制数对应位不同的数量。
// 给你一个整数数组 nums，请你计算并返回 nums 中任意两个数之间 汉明距离的总和 。
//
// 示例 1：
// 输入：nums = [4,14,2]
// 输出：6
// 解释：在二进制表示中，4 表示为 0100 ，14 表示为 1110 ，2表示为 0010 。（这样表示是为了体现后四位之间关系）
// 所以答案为：
// HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6
//
// 示例 2：
// 输入：nums = [4,14,4]
// 输出：4
//
// 提示：
// 1 <= nums.length <= 104
// 0 <= nums[i] <= 109
// 给定输入的对应答案符合 32-bit 整数范围
func totalHammingDistance(nums []int) int {
	// 计算每位 1 的个数
	counts := make([]int, 32)
	for _, num := range nums {
		for i := 0; num > 0; i++ {
			counts[i] += 1 & num
			num >>= 1
		}
	}
	result, n := 0, len(nums)
	for _, count := range counts {
		// count 个1  n-count个0
		result += count * (n - count)
	}

	return result
}

// 479. 最大回文数乘积
// 你需要找到由两个 n 位数的乘积组成的最大回文数。
// 由于结果会很大，你只需返回最大回文数 mod 1337得到的结果。
//
// 示例:
// 输入: 2
// 输出: 987
// 解释: 99 x 91 = 9009, 9009 % 1337 = 987
//
// 说明:
// n 的取值范围为 [1,8]。
func largestPalindrome(n int) int {
	if n == 1 {
		return 9
	}
	maxVal := 1
	for i := 0; i < n; i++ {
		maxVal *= 10
	}
	mod := 1337
	maxVal--
	// 而相乘可以构成9的尾数只有3(33),7(77),9(9*1)
	for i := maxVal - 1; i > maxVal/10; i-- {
		// 构造回文数 9889 后缀
		prev, sufNum := i, i
		for sufNum > 0 {
			prev = prev*10 + sufNum%10
			sufNum /= 10
		}
		num := maxVal
		for num > 0 && num*num >= prev {
			if prev%num == 0 {
				return prev % mod
			} else if num%10 == 9 {
				num -= 2
			} else {
				num -= 4
			}
		}
	}
	return -1
}

// 483. 最小好进制
// 对于给定的整数 n, 如果n的k（k>=2）进制数的所有数位全为1，则称 k（k>=2）是 n 的一个好进制。
//
// 以字符串的形式给出 n, 以字符串的形式返回 n 的最小好进制。
//
// 示例 1：
// 输入："13" 输出："3"
// 解释：13 的 3 进制是 111。
//
// 示例 2：
// 输入："4681" 输出："8"
// 解释：4681 的 8 进制是 11111。
//
// 示例 3：
// 输入："1000000000000000000"
// 输出："999999999999999999"
// 解释：1000000000000000000 的 999999999999999999 进制是 11。
//
// 提示：
// n的取值范围是 [3, 10^18]。
// 输入总是有效且没有前导 0。
func smallestGoodBase(n string) string {
	// 最后1位是1
	num, _ := strconv.Atoi(n)

	for m := 59; m > 1; m-- {
		k := int(math.Pow(float64(num), 1.0/float64(m)))
		if k <= 1 {
			continue
		}
		sum := 0
		for i := 0; i <= m; i++ {
			sum = sum*k + 1
		}
		if sum == num {
			return strconv.Itoa(k)
		}
	}
	return strconv.Itoa(num - 1)
}

// 537. 复数乘法
// 复数 可以用字符串表示，遵循 "实部+虚部i" 的形式，并满足下述条件：
//
// 实部 是一个整数，取值范围是 [-100, 100]
// 虚部 也是一个整数，取值范围是 [-100, 100]
// i2 == -1
// 给你两个字符串表示的复数 num1 和 num2 ，请你遵循复数表示形式，返回表示它们乘积的字符串。
//
// 示例 1：
// 输入：num1 = "1+1i", num2 = "1+1i"
// 输出："0+2i"
// 解释：(1 + i) * (1 + i) = 1 + i2 + 2 * i = 2i ，你需要将它转换为 0+2i 的形式。
//
// 示例 2：
// 输入：num1 = "1+-1i", num2 = "1+-1i"
// 输出："0+-2i"
// 解释：(1 - i) * (1 - i) = 1 + i2 - 2 * i = -2i ，你需要将它转换为 0+-2i 的形式。
//
// 提示：
// num1 和 num2 都是有效的复数表示。
func complexNumberMultiply(num1 string, num2 string) string {
	var num1Real, num1Imag, num2Real, num2Imag int

	_, err := fmt.Sscanf(num1, "%d+%di", &num1Real, &num1Imag)
	if err != nil {
		return ""
	}
	_, err = fmt.Sscanf(num2, "%d+%di", &num2Real, &num2Imag)
	if err != nil {
		return ""
	}

	return fmt.Sprintf("%d+%di", num1Real*num2Real-num1Imag*num2Imag,
		num1Real*num2Imag+num1Imag*num2Real)
}

// 553. 最优除法
// 给定一组正整数，相邻的整数之间将会进行浮点除法操作。例如， [2,3,4] -> 2 / 3 / 4 。
//
// 但是，你可以在任意位置添加任意数目的括号，来改变算数的优先级。你需要找出怎么添加括号，才能得到最大的结果，并且返回相应的字符串格式的表达式。你的表达式不应该含有冗余的括号。
//
// 示例：
// 输入: [1000,100,10,2] 输出: "1000/(100/10/2)"
// 解释:
// 1000/(100/10/2) = 1000/((100/10)/2) = 200
// 但是，以下加粗的括号 "1000/((100/10)/2)" 是冗余的，
// 因为他们并不影响操作的优先级，所以你需要返回 "1000/(100/10/2)"。
//
// 其他用例:
// 1000/(100/10)/2 = 50
// 1000/(100/(10/2)) = 50
// 1000/100/10/2 = 0.5
// 1000/100/(10/2) = 2
// 说明:
// 输入数组的长度在 [1, 10] 之间。
// 数组中每个元素的大小都在 [2, 1000] 之间。
// 每个测试用例只有一个最优除法解。
func optimalDivision(nums []int) string {
	n := len(nums)

	if n == 1 {
		return strconv.Itoa(nums[0])
	}
	if n == 2 {
		return fmt.Sprintf("%d/%d", nums[0], nums[1])
	}
	// 贪心
	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("%d/(%d", nums[0], nums[1]))

	for i := 2; i < n; i++ {
		builder.WriteString(fmt.Sprintf("/%d", nums[i]))
	}
	builder.WriteString(")")

	return builder.String()
}

// 587. 安装栅栏
// 在一个二维的花园中，有一些用 (x, y) 坐标表示的树。由于安装费用十分昂贵，你的任务是先用最短的绳子围起所有的树。只有当所有的树都被绳子包围时，花园才能围好栅栏。你需要找到正好位于栅栏边界上的树的坐标。
//
// 示例 1:
// 输入: [[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]]
// 输出: [[1,1],[2,0],[4,2],[3,3],[2,4]]
// 解释:
//
// 示例 2:
// 输入: [[1,2],[2,2],[4,2]]
// 输出: [[1,2],[2,2],[4,2]]
// 解释:
// 即使树都在一条直线上，你也需要先用绳子包围它们。
//
// 注意:
// 所有的树应当被围在一起。你不能剪断绳子来包围树或者把树分成一组以上。
// 输入的整数在 0 到 100 之间。
// 花园至少有一棵树。
// 所有树的坐标都是不同的。
// 输入的点没有顺序。输出顺序也没有要求。
func outerTrees(trees [][]int) [][]int {
	n := len(trees)
	if n <= 3 {
		return trees
	}

	getLeft := func() []int {
		left := trees[0]
		for i := 1; i < n; i++ {
			if trees[i][0] < left[0] {
				left = trees[i]
			}
		}
		return left
	}

	// 角度
	getOrientation := func(left, p, q []int) int {
		return (p[1]-left[1])*(q[0]-left[0]) - (p[0]-left[0])*(q[1]-left[1])
	}

	// 距离
	getDistance := func(p, q []int) int {
		return (p[0]-q[0])*(p[0]-q[0]) + (p[1]-q[1])*(p[1]-q[1])
	}
	// 找到最左端的点
	left := getLeft()
	// 排序
	sort.Slice(trees, func(i, j int) bool {
		diff := getOrientation(left, trees[i], trees[j]) - getOrientation(left, trees[j], trees[i])
		if diff == 0 {
			return getDistance(left, trees[i]) < getDistance(left, trees[j])
		}
		return diff < 0
	})
	// 最后一条边如果有多个点, 需要交换顺序
	idx := n - 1
	for idx > 0 && getOrientation(left, trees[n-1], trees[idx-1]) == 0 {
		idx--
	}
	for l, r := idx, n-1; l < r; {
		trees[l], trees[r] = trees[r], trees[l]
		l++
		r--
	}
	stack := list.New()
	for _, point := range trees {
		if stack.Len() < 2 {
			stack.PushBack(point)
			continue
		}
		back := stack.Back()
		stack.Remove(back)
		backPoint := back.Value.([]int)
		for stack.Len() >= 2 && getOrientation(stack.Back().Value.([]int), backPoint, point) > 0 {
			back = stack.Back()
			stack.Remove(back)
			backPoint = back.Value.([]int)
		}
		stack.PushBack(backPoint)
		stack.PushBack(point)
	}
	result := make([][]int, stack.Len())
	for i := stack.Len() - 1; i >= 0; i-- {
		back := stack.Back()
		stack.Remove(back)
		result[i] = back.Value.([]int)
	}

	return result
}

// 593. 有效的正方形
// 给定二维空间中四点的坐标，返回四点是否可以构造一个正方形。
//
// 一个点的坐标（x，y）由一个有两个整数的整数数组表示。
//
// 示例:
// 输入: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]
// 输出: True
//
// 注意:
// 所有输入整数都在 [-10000，10000] 范围内。
// 一个有效的正方形有四个等长的正长和四个等角（90度角）。
// 输入点没有顺序。
func validSquare(p1 []int, p2 []int, p3 []int, p4 []int) bool {
	// 四个点 -> 六条边
	sides := make([]int, 6)

	getSideSquare := func(p, q []int) int {
		return (p[0]-q[0])*(p[0]-q[0]) + (p[1]-q[1])*(p[1]-q[1])
	}
	sides[0] = getSideSquare(p1, p2)
	sides[1] = getSideSquare(p1, p3)
	sides[2] = getSideSquare(p1, p4)
	sides[3] = getSideSquare(p2, p3)
	sides[4] = getSideSquare(p2, p4)
	sides[5] = getSideSquare(p3, p4)
	sort.Ints(sides)

	// 前4条边相等
	side, diagonal := sides[0], sides[4]
	if side == 0 {
		return false
	}
	if side != sides[1] || side != sides[2] || side != sides[3] {
		return false
	}
	if diagonal != sides[5] {
		return false
	}
	// 对角线的平方是 边长平方的两倍
	return side*2 == diagonal
}

// 1154. 一年中的第几天
// 给你一个字符串 date ，按 YYYY-MM-DD 格式表示一个 现行公元纪年法 日期。请你计算并返回该日期是当年的第几天。
//
// 通常情况下，我们认为 1 月 1 日是每年的第 1 天，1 月 2 日是每年的第 2 天，依此类推。每个月的天数与现行公元纪年法（格里高利历）一致。
//
// 示例 1：
// 输入：date = "2019-01-09" 输出：9
//
// 示例 2：
// 输入：date = "2019-02-10" 输出：41
//
// 示例 3：
// 输入：date = "2003-03-01" 输出：60
//
// 示例 4：
// 输入：date = "2004-03-01" 输出：61
//
// 提示：
// date.length == 10
// date[4] == date[7] == '-'，其他的 date[i] 都是数字
// date 表示的范围从 1900 年 1 月 1 日至 2019 年 12 月 31 日
func dayOfYear(date string) int {
	isLeapYear := func(year int) bool {
		if year%400 == 0 {
			return true
		}
		return year%100 != 0 && year%4 == 0
	}

	year, _ := strconv.Atoi(date[:4])
	month, _ := strconv.Atoi(date[5:7])
	day, _ := strconv.Atoi(date[8:])
	days := []int{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334}
	result := 0
	result += days[month-1]
	result += day
	if isLeapYear(year) && month > 2 {
		result += 1
	}
	return result
}

// 633. 平方数之和
// 给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c 。
//
// 示例 1：
// 输入：c = 5 输出：true
// 解释：1 * 1 + 2 * 2 = 5
//
// 示例 2：
// 输入：c = 3 输出：false
//
// 示例 3：
// 输入：c = 4 输出：true
//
// 示例 4：
// 输入：c = 2 输出：true
//
// 示例 5：
// 输入：c = 1 输出：true
//
// 提示：
// 0 <= c <= 231 - 1
func judgeSquareSum(c int) bool {
	// 费马平方和定理告诉我们：
	//
	// 一个非负整数 cc 能够表示为两个整数的平方和，当且仅当 cc 的所有形如 4k+34k+3 的质因子的幂次均为偶数。
	//
	// 证明方法可以见 这里。
	//
	// 因此我们对 c 进行质因数分解，再判断形如 4k+3 的质因子的幂次是否均为偶数即可。
	if c < 3 {
		return true
	}

	for i := 1; i*i <= c; i++ {
		num := c - i*i
		b := math.Sqrt(float64(num))
		if b == math.Floor(b) {
			return true
		}

	}

	return false
}

// 650. 只有两个键的键盘
// 最初记事本上只有一个字符 'A' 。你每次可以对这个记事本进行两种操作：
//
// Copy All（复制全部）：复制这个记事本中的所有字符（不允许仅复制部分字符）。
// Paste（粘贴）：粘贴 上一次 复制的字符。
// 给你一个数字 n ，你需要使用最少的操作次数，在记事本上输出 恰好 n 个 'A' 。返回能够打印出 n 个 'A' 的最少操作次数。
//
// 示例 1：
// 输入：3 输出：3
// 解释：
// 最初, 只有一个字符 'A'。
// 第 1 步, 使用 Copy All 操作。
// 第 2 步, 使用 Paste 操作来获得 'AA'。
// 第 3 步, 使用 Paste 操作来获得 'AAA'。
//
// 示例 2：
// 输入：n = 1
// 输出：0
//
// 提示：
// 1 <= n <= 1000
func minSteps(n int) int {
	if n <= 1 {
		return 0
	}
	// 不允许部分复制
	// 素数分解
	d := 2
	result := 0
	for n > 1 {
		for n%d == 0 {
			result += d
			n /= d
		}
		d++
	}
	return result
}

// 670. 最大交换
// 给定一个非负整数，你至多可以交换一次数字中的任意两位。返回你能得到的最大值。
//
// 示例 1 :
// 输入: 2736
// 输出: 7236
// 解释: 交换数字2和数字7。
//
// 示例 2 :
// 输入: 9973
// 输出: 9973
// 解释: 不需要交换。
// 注意:
// 给定数字的范围是 [0, 108]
func maximumSwap(num int) int {
	nums := [9]int{}
	idx := 0
	for num > 0 {
		nums[idx] = num % 10
		idx++
		num /= 10
	}
	// 从后往前 找最大的
	maxIndexs := [9]int{}
	maxIdx := 0
	for i := 0; i < idx; i++ {
		if nums[i] > nums[maxIdx] {
			maxIdx = i
		}
		maxIndexs[i] = maxIdx
	}
	// 从前往后
	for i := idx - 1; i >= 0; i-- {
		maxIdx = maxIndexs[i]
		if nums[maxIdx] != nums[i] {
			nums[maxIdx], nums[i] = nums[i], nums[maxIdx]
			break
		}
	}
	num = 0
	for i := idx - 1; i >= 0; i-- {
		num = num*10 + nums[i]
	}
	return num
}

// 672. 灯泡开关 Ⅱ
// 现有一个房间，墙上挂有 n 只已经打开的灯泡和 4 个按钮。在进行了 m 次未知操作后，你需要返回这 n 只灯泡可能有多少种不同的状态。
//
// 假设这 n 只灯泡被编号为 [1, 2, 3 ..., n]，这 4 个按钮的功能如下：
//
// 1.将所有灯泡的状态反转（即开变为关，关变为开）
// 2.将编号为偶数的灯泡的状态反转
// 3.将编号为奇数的灯泡的状态反转
// 4.将编号为 3k+1 的灯泡的状态反转（k = 0, 1, 2, ...)
//
// 示例 1:
// 输入: n = 1, m = 1. 输出: 2
// 说明: 状态为: [开], [关]
//
// 示例 2:
// 输入: n = 2, m = 1.
// 输出: 3
// 说明: 状态为: [开, 关], [关, 开], [关, 关]
//
// 示例 3:
// 输入: n = 3, m = 1.
// 输出: 4
// 说明: 状态为: [关, 开, 关], [开, 关, 开], [关, 关, 关], [关, 开, 开].
// 注意： n 和 m 都属于 [0, 1000].
func flipLights(n int, presses int) int {
	// 操作2，3 是2个循环; 操作4 是 3个循环
	// 所有操作应该是 6 个灯泡 循环
	// 所以前6个灯泡决定 所有的状态

	// 操作 1 ~ 4  a, b, c , d
	// 灯1 1 + a + c + d
	// 灯2 1 + a + b
	// 灯3 1 + a + c
	// 灯4 1 + a + b + d
	// 灯5 1 + a + c
	// 灯6 1 + a + b

	// presses = 0  111111
	// presses = 1  000000 101010 010101 100100
	// presses = 2  111111 101010 010101 100100
	//              000000 001110 110001
	// presses = 3  111111 101010 010101 100100
	//              000000 001110 110001 010101
	n = min(n, 3)
	result := 0
	if presses == 0 {
		result = 1
	} else if presses == 1 {
		if n == 1 {
			result = 2
		} else if n == 2 {
			result = 3
		} else {
			result = 4
		}
	} else if presses == 2 {
		if n == 1 {
			result = 2
		} else if n == 2 {
			result = 4
		} else {
			result = 7
		}
	} else {
		if n == 1 {
			result = 2
		} else if n == 2 {
			result = 4
		} else {
			result = 8
		}
	}

	return result
}

// 1447. 最简分数
// 给你一个整数 n ，请你返回所有 0 到 1 之间（不包括 0 和 1）满足分母小于等于  n 的 最简 分数 。分数可以以 任意 顺序返回。
//
// 示例 1：
// 输入：n = 2
// 输出：["1/2"]
// 解释："1/2" 是唯一一个分母小于等于 2 的最简分数。
//
// 示例 2：
// 输入：n = 3
// 输出：["1/2","1/3","2/3"]
//
// 示例 3：
// 输入：n = 4
// 输出：["1/2","1/3","1/4","2/3","3/4"]
// 解释："2/4" 不是最简分数，因为它可以化简为 "1/2" 。
//
// 示例 4：
// 输入：n = 1
// 输出：[]
//
// 提示：
// 1 <= n <= 100
func simplifiedFractions(n int) []string {
	result := make([]string, 0)
	for i := 1; i < n; i++ {
		for j := i + 1; j <= n; j++ {
			if getGcd(i, j) != 1 {
				continue
			}
			result = append(result, fmt.Sprintf("%d/%d", i, j))
		}
	}
	return result
}

// 728. 自除数
// 自除数 是指可以被它包含的每一位数整除的数。
//
// 例如，128 是一个 自除数 ，因为 128 % 1 == 0，128 % 2 == 0，128 % 8 == 0。
// 自除数 不允许包含 0 。
//
// 给定两个整数 left 和 right ，返回一个列表，列表的元素是范围 [left, right] 内所有的 自除数 。
//
// 示例 1：
// 输入：left = 1, right = 22
// 输出：[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
//
// 示例 2:
// 输入：left = 47, right = 85
// 输出：[48,55,66,77]
//
// 提示：
// 1 <= left <= right <= 104
func selfDividingNumbers(left int, right int) []int {

	isDividingNumber := func(num int) bool {
		original := num
		divideNum := 0
		for num != 0 {
			divideNum = num % 10
			if divideNum == 0 || original%divideNum != 0 {
				return false
			}

			num /= 10
		}

		return true
	}

	result := make([]int, 0)

	for i := left; i <= right; i++ {
		if isDividingNumber(i) {
			result = append(result, i)
		}
	}

	return result
}

// 868. 二进制间距
// 给定一个正整数 n，找到并返回 n 的二进制表示中两个 相邻 1 之间的 最长距离 。如果不存在两个相邻的 1，返回 0 。
//
// 如果只有 0 将两个 1 分隔开（可能不存在 0 ），则认为这两个 1 彼此 相邻 。两个 1 之间的距离是它们的二进制表示中位置的绝对差。例如，"1001" 中的两个 1 的距离为 3 。
//
// 示例 1：
// 输入：n = 22
// 输出：2
// 解释：22 的二进制是 "10110" 。
// 在 22 的二进制表示中，有三个 1，组成两对相邻的 1 。
// 第一对相邻的 1 中，两个 1 之间的距离为 2 。
// 第二对相邻的 1 中，两个 1 之间的距离为 1 。
// 答案取两个距离之中最大的，也就是 2 。
//
// 示例 2：
// 输入：n = 8
// 输出：0
// 解释：8 的二进制是 "1000" 。
// 在 8 的二进制表示中没有相邻的两个 1，所以返回 0 。
//
// 示例 3：
// 输入：n = 5
// 输出：2
// 解释：5 的二进制是 "101" 。
//
// 提示：
// 1 <= n <= 109
func binaryGap(n int) int {
	for n&1 == 0 {
		n >>= 1
	}
	result, count := 0, 0
	for n != 0 {
		if n&1 == 1 {
			if count > result {
				result = count
			}
			count = 0
		}
		count++
		n >>= 1
	}

	return result
}

// 883. 三维形体投影面积
// 在 n x n 的网格 grid 中，我们放置了一些与 x，y，z 三轴对齐的 1 x 1 x 1 立方体。
//
// 每个值 v = grid[i][j] 表示 v 个正方体叠放在单元格 (i, j) 上。
//
// 现在，我们查看这些立方体在 xy 、yz 和 zx 平面上的投影。
//
// 投影 就像影子，将 三维 形体映射到一个 二维 平面上。从顶部、前面和侧面看立方体时，我们会看到“影子”。
//
// 返回 所有三个投影的总面积 。
//
// 示例 1：
// 输入：[[1,2],[3,4]]
// 输出：17
// 解释：这里有该形体在三个轴对齐平面上的三个投影(“阴影部分”)。
//
// 示例 2:
// 输入：grid = [[2]]
// 输出：5
//
// 示例 3：
// 输入：[[1,0],[0,2]]
// 输出：8
//
// 提示：
// n == grid.length == grid[i].length
// 1 <= n <= 50
// 0 <= grid[i][j] <= 50
func projectionArea(grid [][]int) int {
	n := len(grid)

	result := 0
	// 行
	for i := 0; i < n; i++ {
		maxVal := 0
		for j := 0; j < n; j++ {
			if grid[i][j] > 0 {
				// 俯视图占地
				result++
			}
			maxVal = max(maxVal, grid[i][j])
		}
		result += maxVal
	}
	// 列
	for j := 0; j < n; j++ {
		maxVal := 0
		for i := 0; i < n; i++ {
			maxVal = max(maxVal, grid[i][j])
		}
		result += maxVal
	}

	return result
}

// 908. 最小差值 I
// 给你一个整数数组 nums，和一个整数 k 。
//
// 在一个操作中，您可以选择 0 <= i < nums.length 的任何索引 i 。将 nums[i] 改为 nums[i] + x ，其中 x 是一个范围为 [-k, k] 的整数。对于每个索引 i ，最多 只能 应用 一次 此操作。
//
// nums 的 分数 是 nums 中最大和最小元素的差值。
//
// 在对  nums 中的每个索引最多应用一次上述操作后，返回 nums 的最低 分数 。
//
// 示例 1：
// 输入：nums = [1], k = 0
// 输出：0
// 解释：分数是 max(nums) - min(nums) = 1 - 1 = 0。
//
// 示例 2：
// 输入：nums = [0,10], k = 2
// 输出：6
// 解释：将 nums 改为 [2,8]。分数是 max(nums) - min(nums) = 8 - 2 = 6。
//
// 示例 3：
//
// 输入：nums = [1,3,6], k = 3
// 输出：0
// 解释：将 nums 改为 [4,4,4]。分数是 max(nums) - min(nums) = 4 - 4 = 0。
//
// 提示：
// 1 <= nums.length <= 104
// 0 <= nums[i] <= 104
// 0 <= k <= 104
func smallestRangeI(nums []int, k int) int {
	n := len(nums)
	if n <= 1 {
		return 0
	}
	maxVal, minVal := nums[0], nums[0]
	for i := 1; i < n; i++ {
		maxVal = max(maxVal, nums[i])
		minVal = min(minVal, nums[i])
	}
	if maxVal-minVal <= 2*k {
		return 0
	}
	return maxVal - minVal - 2*k
}

// 942. 增减字符串匹配
// 由范围 [0,n] 内所有整数组成的 n + 1 个整数的排列序列可以表示为长度为 n 的字符串 s ，其中:
//
// 如果 perm[i] < perm[i + 1] ，那么 s[i] == 'I'
// 如果 perm[i] > perm[i + 1] ，那么 s[i] == 'D'
// 给定一个字符串 s ，重构排列 perm 并返回它。如果有多个有效排列perm，则返回其中 任何一个 。
//
// 示例 1：
// 输入：s = "IDID"
// 输出：[0,4,1,3,2]
//
// 示例 2：
// 输入：s = "III"
// 输出：[0,1,2,3]
//
// 示例 3：
// 输入：s = "DDI"
// 输出：[3,2,0,1]
//
// 提示：
// 1 <= s.length <= 105
// s 只包含字符 "I" 或 "D"
func diStringMatch(s string) []int {
	n := len(s)
	low, high := 0, n
	result := make([]int, n+1)
	for i, c := range s {
		if c == 'D' {
			result[i] = high
			high--
		} else {
			result[i] = low
			low++
		}
	}
	result[n] = low
	return result
}

// 1037. 有效的回旋镖
// 给定一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点，如果这些点构成一个 回旋镖 则返回 true 。
//
// 回旋镖 定义为一组三个点，这些点 各不相同 且 不在一条直线上 。
//
// 示例 1：
// 输入：points = [[1,1],[2,3],[3,2]]
// 输出：true
//
// 示例 2：
// 输入：points = [[1,1],[2,2],[3,3]]
// 输出：false
//
// 提示：
// points.length == 3
// points[i].length == 2
// 0 <= xi, yi <= 100
func isBoomerang(points [][]int) bool {
	point1, point2, point3 := points[0], points[1], points[2]

	return !inLine(point1, point2, point3)
}

func inLine(point1, point2, point3 []int) bool {
	x1, x2, x3 := point1[0], point2[0], point3[0]
	y1, y2, y3 := point1[1], point2[1], point3[1]

	return (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
}

// 1175. 质数排列
// 请你帮忙给从 1 到 n 的数设计排列方案，使得所有的「质数」都应该被放在「质数索引」（索引从 1 开始）上；你需要返回可能的方案总数。
//
// 让我们一起来回顾一下「质数」：质数一定是大于 1 的，并且不能用两个小于它的正整数的乘积来表示。
//
// 由于答案可能会很大，所以请你返回答案 模 mod 10^9 + 7 之后的结果即可。
//
// 示例 1：
// 输入：n = 5
// 输出：12
// 解释：举个例子，[1,2,5,4,3] 是一个有效的排列，但 [5,2,3,4,1] 不是，因为在第二种情况里质数 5 被错误地放在索引为 1 的位置上。
//
// 示例 2：
// 输入：n = 100
// 输出：682289015
//
// 提示：
// 1 <= n <= 100
func numPrimeArrangements(n int) int {
	if n < 3 {
		return 1
	}
	count := 0
	nums := make([]bool, n+1)
	for i := 2; i*i <= n; i++ {
		if nums[i] {
			continue
		}
		for j := i * i; j <= n; j += i {
			if nums[j] {
				continue
			}
			nums[j] = true
			count++
		}
	}

	count++
	MOD := 1000000007

	num := 1
	for i := 1; i <= count; i++ {
		num = (num * i) % MOD
	}
	for i := 1; i <= n-count; i++ {
		num = (num * i) % MOD
	}
	return num
}

// 1217. 玩筹码
// 有 n 个筹码。第 i 个筹码的位置是 position[i] 。
//
// 我们需要把所有筹码移到同一个位置。在一步中，我们可以将第 i 个筹码的位置从 position[i] 改变为:
//
// position[i] + 2 或 position[i] - 2 ，此时 cost = 0
// position[i] + 1 或 position[i] - 1 ，此时 cost = 1
// 返回将所有筹码移动到同一位置上所需要的 最小代价 。
//
// 示例 1：
// 输入：position = [1,2,3]
// 输出：1
// 解释：第一步:将位置3的筹码移动到位置1，成本为0。
// 第二步:将位置2的筹码移动到位置1，成本= 1。
// 总成本是1。
//
// 示例 2：
// 输入：position = [2,2,2,3,3]
// 输出：2
// 解释：我们可以把位置3的两个筹码移到位置2。每一步的成本为1。总成本= 2。
//
// 示例 3:
// 输入：position = [1,1000000000]
// 输出：1
//
// 提示：
// 1 <= chips.length <= 100
// 1 <= chips[i] <= 10^9
func minCostToMoveChips(position []int) int {
	odd, even := 0, 0
	for _, num := range position {
		if num&1 == 1 {
			odd++
		} else {
			even++
		}
	}

	return min(even, odd)
}

// 736. Lisp 语法解析
// 给你一个类似 Lisp 语句的字符串表达式 expression，求出其计算结果。
//
// 表达式语法如下所示:
//
// 表达式可以为整数，let 表达式，add 表达式，mult 表达式，或赋值的变量。表达式的结果总是一个整数。
// (整数可以是正整数、负整数、0)
// let 表达式采用 "(let v1 e1 v2 e2 ... vn en expr)" 的形式，其中 let 总是以字符串 "let"来表示，接下来会跟随一对或多对交替的变量和表达式，也就是说，第一个变量 v1被分配为表达式 e1 的值，第二个变量 v2 被分配为表达式 e2 的值，依次类推；最终 let 表达式的值为 expr表达式的值。
// add 表达式表示为 "(add e1 e2)" ，其中 add 总是以字符串 "add" 来表示，该表达式总是包含两个表达式 e1、e2 ，最终结果是 e1 表达式的值与 e2 表达式的值之 和 。
// mult 表达式表示为 "(mult e1 e2)" ，其中 mult 总是以字符串 "mult" 表示，该表达式总是包含两个表达式 e1、e2，最终结果是 e1 表达式的值与 e2 表达式的值之 积 。
// 在该题目中，变量名以小写字符开始，之后跟随 0 个或多个小写字符或数字。为了方便，"add" ，"let" ，"mult" 会被定义为 "关键字" ，不会用作变量名。
// 最后，要说一下作用域的概念。计算变量名所对应的表达式时，在计算上下文中，首先检查最内层作用域（按括号计），然后按顺序依次检查外部作用域。测试用例中每一个表达式都是合法的。有关作用域的更多详细信息，请参阅示例。
//
// 示例 1：
// 输入：expression = "(let x 2 (mult x (let x 3 y 4 (add x y))))"
// 输出：14
// 解释：
// 计算表达式 (add x y), 在检查变量 x 值时，
// 在变量的上下文中由最内层作用域依次向外检查。
// 首先找到 x = 3, 所以此处的 x 值是 3 。
//
// 示例 2：
// 输入：expression = "(let x 3 x 2 x)"
// 输出：2
// 解释：let 语句中的赋值运算按顺序处理即可。
//
// 示例 3：
// 输入：expression = "(let x 1 y 2 x (add x y) (add x y))"
// 输出：5
// 解释：
// 第一个 (add x y) 计算结果是 3，并且将此值赋给了 x 。
// 第二个 (add x y) 计算结果是 3 + 2 = 5 。
//
// 提示：
// 1 <= expression.length <= 2000
// exprssion 中不含前导和尾随空格
// expressoin 中的不同部分（token）之间用单个空格进行分隔
// 答案和所有中间计算结果都符合 32-bit 整数范围
// 测试用例中的表达式均为合法的且最终结果为整数
func evaluate(expression string) int {

	return cal(expression, make(map[string]int))
}

func cal(expression string, params map[string]int) int {
	firstChar := expression[0]
	if firstChar == '(' {
		expression = expression[1 : len(expression)-1]
	} else {
		// 变量
		if unicode.IsLetter(rune(firstChar)) {
			return params[expression]
		} else {
			num, _ := strconv.Atoi(expression)
			return num
		}
	}
	exps := splitExpression(expression)
	op := exps[0]

	tmpParams := make(map[string]int)
	for k, v := range params {
		tmpParams[k] = v
	}

	switch op {
	case "let":
		{
			for i := 1; i < len(exps)-1; i += 2 {
				params[exps[i]] = cal(exps[i+1], tmpParams)
			}
			// 计算最后一个表达式的值
			exp := exps[len(exps)-1]
			return cal(exp, tmpParams)
		}
	case "add":
		return cal(exps[1], tmpParams) + cal(exps[2], tmpParams)
	case "mult":
		return cal(exps[1], tmpParams) * cal(exps[2], tmpParams)

	}

	return 0
}

func splitExpression(expression string) []string {
	// 拆分表达式
	result := make([]string, 0)
	start, leftCount := 0, 0
	for i, c := range expression {
		if c == '(' {
			leftCount++
		} else if c == ')' {
			leftCount--
		} else if leftCount == 0 && c == ' ' {
			result = append(result, expression[start:i])
			start = i + 1
		}
	}
	result = append(result, expression[start:])
	return result
}

// 754. 到达终点数字
// 在一根无限长的数轴上，你站在0的位置。终点在target的位置。
//
// 你可以做一些数量的移动 numMoves :
//
// 每次你可以选择向左或向右移动。
// 第 i 次移动（从  i == 1 开始，到 i == numMoves ），在选择的方向上走 i 步。
// 给定整数 target ，返回 到达目标所需的 最小 移动次数(即最小 numMoves ) 。
//
// 示例 1:
// 输入: target = 2
// 输出: 3
// 解释:
// 第一次移动，从 0 到 1 。
// 第二次移动，从 1 到 -1 。
// 第三次移动，从 -1 到 2 。
//
// 示例 2:
// 输入: target = 3
// 输出: 2
// 解释:
// 第一次移动，从 0 到 1 。
// 第二次移动，从 1 到 3 。
//
// 提示:
// -109 <= target <= 109
// target != 0
func reachNumber(target int) int {
	target = abs(target)
	k := 0
	for target > 0 {
		k++
		target -= k
	}
	// sum(1 ~ k) > target
	// 让部分元素变成负数
	// delta 是 余数
	// 1. 余数是偶数 可以 通过 把 (delta/2)个数字变成负数  -1 + 2 - 3 + 4 (构造 2个 1)

	if target&1 == 0 {
		return k
	}

	// 2. 余数是奇数
	return k + 1 + k%2
}

// 762. 二进制表示中质数个计算置位
// 给你两个整数 left 和 right ，在闭区间 [left, right] 范围内，统计并返回 计算置位位数为质数 的整数个数。
//
// 计算置位位数 就是二进制表示中 1 的个数。
//
// 例如， 21 的二进制表示 10101 有 3 个计算置位。
//
// 示例 1：
// 输入：left = 6, right = 10
// 输出：4
// 解释：
// 6 -> 110 (2 个计算置位，2 是质数)
// 7 -> 111 (3 个计算置位，3 是质数)
// 9 -> 1001 (2 个计算置位，2 是质数)
// 10-> 1010 (2 个计算置位，2 是质数)
// 共计 4 个计算置位为质数的数字。
//
// 示例 2：
// 输入：left = 10, right = 15
// 输出：5
// 解释：
// 10 -> 1010 (2 个计算置位, 2 是质数)
// 11 -> 1011 (3 个计算置位, 3 是质数)
// 12 -> 1100 (2 个计算置位, 2 是质数)
// 13 -> 1101 (3 个计算置位, 3 是质数)
// 14 -> 1110 (3 个计算置位, 3 是质数)
// 15 -> 1111 (4 个计算置位, 4 不是质数)
// 共计 5 个计算置位为质数的数字。
func countPrimeSetBits(left int, right int) int {
	result := 0
	primes := []int{0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0}
	for num := left; num <= right; num++ {
		bit := bits.OnesCount(uint(num))
		result += primes[bit]
	}
	return result
}
