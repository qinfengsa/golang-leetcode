package math

import (
	"fmt"
	"log"
	"math"
	"math/bits"
	"strconv"
	"strings"
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
		result = fmt.Sprintf("%d", num%7) + result
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
	for i := 2; i*i < num; i++ {
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
func fib(N int) int {
	if N == 0 {
		return 0
	}
	a, b := 0, 1
	for i := 2; i <= N; i++ {
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
	max := 0
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
	max = max1 * max2 * max3
	if min2 < 0 {
		num := min1 * min2 * max1
		if num > max {
			max = num
		}
	}
	return max
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
	last := n & 1
	n >>= 1
	for n > 0 {
		tmp := n & 1
		if tmp == last {
			return false
		}
		n >>= 1
		last = tmp
	}
	return true
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

	inLine := func(point1, point2, point3 []int) bool {
		x1, x2, x3 := point1[0], point2[0], point3[0]
		y1, y2, y3 := point1[1], point2[1], point3[1]

		return (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
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
