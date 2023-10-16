package string

import (
	"container/list"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// 13. 罗马数字转整数
// 罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。
//
// 字符          数值
// I             1
// V             5
// X             10
// L             50
// C             100
// D             500
// M             1000
// 例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
//
// 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
//
// I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
// X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。
// C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
// 给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。
//
// 示例 1: 输入: "III" 输出: 3
// 示例 2: 输入: "IV" 输出: 4
// 示例 3: 输入: "IX" 输出: 9
// 示例 4: 输入: "LVIII" 输出: 58  解释: L = 50, V= 5, III = 3.
// 示例 5: 输入: "MCMXCIV" 输出: 1994 解释: M = 1000, CM = 900, XC = 90, IV = 4.
// 提示：题目所给测试用例皆符合罗马数字书写规则，不会出现跨位等情况。
// IC 和 IM 这样的例子并不符合题目要求，49 应该写作 XLIX，999 应该写作 CMXCIX 。
// 关于罗马数字的详尽书写规则，可以参考
func romanToInt(s string) int {
	length := len(s)
	numMap := map[byte]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	num := 0
	for i := 0; i < length; i++ {
		c := s[i]
		tmp := numMap[c]
		if i < length-1 {
			if c == 'I' && (s[i+1] == 'V' || s[i+1] == 'X') {
				tmp = -tmp
			}
			if c == 'X' && (s[i+1] == 'L' || s[i+1] == 'C') {
				tmp = -tmp
			}
			if c == 'C' && (s[i+1] == 'D' || s[i+1] == 'M') {
				tmp = -tmp
			}
		}
		num += tmp
	}
	return num
}

// 14. 最长公共前缀
// 编写一个函数来查找字符串数组中的最长公共前缀。
// 如果不存在公共前缀，返回空字符串 ""。
//
// 示例 1: 输入: ["flower","flow","flight"] 输出: "fl"
// 示例 2: 输入: ["dog","racecar","car"] 输出: "" 解释: 输入不存在公共前缀。
// 说明: 所有输入只包含小写字母 a-z 。
func longestCommonPrefix(strs []string) string {
	length := len(strs)
	if length == 0 {
		return ""
	}
	first := strs[0]
	if length == 1 {
		return first
	}
	i := 0
out:
	for ; i < len(first); i++ {
		c := first[i]
		for j := 1; j < length; j++ {
			if i >= len(strs[j]) {
				break out
			}
			if strs[j][i] != c {
				break out
			}
		}
	}
	return first[0:i]
}

// 28. 实现 strStr()
// 实现 strStr() 函数。
//
// 给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
//
// 示例 1: 输入: haystack = "hello", needle = "ll" 输出: 2
// 示例 2: 输入: haystack = "aaaaa", needle = "bba" 输出: -1
// 说明:
//
// 当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
//
// 对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。
func strStr(haystack string, needle string) int {
	len1, len2 := len(haystack), len(needle)
	if len2 == 0 {
		return 0
	}

	for i := 0; i <= len1-len2; i++ {
		if haystack[i] != needle[0] {
			continue
		}
		flag := true
		for j := 1; j < len2; j++ {
			if haystack[i+j] != needle[j] {
				flag = false
				break
			}
		}
		if flag {
			return i
		}
	}

	return -1
}

// 38. 外观数列
// 给定一个正整数 n（1 ≤ n ≤ 30），输出外观数列的第 n 项。
//
// 注意：整数序列中的每一项将表示为一个字符串。
//
// 「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：
//
// 1.     1
// 2.     11
// 3.     21
// 4.     1211
// 5.     111221
// 第一项是数字 1
//
// 描述前一项，这个数是 1 即 “一个 1 ”，记作 11
//
// 描述前一项，这个数是 11 即 “两个 1 ” ，记作 21
//
// 描述前一项，这个数是 21 即 “一个 2 一个 1 ” ，记作 1211
//
// 描述前一项，这个数是 1211 即 “一个 1 一个 2 两个 1 ” ，记作 111221
//
// 示例 1:
//
// 输入: 1 输出: "1" 解释：这是一个基本样例。
// 示例 2:
//
// 输入: 4
// 输出: "1211"
// 解释：当 n = 3 时，序列是 "21"，其中我们有 "2" 和 "1" 两组，"2" 可以读作 "12"，也就是出现频次 = 1 而 值 = 2；类似 "1" 可以读作 "11"。所以答案是 "12" 和 "11" 组合在一起，也就是 "1211"。
func countAndSay(n int) string {
	if n == 1 {
		return "1"
	}
	if n == 2 {
		return "11"
	}

	str := countAndSay(n - 1)
	length := len(str)
	c, count := str[0], 1
	result := strings.Builder{}
	for i := 1; i <= length; i++ {
		if i == length || c != str[i] {
			result.WriteString(fmt.Sprintf("%s%s", strconv.Itoa(count), string(c)))
			if i < length {
				c = str[i]
				count = 1
			}

		} else {
			count++
		}
	}
	return result.String()
}

// 58. 最后一个单词的长度
// 给定一个仅包含大小写字母和空格 ' ' 的字符串 s，返回其最后一个单词的长度。如果字符串从左向右滚动显示，那么最后一个单词就是最后出现的单词。
//
// 如果不存在最后一个单词，请返回 0 。
//
// 说明：一个单词是指仅由字母组成、不包含任何空格字符的 最大子字符串。
// 示例: 输入: "Hello World" 输出: 5
func lengthOfLastWord(s string) int {
	length, result := len(s), 0
	i := length - 1
	for i >= 0 && s[i] == ' ' {
		i--
	}
	for i >= 0 && s[i] != ' ' {
		i--
		result++
	}
	return result
}

func addBinaryTest() {
	a, b := "1010", "1011"
	result := addBinary(a, b)
	fmt.Println(result)
}

// 67. 二进制求和
// 给你两个二进制字符串，返回它们的和（用二进制表示）。
//
// 输入为 非空 字符串且只包含数字 1 和 0。
// 示例 1: 输入: a = "11", b = "1" 输出: "100"
// 示例 2: 输入: a = "1010", b = "1011" 输出: "10101"
// 提示：
//
// 每个字符串仅由字符 '0' 或 '1' 组成。
// 1 <= a.length, b.length <= 10^4
// 字符串如果不是 "0" ，就都不含前导零。
func addBinary(a string, b string) string {
	len1, len2 := len(a), len(b)
	maxLen := len1
	if len2 > maxLen {
		maxLen = len2
	}
	var last uint8 = 0
	result := ""
	for i := 0; i < maxLen; i++ {
		var num1, num2 uint8 = 0, 0
		if len1-i-1 >= 0 {
			num1 = a[len1-i-1] - '0'
		}
		if len2-i-1 >= 0 {
			num2 = b[len2-i-1] - '0'
		}
		num := num1 + num2 + last
		if num >= 2 {
			num -= 2
			last = 1
		} else {
			last = 0
		}
		result = fmt.Sprintf("%d", num) + result
	}
	if last == 1 {
		result = "1" + result
	}
	return result
}

// 125. 验证回文串
// 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
//
// 说明：本题中，我们将空字符串定义为有效的回文串。
//
// 示例 1: 输入: "A man, a plan, a canal: Panama" 输出: true
// 示例 2: 输入: "race a car" 输出: false
func isPalindrome(s string) bool {
	left, right := 0, len(s)-1

	for left < right {
		for left < right && !isNumber(s[left]) && !isLetter(s[left]) {
			left++
		}
		for left < right && !isNumber(s[right]) && !isLetter(s[right]) {
			right--
		}
		if getLowLetter(s[left]) != getLowLetter(s[right]) {
			return false
		}
		left++
		right--
	}
	return true
}

func isNumber(c uint8) bool {
	return c >= '0' && c <= '9'
}

func isLetter(c uint8) bool {

	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func getLowLetter(c uint8) uint8 {
	if c >= 'A' && c <= 'Z' {
		return c + 32
	}
	return c
}

// 344. 反转字符串
// 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
//
// 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
//
// 你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
//
// 示例 1：
//
// 输入：["h","e","l","l","o"] 输出：["o","l","l","e","h"]
// 示例 2：
//
// 输入：["H","a","n","n","a","h"] 输出：["h","a","n","n","a","H"]
func reverseString(s []byte) {
	left, right := 0, len(s)-1
	for left < right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
}

// 345. 反转字符串中的元音字母
// 编写一个函数，以字符串作为输入，反转该字符串中的元音字母。
//
// 示例 1：
//
// 输入："hello"
// 输出："holle"
// 示例 2：
//
// 输入："leetcode"
// 输出："leotcede"
//
// 提示：
//
// 元音字母不包含字母 "y" 。
func reverseVowels(s string) string {
	arr := []byte(s)
	left, right := 0, len(arr)-1
	for left < right {
		for left < right && !isVowel(arr[left]) {
			left++
		}
		for left < right && !isVowel(arr[right]) {
			right--
		}
		arr[left], arr[right] = arr[right], arr[left]
		left++
		right--

	}
	return string(arr)
}

func isVowel(ch byte) bool {
	switch ch {
	case 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U':
		return true
	default:
		return false
	}
}

// 383. 赎金信
// 给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
//
// (题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)
//
// 注意： 你可以假设两个字符串均只含有小写字母。
//
// canConstruct("a", "b") -> false
// canConstruct("aa", "ab") -> false
// canConstruct("aa", "aab") -> true
func canConstruct(ransomNote string, magazine string) bool {
	letters := [26]int{}
	for _, c := range magazine {
		letters[c-'a']++
	}

	for _, c := range ransomNote {
		if letters[c-'a'] == 0 {
			return false
		}
		letters[c-'a']--
	}

	return true
}

// 387. 字符串中的第一个唯一字符
// 给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。
// 示例：
// s = "leetcode" 返回 0
//
// s = "loveleetcode" 返回 2
//
// 提示：你可以假定该字符串只包含小写字母。
func firstUniqChar(s string) int {
	letters := [26]int{}
	for _, c := range s {
		letters[c-'a']++
	}
	for i, c := range s {
		if letters[c-'a'] == 1 {
			return i
		}
	}
	return -1
}

// 389. 找不同
// 给定两个字符串 s 和 t，它们只包含小写字母。
//
// 字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
//
// 请找出在 t 中被添加的字母。
// 示例 1：输入：s = "abcd", t = "abcde" 输出："e"
// 解释：'e' 是那个被添加的字母。
//
// 示例 2：输入：s = "", t = "y" 输出："y"
//
// 示例 3：输入：s = "a", t = "aa" 输出："a"
//
// 示例 4：输入：s = "ae", t = "aea" 输出："a"
//
// 提示：
//
// 0 <= s.length <= 1000
// t.length == s.length + 1
// s 和 t 只包含小写字母
func findTheDifference(s string, t string) byte {
	var result int32 = 0
	for _, c := range s {
		result ^= c
	}
	for _, c := range t {
		result ^= c
	}
	return byte(result)
}

// 412. Fizz Buzz
// 写一个程序，输出从 1 到 n 数字的字符串表示。
//
// 1. 如果 n 是3的倍数，输出“Fizz”；
//
// 2. 如果 n 是5的倍数，输出“Buzz”；
//
// 3.如果 n 同时是3和5的倍数，输出 “FizzBuzz”。
//
// 示例：
//
// n = 15,
//
// 返回:
// [
//
//	"1",
//	"2",
//	"Fizz",
//	"4",
//	"Buzz",
//	"Fizz",
//	"7",
//	"8",
//	"Fizz",
//	"Buzz",
//	"11",
//	"Fizz",
//	"13",
//	"14",
//	"FizzBuzz"
//
// ]
func fizzBuzz(n int) []string {
	result := make([]string, n)
	for i := 1; i <= n; i++ {
		b1, b2 := i%3 == 0, i%5 == 0
		if b1 && b2 {
			result[i-1] = "FizzBuzz"
		} else if b1 {
			result[i-1] = "Fizz"
		} else if b2 {
			result[i-1] = "Buzz"
		} else {
			result[i-1] = fmt.Sprintf("%d", i)
		}
	}
	return result
}

// 415. 字符串相加
// 给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。
//
// 提示：
//
// num1 和num2 的长度都小于 5100
// num1 和num2 都只包含数字 0-9
// num1 和num2 都不包含任何前导零
// 你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式
func addStrings(num1 string, num2 string) string {

	len1, len2 := len(num1), len(num2)
	if len1 < len2 {
		num1, num2 = num2, num1
		len1, len2 = len2, len1
	}
	index1, index2 := len1-1, len2-1
	index := index1
	str := make([]byte, len1)
	var pre byte
	for index1 >= 0 || index2 >= 0 {
		var c, n2 byte
		n1 := num1[index1] - '0'
		index1--
		if index2 >= 0 {
			n2 = num2[index2] - '0'
			index2--
		}
		c = n1 + n2 + pre
		if c >= 10 {
			c -= 10
			pre = 1
		} else {
			pre = 0
		}
		str[index] = c + '0'
		index--
	}
	result := string(str)
	if pre > 0 {
		return "1" + result
	}
	return result
}

// 434. 字符串中的单词数
// 统计字符串中的单词个数，这里的单词指的是连续的不是空格的字符。
//
// 请注意，你可以假定字符串里不包括任何不可打印的字符。
//
// 示例:
//
// 输入: "Hello, my name is John" 输出: 5
// 解释: 这里的单词是指连续的不是空格的字符，所以 "Hello," 算作 1 个单词。
func countSegments(s string) int {
	index, size := 0, len(s)
	result := 0
	for index < size && s[index] == ' ' {
		index++
	}
	for index < size {
		result++
		for index < size && s[index] != ' ' {
			index++
		}
		for index < size && s[index] == ' ' {
			index++
		}
	}
	return result
}

// 459. 重复的子字符串
// 给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。
//
// 示例 1:
// 输入: "abab" 输出: True
// 解释: 可由子字符串 "ab" 重复两次构成。
//
// 示例 2:
// 输入: "aba" 输出: False
//
// 示例 3:
// 输入: "abcabcabcabc" 输出: True
// 解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
func repeatedSubstringPattern(s string) bool {
	size := len(s)
	if size < 2 {
		return false
	}
	ch := s[0]
	flag := true
	for i := 1; i < size; i++ {
		if s[i] != ch {
			flag = false
		}
	}
	if flag {
		return true
	}
	for i := 2; i <= size>>1; i++ {
		if size%i != 0 {
			continue
		}
		match := true
		for j := i; j < size; j++ {
			if s[j] != s[j-i] {
				match = false
				break
			}
		}
		if match {
			return true
		}

	}
	return false
}

// 482. 密钥格式化
// 有一个密钥字符串 S ，只包含字母，数字以及 '-'（破折号）。其中， N 个 '-' 将字符串分成了 N+1 组。
//
// 给你一个数字 K，请你重新格式化字符串，使每个分组恰好包含 K 个字符。特别地，第一个分组包含的字符个数必须小于等于 K，但至少要包含 1 个字符。两个分组之间需要用 '-'（破折号）隔开，并且将所有的小写字母转换为大写字母。
//
// 给定非空字符串 S 和数字 K，按照上面描述的规则进行格式化。
//
// 示例 1：
// 输入：S = "5F3Z-2e-9-w", K = 4 输出："5F3Z-2E9W"
// 解释：字符串 S 被分成了两个部分，每部分 4 个字符；
//
//	注意，两个额外的破折号需要删掉。
//
// 示例 2：
// 输入：S = "2-5g-3-J", K = 2 输出："2-5G-3J"
// 解释：字符串 S 被分成了 3 个部分，按照前面的规则描述，第一部分的字符可以少于给定的数量，其余部分皆为 2 个字符。
//
// 提示:
// S 的长度可能很长，请按需分配大小。K 为正整数。
// S 只包含字母数字（a-z，A-Z，0-9）以及破折号'-'
// S 非空
func licenseKeyFormatting(s string, k int) string {
	size := len(s)
	clen := size + size/k + 1
	chars := make([]byte, clen)
	index, count := clen-1, 0
	for i := size - 1; i >= 0; i-- {
		c := s[i]
		if count == k {
			chars[index] = '-'
			index--
			count = 0
		}
		if c == '-' {
			continue
		}
		if c >= 'a' && c <= 'z' {
			c = c - 32
		}
		chars[index] = c
		index--
		count++
	}
	for index+1 < clen && chars[index+1] == '-' {
		index++
	}
	return string(chars[index+1:])
}

// 520. 检测大写字母
// 给定一个单词，你需要判断单词的大写使用是否正确。
//
// 我们定义，在以下情况时，单词的大写用法是正确的：
//
// 全部字母都是大写，比如"USA"。
// 单词中所有字母都不是大写，比如"leetcode"。
// 如果单词不只含有一个字母，只有首字母大写， 比如 "Google"。
// 否则，我们定义这个单词没有正确使用大写字母。
//
// 示例 1:
//
// 输入: "USA"
// 输出: True
// 示例 2:
//
// 输入: "FlaG"
// 输出: False
// 注意: 输入是由大写和小写拉丁字母组成的非空单词。
func detectCapitalUse(word string) bool {
	n := len(word)
	if word[0] >= 'a' {
		// 首字母小写
		for i := 0; i < n; i++ {
			if word[i] <= 'Z' {
				return false
			}
		}
	} else {
		// 首字母大写
		if word[n-1] >= 'a' {
			for i := 1; i < n-1; i++ {
				if word[i] <= 'Z' {
					return false
				}
			}
		} else { // 全大写
			for i := 1; i < n-1; i++ {
				if word[i] >= 'a' {
					return false
				}
			}
		}
	}

	return true
}

// 521. 最长特殊序列 Ⅰ
// 给你两个字符串，请你从这两个字符串中找出最长的特殊序列。
//
// 「最长特殊序列」定义如下：该序列为某字符串独有的最长子序列（即不能是其他字符串的子序列）。
// 子序列 可以通过删去字符串中的某些字符实现，但不能改变剩余字符的相对顺序。空序列为所有字符串的子序列，任何字符串为其自身的子序列。
// 输入为两个字符串，输出最长特殊序列的长度。如果不存在，则返回 -1。
//
// 示例 1：
// 输入: "aba", "cdc" 输出: 3
// 解释: 最长特殊序列可为 "aba" (或 "cdc")，两者均为自身的子序列且不是对方的子序列。
//
// 示例 2：
// 输入：a = "aaa", b = "bbb" 输出：3
//
// 示例 3：
// 输入：a = "aaa", b = "aaa" 输出：-1
//
// 提示：
// 两个字符串长度均处于区间 [1 - 100] 。 字符串中的字符仅含有 'a'~'z' 。
func findLUSlength(a string, b string) int {
	size1, size2 := len(a), len(b)

	if size1 > size2 {
		return size1
	}
	if size2 > size1 {
		return size2
	}
	if a == b {
		return -1
	}
	return size1
}

// 541. 反转字符串 II
// 给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。
//
// 如果剩余字符少于 k 个，则将剩余字符全部反转。
// 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
//
// 示例:
//
// 输入: s = "abcdefg", k = 2
// 输出: "bacdfeg"
//
// 提示：
//
// 该字符串只包含小写英文字母。
// 给定字符串的长度和 k 在 [1, 10000] 范围内。
func reverseStr(s string, k int) string {
	chars, size := []byte(s), len(s)

	for i := 0; i < size; i += 2 * k {
		left, right := i, i+k-1
		if right >= size {
			right = size - 1
		}
		for left < right {
			chars[left], chars[right] = chars[right], chars[left]
			left++
			right--
		}
	}

	return string(chars)
}

// 551. 学生出勤记录 I
// 给定一个字符串来代表一个学生的出勤记录，这个记录仅包含以下三个字符：
//
// 'A' : Absent，缺勤
// 'L' : Late，迟到
// 'P' : Present，到场
// 如果一个学生的出勤记录中不超过一个'A'(缺勤)并且不超过两个连续的'L'(迟到),那么这个学生会被奖赏。
//
// 你需要根据这个学生的出勤记录判断他是否会被奖赏。
//
// 示例 1:
// 输入: "PPALLP" 输出: True
//
// 示例 2:
// 输入: "PPALLL" 输出: False
func checkRecord(s string) bool {
	a, l := 0, 0
	for _, c := range s {
		if c == 'A' {
			a++
		}
		if c == 'L' {
			l++
		} else {
			l = 0
		}
		if a > 1 || l > 2 {
			return false
		}
	}

	return true
}

// 557. 反转字符串中的单词 III
// 给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
// 示例：
// 输入："Let's take LeetCode contest" 输出："s'teL ekat edoCteeL tsetnoc"
//
// 提示：
// 在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。
func reverseWordsIII(s string) string {
	size := len(s)
	i := 0
	chars := []byte(s)
	for i < size {
		left, right := i, i
		for right+1 < size && s[right+1] != ' ' {
			right++
		}
		i = right + 2
		for left < right {
			chars[left], chars[right] = chars[right], chars[left]
			left++
			right--
		}
	}

	return string(chars)
}

// 657. 机器人能否返回原点
// 在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，判断这个机器人在完成移动后是否在 (0, 0) 处结束。
//
// 移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），U（上）和 D（下）。如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。
//
// 注意：机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。
//
// 示例 1:
//
// 输入: "UD"
// 输出: true
// 解释：机器人向上移动一次，然后向下移动一次。所有动作都具有相同的幅度，因此它最终回到它开始的原点。因此，我们返回 true。
// 示例 2:
//
// 输入: "LL"
// 输出: false
// 解释：机器人向左移动两次。它最终位于原点的左侧，距原点有两次 “移动” 的距离。我们返回 false，因为它在移动结束时没有返回原点。
func judgeCircle(moves string) bool {
	x, y := 0, 0

	for _, c := range moves {
		switch c {
		// R（右），L（左），U（上）和 D
		case 'R':
			x++
		case 'L':
			x--
		case 'U':
			y++
		case 'D':
			y--

		}
	}

	return x == 0 && y == 0
}

// 680. 验证回文字符串 Ⅱ
// 给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
//
// 示例 1:
// 输入: "aba" 输出: True
//
// 示例 2:
// 输入: "abca" 输出: True
// 解释: 你可以删除c字符。
// 注意:
// 字符串只包含从 a-z 的小写字母。字符串的最大长度是50000。
func validPalindrome(s string) bool {

	var validPalindrome2 = func(s string, left int, right int) bool {
		for left < right {
			if s[left] != s[right] {
				return false
			}
			left++
			right--
		}
		return true
	}
	left, right := 0, len(s)-1
	for left < right {
		if s[left] != s[right] {
			return validPalindrome2(s, left+1, right) || validPalindrome2(s, left, right-1)
		}
		left++
		right--

	}

	return true
}

// 696. 计数二进制子串
// 给定一个字符串 s，计算具有相同数量0和1的非空(连续)子字符串的数量，并且这些子字符串中的所有0和所有1都是组合在一起的。
//
// 重复出现的子串要计算它们出现的次数。
//
// 示例 1 :
// 输入: "00110011" 输出: 6
// 解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。
//
// 请注意，一些重复出现的子串要计算它们出现的次数。
// 另外，“00110011”不是有效的子串，因为所有的0（和1）没有组合在一起。
//
// 示例 2 :
// 输入: "10101" 输出: 4
// 解释: 有4个子串：“10”，“01”，“10”，“01”，它们具有相同数量的连续1和0。
//
// 注意： s.length 在1到50,000之间。
// s 只包含“0”或“1”字符。
func countBinarySubstrings(s string) int {
	lastCount, count := 0, 1
	result := 0
	for i := 1; i < len(s); i++ {
		if s[i] == s[i-1] {
			count++
		} else {
			lastCount = count
			count = 1
		}
		if count <= lastCount {
			result++
		}
	}

	return result
}

// 709. 转换成小写字母
// 实现函数 ToLowerCase()，该函数接收一个字符串参数 str，并将该字符串中的大写字母转换成小写字母，之后返回新的字符串。
//
// 示例 1：
// 输入: "Hello" 输出: "hello"
//
// 示例 2：
// 输入: "here" 输出: "here"
//
// 示例 3：
// 输入: "LOVELY" 输出: "lovely"
func toLowerCase(str string) string {
	chars := []byte(str)
	for i, c := range chars {
		if c >= 'A' && c <= 'Z' {
			chars[i] = c + 32
		}
	}

	return string(chars)
}

// 1370. 上升下降字符串
// 给你一个字符串 s ，请你根据下面的算法重新构造字符串：
//
// 从 s 中选出 最小 的字符，将它 接在 结果字符串的后面。
// 从 s 剩余字符中选出 最小 的字符，且该字符比上一个添加的字符大，将它 接在 结果字符串后面。
// 重复步骤 2 ，直到你没法从 s 中选择字符。
// 从 s 中选出 最大 的字符，将它 接在 结果字符串的后面。
// 从 s 剩余字符中选出 最大 的字符，且该字符比上一个添加的字符小，将它 接在 结果字符串后面。
// 重复步骤 5 ，直到你没法从 s 中选择字符。
// 重复步骤 1 到 6 ，直到 s 中所有字符都已经被选过。
// 在任何一步中，如果最小或者最大字符不止一个 ，你可以选择其中任意一个，并将其添加到结果字符串。
//
// 请你返回将 s 中字符重新排序后的 结果字符串 。
//
// 示例 1：
// 输入：s = "aaaabbbbcccc" 输出："abccbaabccba"
// 解释：第一轮的步骤 1，2，3 后，结果字符串为 result = "abc"
// 第一轮的步骤 4，5，6 后，结果字符串为 result = "abccba"
// 第一轮结束，现在 s = "aabbcc" ，我们再次回到步骤 1
// 第二轮的步骤 1，2，3 后，结果字符串为 result = "abccbaabc"
// 第二轮的步骤 4，5，6 后，结果字符串为 result = "abccbaabccba"
//
// 示例 2：
// 输入：s = "rat" 输出："art"
// 解释：单词 "rat" 在上述算法重排序以后变成 "art"
//
// 示例 3：
// 输入：s = "leetcode" 输出："cdelotee"
//
// 示例 4：
// 输入：s = "ggggggg" 输出："ggggggg"
//
// 示例 5：
// 输入：s = "spo" 输出："ops"
//
// 提示： 1 <= s.length <= 500
// s 只包含小写英文字母。
func sortString(s string) string {

	idx, size := 0, len(s)
	chars := make([]byte, size)
	letters := [26]int{}
	for _, c := range s {
		letters[c-'a']++
	}
	for idx < size {
		for i := 0; i < 26; i++ {
			if letters[i] > 0 {
				chars[idx] = byte('a' + i)
				idx++
				letters[i]--
			}
		}
		for i := 25; i >= 0; i-- {
			if letters[i] > 0 {
				chars[idx] = byte('a' + i)
				idx++
				letters[i]--
			}
		}
	}
	return string(chars)
}

func removeDuplicates2(S string) string {

	// 使用栈
	stack := list.New()
	for i, _ := range S {
		if stack.Len() != 0 && stack.Back().Value.(byte) == S[i] {
			back := stack.Back()
			stack.Remove(back)
			continue
		}
		stack.PushBack(S[i])
	}
	chars := make([]byte, stack.Len())
	size1 := stack.Len()
	for i := 0; i < size1; i++ {
		chars[i] = stack.Front().Value.(byte)
		front := stack.Front()
		stack.Remove(front)
	}
	return string(chars)
}

// 87. 扰乱字符串
// 使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：
// 如果字符串的长度为 1 ，算法停止
// 如果字符串的长度 > 1 ，执行下述步骤：
// 在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。
// 随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。
// 在 x 和 y 这两个子字符串上继续从步骤 1 开始递归执行此算法。
// 给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：s1 = "great", s2 = "rgeat" 输出：true
//
// 解释：s1 上可能发生的一种情形是：
// "great" --> "gr/eat" // 在一个随机下标处分割得到两个子字符串
// "gr/eat" --> "gr/eat" // 随机决定：「保持这两个子字符串的顺序不变」
// "gr/eat" --> "g/r / e/at" // 在子字符串上递归执行此算法。两个子字符串分别在随机下标处进行一轮分割
// "g/r / e/at" --> "r/g / e/at" // 随机决定：第一组「交换两个子字符串」，第二组「保持这两个子字符串的顺序不变」
// "r/g / e/at" --> "r/g / e/ a/t" // 继续递归执行此算法，将 "at" 分割得到 "a/t"
// "r/g / e/ a/t" --> "r/g / e/ a/t" // 随机决定：「保持这两个子字符串的顺序不变」
// 算法终止，结果字符串和 s2 相同，都是 "rgeat"
// 这是一种能够扰乱 s1 得到 s2 的情形，可以认为 s2 是 s1 的扰乱字符串，返回 true
//
// 示例 2：
// 输入：s1 = "abcde", s2 = "caebd" 输出：false
//
// 示例 3：
// 输入：s1 = "a", s2 = "a" 输出：true
//
// 提示：
// s1.length == s2.length
// 1 <= s1.length <= 30
// s1 和 s2 由小写英文字母组成
func isScramble(s1 string, s2 string) bool {
	if s1 == s2 {
		return true
	}
	size1, size2 := len(s1), len(s2)
	if size1 != size2 {
		return false
	}
	n := size1
	// dp[i][j][k] s1的i位 s2 的 j 位 长度k 是否 扰乱字符串
	dp := make([][][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([][]bool, n)
		for j := 0; j < n; j++ {
			dp[i][j] = make([]bool, n+1)
			dp[i][j][1] = s1[i] == s2[j]
		}
	}

	for l := 2; l <= n; l++ {
		for i := 0; i+l <= n; i++ {
			for j := 0; j+l <= n; j++ {
				// 枚举划分位置
				for k := 1; k < l; k++ {
					// 第一种情况：S1 -> T1, S2 -> T2
					if dp[i][j][k] && dp[i+k][j+k][l-k] {
						dp[i][j][l] = true
						break
					}
					// 第二种情况：S1 -> T2, S2 -> T1
					// S1 起点 i，T2 起点 j + 前面那段长度 len-k ，S2 起点 i + 前面长度k
					if dp[i][j+l-k][k] && dp[i+k][j][l-k] {
						dp[i][j][l] = true
						break
					}
				}

			}
		}
	}

	return dp[0][0][n]
}

// 3. 无重复字符的最长子串
//
// 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
//
// 示例 1:
// 输入: s = "abcabcbb" 输出: 3
// 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
//
// 示例 2:
// 输入: s = "bbbbb" 输出: 1
// 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
//
// 示例 3:
// 输入: s = "pwwkew" 输出: 3
// 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
//
//	请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
//
// 示例 4:
// 输入: s = "" 输出: 0
//
// 提示：
// 0 <= s.length <= 5 * 104
// s 由英文字母、数字、符号和空格组成
func lengthOfLongestSubstring(s string) int {
	size := len(s)
	if size <= 1 {
		return size
	}
	result := 1
	letters := make([]int, 26)
	// 滑动窗口
	left := 0

	for i, c := range s {
		idx := c - 'a'
		letters[idx]++
		for letters[idx] > 1 {
			leftNum := s[left] - 'a'
			letters[leftNum]--
			left++
		}
		l := i - left + 1
		result = max(result, l)
	}

	return result
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 5. 最长回文子串
// 给你一个字符串 s，找到 s 中最长的回文子串。
//
// 示例 1：
// 输入：s = "babad" 输出："bab"
// 解释："aba" 同样是符合题意的答案。
//
// 示例 2：
// 输入：s = "cbbd" 输出："bb"
//
// 示例 3：
// 输入：s = "a" 输出："a"
//
// 示例 4：
// 输入：s = "ac" 输出："a"
//
// 提示：
// 1 <= s.length <= 1000
// s 仅由数字和英文字母（大写和/或小写）组成
func longestPalindrome(s string) string {
	size := len(s)
	result := ""
	for i := 0; i < size; i++ {
		left, right := i, i

		for right+1 < size && s[right+1] == s[right] {
			right++
		}
		i = right
		for left-1 >= 0 && right+1 < size && s[left-1] == s[right+1] {
			left--
			right++
		}
		l := right - left + 1

		if l > len(result) {
			result = s[left : right+1]
			fmt.Printf("left:%d right:%d result:%s \n ", left, right, result)
		}

	}
	return result
}

// 6. Z 字形变换
//
// 将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
// 比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
//
// P   A   H   N
// A P L S I I G
// Y   I   R
// 之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
//
// 请你实现这个将字符串进行指定行数变换的函数：
// string convert(string s, int numRows);
//
// 示例 1：
// 输入：s = "PAYPALISHIRING", numRows = 3 输出："PAHNAPLSIIGYIR"
//
// 示例 2：
// 输入：s = "PAYPALISHIRING", numRows = 4 输出："PINALSIGYAHRPI"
// 解释：
// P     I    N
// A   L S  I G
// Y A   H R
// P     I
//
// 示例 3：
// 输入：s = "A", numRows = 1 输出："A"
//
// 提示：
//
// 1 <= s.length <= 1000
// s 由英文字母（小写和大写）、',' 和 '.' 组成
// 1 <= numRows <= 1000
func convert(s string, numRows int) string {
	size, idx := len(s), 0
	if numRows < 2 {
		return s
	}
	chars := make([]byte, size)
	zLen := numRows*2 - 2
	for row := 0; row < numRows; row++ {
		fIdx, sIdx := row, zLen-row
		for fIdx < size {
			chars[idx] = s[fIdx]
			idx++
			fIdx += zLen
			if row > 0 && row < numRows-1 && sIdx < size {
				chars[idx] = s[sIdx]
				idx++
				sIdx += zLen
			}
		}

	}
	fmt.Printf("idx:%d \n", idx)

	return string(chars)
}

// 8. 字符串转换整数 (atoi)
//
// 请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
// 函数 myAtoi(string s) 的算法如下：
//
// 读入字符串并丢弃无用的前导空格
// 检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
// 读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
// 将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
// 如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
// 返回整数作为最终结果。
// 注意：
//
// 本题中的空白字符只包括空格字符 ' ' 。
// 除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
//
// 示例 1：
// 输入：s = "42" 输出：42
// 解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
// 第 1 步："42"（当前没有读入字符，因为没有前导空格）
//
//	^
//
// 第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
//
//	^
//
// 第 3 步："42"（读入 "42"）
//
//	^
//
// 解析得到整数 42 。
// 由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
//
// 示例 2：
// 输入：s = "   -42" 输出：-42
// 解释：
// 第 1 步："   -42"（读入前导空格，但忽视掉）
//
//	^
//
// 第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
//
//	^
//
// 第 3 步："   -42"（读入 "42"）
//
//	^
//
// 解析得到整数 -42 。
// 由于 "-42" 在范围 [-231, 231 - 1] 内，最终结果为 -42 。
//
// 示例 3：
// 输入：s = "4193 with words" 输出：4193
// 解释：
// 第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
//
//	^
//
// 第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
//
//	^
//
// 第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
//
//	^
//
// 解析得到整数 4193 。
// 由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。
//
// 示例 4：
// 输入：s = "words and 987" 输出：0
// 解释：
// 第 1 步："words and 987"（当前没有读入字符，因为没有前导空格）
//
//	^
//
// 第 2 步："words and 987"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
//
//	^
//
// 第 3 步："words and 987"（由于当前字符 'w' 不是一个数字，所以读入停止）
//
//	^
//
// 解析得到整数 0 ，因为没有读入任何数字。
// 由于 0 在范围 [-231, 231 - 1] 内，最终结果为 0 。
//
// 示例 5：
// 输入：s = "-91283472332" 输出：-2147483648
// 解释：
// 第 1 步："-91283472332"（当前没有读入字符，因为没有前导空格）
//
//	^
//
// 第 2 步："-91283472332"（读入 '-' 字符，所以结果应该是负数）
//
//	^
//
// 第 3 步："-91283472332"（读入 "91283472332"）
//
//	^
//
// 解析得到整数 -91283472332 。
// 由于 -91283472332 小于范围 [-231, 231 - 1] 的下界，最终结果被截断为 -231 = -2147483648 。
//
// 提示：
// 0 <= s.length <= 200
// s 由英文字母（大写和小写）、数字（0-9）、' '、'+'、'-' 和 '.' 组成
func myAtoi(s string) int {

	result, start, size := 0, 0, len(s)
	if size == 0 {
		return 0
	}
	// 负数
	negative := false

	for start < size && s[start] == ' ' {
		start++
	}
	if start == size {
		return 0
	}
	if s[start] == '+' {
		start++
	} else if s[start] == '-' {
		start++
		negative = true
	}
	if start == size || s[start] < '0' || s[start] > '9' {
		return 0
	}
	MIN, MAX := -(1 << 31), (1<<31)-1

	tmpMin, tmpMax := MIN/10, MAX/10
	for i := start; i < size; i++ {
		if s[i] < '0' || s[i] > '9' {
			break
		}
		num := int(s[i] - '0')
		if negative {
			num = -num
		}
		if result > tmpMax || (result == tmpMax && num >= 7) {
			return MAX
		}
		if result < tmpMin || (result == tmpMin && num <= -8) {
			return MIN
		}

		result = result*10 + num
	}

	return result
}

// 12. 整数转罗马数字
// 罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。
//
// 字符          数值
// I             1
// V             5
// X             10
// L             50
// C             100
// D             500
// M             1000
// 例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。
//
// 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：
//
// I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
// X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。
// C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
// 给你一个整数，将其转为罗马数字。
//
// 示例 1:
// 输入: num = 3 输出: "III"
//
// 示例 2:
// 输入: num = 4 输出: "IV"
//
// 示例 3:
// 输入: num = 9 输出: "IX"
//
// 示例 4:
// 输入: num = 58 输出: "LVIII"
// 解释: L = 50, V = 5, III = 3.
//
// 示例 5:
// 输入: num = 1994 输出: "MCMXCIV"
// 解释: M = 1000, CM = 900, XC = 90, IV = 4.
//
// 提示：
// 1 <= num <= 3999
func intToRoman(num int) string {
	nums := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
	romans := []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
	var builder strings.Builder
	for i := 0; i < len(nums); i++ {
		for num >= nums[i] {
			num -= nums[i]
			builder.WriteString(romans[i])
		}
	}
	return builder.String()
}

// 30. 串联所有单词的子串
// 给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
// 注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。
//
// 示例 1：
// 输入：s = "barfoothefoobarman", words = ["foo","bar"] 输出：[0,9]
// 解释： 从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
// 输出的顺序不重要, [9,0] 也是有效答案。
//
// 示例 2：
// 输入：s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"] 输出：[]
//
// 示例 3：
// 输入：s = "barfoofoobarthefoobarman", words = ["bar","foo","the"] 输出：[6,9,12]
//
// 提示：
// 1 <= s.length <= 104
// s 由小写英文字母组成
// 1 <= words.length <= 5000
// 1 <= words[i].length <= 30
// words[i] 由小写英文字母组成
func findSubstring(s string, words []string) []int {
	result := make([]int, 0)
	size, wordLen := len(s), len(words[0])
	totalLen := wordLen * len(words)
	if size < totalLen {
		return result
	}
	allWords := make(map[string]int)
	for _, word := range words {
		allWords[word]++
	}

	compare := func(map1, map2 map[string]int) bool {

		for k, v := range map1 {
			v2 := map2[k]
			if v != v2 {
				return false
			}
		}

		return true
	}

	for k := 0; k < wordLen; k++ {
		wordMap := make(map[string]int)
		curLen := 0
		for i := k; i <= size-wordLen; i += wordLen {
			word := s[i : i+wordLen]
			if _, ok := allWords[word]; ok {
				wordMap[word]++
				curLen++
			} else {
				wordMap = make(map[string]int)
				curLen = 0
				continue
			}
			start := i - totalLen + wordLen
			if start < 0 {
				continue
			}
			if curLen < len(words) {
				continue
			}

			if compare(allWords, wordMap) {
				result = append(result, start)
			}
			startWord := s[start : start+wordLen]
			wordMap[startWord]--
		}
	}

	return result
}

// 43. 字符串相乘
// 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
//
// 示例 1:
// 输入: num1 = "2", num2 = "3" 输出: "6"
//
// 示例 2:
// 输入: num1 = "123", num2 = "456" 输出: "56088"
//
// 说明：
// num1 和 num2 的长度小于110。
// num1 和 num2 只包含数字 0-9。
// num1 和 num2 均不以零开头，除非是数字 0 本身。
// 不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理。
func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	size1, size2 := len(num1), len(num2)
	nums := make([]int, size1+size2)

	for i := size1 - 1; i >= 0; i-- {
		n1 := int(num1[i] - '0')
		for j := size2 - 1; j >= 0; j-- {
			n2 := int(num2[j] - '0')
			sum := nums[i+j+1] + n1*n2
			nums[i+j+1] = sum % 10
			nums[i+j] += sum / 10
		}
	}
	var builder strings.Builder
	for i := 0; i < size1+size2; i++ {
		if i == 0 && nums[i] == 0 {
			continue
		}
		builder.WriteString(strconv.Itoa(nums[i]))
	}
	return builder.String()
}

// 44. 通配符匹配
// 给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
//
// '?' 可以匹配任何单个字符。
// '*' 可以匹配任意字符串（包括空字符串）。
// 两个字符串完全匹配才算匹配成功。
//
// 说明:
// s 可能为空，且只包含从 a-z 的小写字母。
// p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
//
// 示例 1:
// 输入: s = "aa" p = "a"  输出: false
// 解释: "a" 无法匹配 "aa" 整个字符串。
//
// 示例 2:
// 输入: s = "aa" p = "*" 输出: true
// 解释: '*' 可以匹配任意字符串。
//
// 示例 3:
// 输入: s = "cb" p = "?a" 输出: false
// 解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。
//
// 示例 4:
// 输入: s = "adceb" p = "*a*b" 输出: true
// 解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".
//
// 示例 5:
// 输入: s = "acdcb" p = "a*c?b" 输出: false
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	match := make([][]bool, m+1)
	for i := 0; i <= m; i++ {
		match[i] = make([]bool, n+1)
	}
	match[0][0] = true
	for j := 0; j < n; j++ {
		if p[j] == '*' {
			match[0][j+1] = match[0][j]
		}
	}

	for i := 0; i < m; i++ {
		c1 := s[i]
		for j := 0; j < n; j++ {
			c2 := p[j]
			same := c1 == c2 || c2 == '?'
			if same {
				match[i+1][j+1] = match[i][j]
			} else if c2 == '*' {
				match[i+1][j+1] = match[i][j+1] || match[i+1][j]
			}
		}
	}

	return match[m][n]
}

// 65. 有效数字
// 有效数字（按顺序）可以分成以下几个部分：
//
// 一个 小数 或者 整数
// （可选）一个 'e' 或 'E' ，后面跟着一个 整数
// 小数（按顺序）可以分成以下几个部分：
//
// （ 可选）一个符号字符（'+' 或 '-'）
//
//	下述格式之一：
//	至少一位数字，后面跟着一个点 '.'
//	至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
//	一个点 '.' ，后面跟着至少一位数字
//	整数（按顺序）可以分成以下几个部分：
//
// （可选）一个符号字符（'+' 或 '-'）
//
//	 至少一位数字
//	 部分有效数字列举如下：
//
//	 ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
//	部分无效数字列举如下：
//
//	["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]
//	给你一个字符串 s ，如果 s 是一个 有效数字 ，请返回 true 。
//
// 示例 1：
// 输入：s = "0" 输出：true
//
// 示例 2：
// 输入：s = "e" 输出：false
//
// 示例 3：
// 输入：s = "." 输出：false
//
// 示例 4：
// 输入：s = ".1" 输出：true
//
// 提示：
// 1 <= s.length <= 20
// s 仅含英文字母（大写和小写），数字（0-9），加号 '+' ，减号 '-' ，或者点 '.' 。
func isNumber2(s string) bool {
	// 初始状态 符号 整数 左无整数小数点 小数 指数e 指数符号 有效数字
	const S0, S1, S2, S3, S4, S5, S6, S7 = 1, 2, 4, 8, 16, 32, 64, 128

	state := 1
	for _, c := range s {
		switch state {
		case S0:
			{
				if c == '+' || c == '-' {
					state = S1
				} else if unicode.IsDigit(c) {
					state = S2
				} else if c == '.' {
					state = S3
				} else {
					return false
				}
			}
		case S1:
			{
				if unicode.IsDigit(c) {
					state = S2
				} else if c == '.' {
					state = S3
				} else {
					return false
				}

			}
		case S2:
			{
				if c == '.' {
					state = S4
				} else if c == 'e' || c == 'E' {
					state = S5
				} else if !unicode.IsDigit(c) {
					return false
				}

			}
		case S3:
			{
				if unicode.IsDigit(c) {
					state = S4
				} else {
					return false
				}

			}
		case S4:
			{
				if c == 'e' || c == 'E' {
					state = S5
				} else if !unicode.IsDigit(c) {
					return false
				}

			}
		case S5:
			{
				if c == '+' || c == '-' {
					state = S6
				} else if unicode.IsDigit(c) {
					state = S7
				} else {
					return false
				}
			}
		case S6:
			{
				if unicode.IsDigit(c) {
					state = S7
				} else {
					return false
				}
			}
		case S7:
			{
				if !unicode.IsDigit(c) {
					return false
				}

			}
		}

	}

	return state == S2 || state == S4 || state == S7
}

// 68. 文本左右对齐
// 给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
// 你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
// 要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。
// 文本的最后一行应为左对齐，且单词之间不插入额外的空格。
//
// 说明:
// 单词是指由非空格字符组成的字符序列。
// 每个单词的长度大于 0，小于等于 maxWidth。
// 输入单词数组 words 至少包含一个单词。
//
// 示例 1:
// 输入: words = ["This", "is", "an", "example", "of", "text", "justification."] maxWidth = 16
// 输出:
// [
//
//	"This    is    an",
//	"example  of text",
//	"justification.  "
//
// ]
// 示例 2:
// 输入: words = ["What","must","be","acknowledgment","shall","be"] maxWidth = 16
// 输出:
// [
//
//	"What   must   be",
//	"acknowledgment  ",
//	"shall be        "
//
// ]
// 解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
//
//	因为最后一行应为左对齐，而不是左右两端对齐。
//	第二行同样为左对齐，这是因为这行只包含一个单词。
//
// 示例 3:
// 输入: words = ["Science","is","what","we","understand","well","enough","to","explain",
//
//	"to","a","computer.","Art","is","everything","else","we","do"] maxWidth = 20
//
// 输出:
// [
//
//	"Science  is  what we",
//	"understand      well",
//	"enough to explain to",
//	"a  computer.  Art is",
//	"everything  else  we",
//	"do                  "
//
// ]
func fullJustify(words []string, maxWidth int) []string {
	result := make([]string, 0)
	size, cnt := 0, 0

	getRowWord := func(cnt, end, blankCnt int) string {
		var builder strings.Builder
		avgBlankCnt, leftBlankCnt := blankCnt, 0

		if cnt > 1 {
			avgBlankCnt = blankCnt / (cnt - 1)
			leftBlankCnt = blankCnt % (cnt - 1)
		}
		var blankBuilder strings.Builder
		for i := 0; i < avgBlankCnt; i++ {
			blankBuilder.WriteByte(' ')
		}

		for i := end - cnt + 1; i <= end; i++ {
			builder.WriteString(words[i])
			if i < end {
				builder.WriteString(blankBuilder.String())
				if leftBlankCnt > 0 {
					builder.WriteByte(' ')
					leftBlankCnt--
				}
			}
		}
		if cnt == 1 {
			builder.WriteString(blankBuilder.String())
		}
		return builder.String()
	}
	idx := 0
	for idx < len(words) {
		word := words[idx]
		if size+len(word) <= maxWidth-cnt {
			size += len(word)
			cnt++
			idx++
		} else {
			rowWord := getRowWord(cnt, idx-1, maxWidth-size)
			result = append(result, rowWord)
			size = 0
			cnt = 0
		}
	}
	var lastRow strings.Builder
	// 处理最后一行
	if size > 0 {
		idx = len(words) - 1

		for i := idx - cnt + 1; i <= idx; i++ {
			lastRow.WriteString(words[i])
			if i < idx {
				// 最后一行一个空格
				lastRow.WriteByte(' ')
			}
		}
	}
	leftCnt := maxWidth - lastRow.Len()
	for i := 0; i < leftCnt; i++ {
		lastRow.WriteByte(' ')
	}
	result = append(result, lastRow.String())
	return result
}

// 71. 简化路径
// 给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为更加简洁的规范路径。
//
// 在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，'//'）都被视为单个斜杠 '/' 。 对于此问题，任何其他格式的点（例如，'...'）均被视为文件/目录名称。
//
// 请注意，返回的 规范路径 必须遵循下述格式：
//
// 始终以斜杠 '/' 开头。
// 两个目录名之间必须只有一个斜杠 '/' 。
// 最后一个目录名（如果存在）不能 以 '/' 结尾。
// 此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
// 返回简化后得到的 规范路径 。
//
// 示例 1：
// 输入：path = "/home/" 输出："/home"
// 解释：注意，最后一个目录名后面没有斜杠。
//
// 示例 2：
// 输入：path = "/../" 输出："/"
// 解释：从根目录向上一级是不可行的，因为根目录是你可以到达的最高级。
//
// 示例 3：
// 输入：path = "/home//foo/"  输出："/home/foo"
// 解释：在规范路径中，多个连续斜杠需要用一个斜杠替换。
//
// 示例 4：
// 输入：path = "/a/./b/../../c/" 输出："/c"
//
// 提示：
// 1 <= path.length <= 3000
// path 由英文字母，数字，'.'，'/' 或 '_' 组成。
// path 是一个有效的 Unix 风格绝对路径。
func simplifyPath(path string) string {
	size := len(path)
	if size <= 1 {
		return "/"
	}

	stack := list.New()
	for i := 0; i < size; i++ {
		start := i
		for i < size && path[i] != '/' {
			i++
		}
		tmp := path[start:i]
		if i == start {
			continue
		}
		fmt.Println(tmp)
		if tmp == ".." {
			if stack.Len() > 0 {
				back := stack.Back()
				stack.Remove(back)
			}
		} else if tmp != "." {
			stack.PushBack(tmp)
		}
	}
	var builder strings.Builder
	for stack.Len() > 0 {
		front := stack.Front()

		builder.WriteString("/")
		builder.WriteString(front.Value.(string))
		stack.Remove(front)
	}
	if builder.Len() == 0 {
		return "/"
	}
	return builder.String()
}

// 76. 最小覆盖子串
// 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
//
// 注意：
// 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
// 如果 s 中存在这样的子串，我们保证它是唯一的答案。
//
// 示例 1：
// 输入：s = "ADOBECODEBANC", t = "ABC" 输出："BANC"
//
// 示例 2：
// 输入：s = "a", t = "a" 输出："a"
//
// 示例 3:
// 输入: s = "a", t = "aa" 输出: ""
// 解释: t 中两个字符 'a' 均应包含在 s 的子串中，
// 因此没有符合条件的子字符串，返回空字符串。
//
// 提示：
// 1 <= s.length, t.length <= 105
// s 和 t 由英文字母组成
//
// 进阶：你能设计一个在 o(n) 时间内解决此问题的算法吗？
func minWindow(s string, t string) string {
	size1, size2 := len(s), len(t)
	if size1 == 0 || size2 == 0 {
		return ""
	}
	if size1 < size2 {
		return ""
	}

	sLetters, tLetters := make([]int, 52), make([]int, 52)

	getLetterNum := func(c byte) int {
		if c >= 'a' {
			return int(c - 'a')
		}
		return int(c-'A') + 26
	}

	contain := func() bool {
		for i := 0; i < 52; i++ {
			if sLetters[i] < tLetters[i] {
				return false
			}
		}
		return true
	}

	for i := range t {
		tLetters[getLetterNum(t[i])]++
	}

	for i := 0; i < size2-1; i++ {
		sLetters[getLetterNum(s[i])]++
	}

	left, start, minLen := 0, 0, size1+1

	for i := size2 - 1; i < size1; i++ {
		sLetters[getLetterNum(s[i])]++
		for left < size1 && contain() {
			fmt.Println(sLetters)
			fmt.Println(tLetters)
			if i-left+1 < minLen {
				minLen = i - left + 1
				start = left
			}
			sLetters[getLetterNum(s[left])]--
			left++
		}
	}

	if minLen == size1+1 {
		return ""
	}
	fmt.Printf("start:%d len:%d\n", start, minLen)

	return s[start : start+minLen]
}

// 126. 单词接龙 II
// 按字典 wordList 完成从单词 beginWord 到单词 endWord 转化，一个表示此过程的 转换序列 是形式上像 beginWord -> s1 -> s2 -> ... -> sk 这样的单词序列，并满足：
//
// 每对相邻的单词之间仅有单个字母不同。
// 转换过程中的每个单词 si（1 <= i <= k）必须是字典 wordList 中的单词。注意，beginWord 不必是字典 wordList 中的单词。
// sk == endWord
// 给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，如果不存在这样的转换序列，返回一个空列表。每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。
//
// 示例 1：
// 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
// 输出：[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
// 解释：存在 2 种最短的转换序列：
// "hit" -> "hot" -> "dot" -> "dog" -> "cog"
// "hit" -> "hot" -> "lot" -> "log" -> "cog"
//
// 示例 2：
// 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
// 输出：[]
// 解释：endWord "cog" 不在字典 wordList 中，所以不存在符合要求的转换序列。
//
// 提示：
// 1 <= beginWord.length <= 7
// endWord.length == beginWord.length
// 1 <= wordList.length <= 5000
// wordList[i].length == beginWord.length
// beginWord、endWord 和 wordList[i] 由小写英文字母组成
// beginWord != endWord
// wordList 中的所有单词 互不相同
func findLadders(beginWord string, endWord string, wordList []string) [][]string {

	result := make([][]string, 0)

	allComboDict := make(map[string][]string)
	var builder strings.Builder
	notEndWord := true
	for _, word := range wordList {
		if word == endWord {
			notEndWord = false
		}
		for i := range word {
			builder.Reset()
			builder.WriteString(word[0:i])
			builder.WriteString("*")
			builder.WriteString(word[i+1:])
			key := builder.String()
			if _, ok := allComboDict[key]; !ok {
				allComboDict[key] = make([]string, 0)
			}
			allComboDict[key] = append(allComboDict[key], word)
		}
	}
	if notEndWord {
		return result
	}

	// 广度优先遍历
	queue := list.New()
	queue.PushBack(beginWord)
	visited := make(map[string]bool)
	visited[beginWord] = true
	step := 1
	stepMap := make(map[string]int)
	// 单词列表
	from := make(map[string][]string)
	found := false
	for queue.Len() > 0 {
		size := queue.Len()

		for i := 0; i < size; i++ {

			front := queue.Front()
			queue.Remove(front)
			word := front.Value.(string)

			for j := range word {
				builder.Reset()
				builder.WriteString(word[0:j])
				builder.WriteString("*")
				builder.WriteString(word[j+1:])
				key := builder.String()
				if nextWords, ok := allComboDict[key]; ok {
					for _, nextWord := range nextWords {
						if stepMap[nextWord] == step {
							from[nextWord] = append(from[nextWord], word)
						}
						if visited[nextWord] {
							continue
						}
						visited[nextWord] = true
						queue.PushBack(nextWord)
						if nextWord == endWord {
							found = true
						}
						if _, fromOk := from[nextWord]; !fromOk {
							from[nextWord] = make([]string, 0)
						}
						from[nextWord] = append(from[nextWord], word)
						stepMap[nextWord] = step
					}
				}
			}

		}
		step++
		// 已找到最短的
		if found {
			break
		}
	}
	fmt.Println(from)

	var dfs func(curWord string, idx int, path []string)

	dfs = func(curWord string, idx int, path []string) {
		if idx == 0 {
			tmpPath := make([]string, step)
			copy(tmpPath, path)
			tmpPath[0] = beginWord
			result = append(result, tmpPath)
			return
		}
		for _, proWord := range from[curWord] {
			path[idx] = proWord
			dfs(proWord, idx-1, path)
		}
	}

	if found {
		path := make([]string, step)
		path[step-1] = endWord
		dfs(endWord, step-2, path)
	}

	return result
}

// 127. 单词接龙
// 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列：
//
// 序列中第一个单词是 beginWord 。
// 序列中最后一个单词是 endWord 。
// 每次转换只能改变一个字母。
// 转换过程中的中间单词必须是字典 wordList 中的单词。
// 给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。
//
// 示例 1：
// 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"] 输出：5
// 解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
//
// 示例 2：
// 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"] 输出：0
// 解释：endWord "cog" 不在字典中，所以无法进行转换。
//
// 提示：
// 1 <= beginWord.length <= 10
// endWord.length == beginWord.length
// 1 <= wordList.length <= 5000
// wordList[i].length == beginWord.length
// beginWord、endWord 和 wordList[i] 由小写英文字母组成
// beginWord != endWord
// wordList 中的所有字符串 互不相同
func ladderLength(beginWord string, endWord string, wordList []string) int {
	wordMap := make(map[string]bool)
	for _, word := range wordList {
		wordMap[word] = true
	}
	if _, ok := wordMap[endWord]; !ok {
		return 0
	}
	allComboDict := make(map[string][]string)
	for _, word := range wordList {
		for i := range word {
			key := word[0:i] + "*" + word[i+1:]
			if _, ok := allComboDict[key]; !ok {
				allComboDict[key] = make([]string, 0)
			}
			allComboDict[key] = append(allComboDict[key], word)
		}
	}
	// 广度优先遍历
	queue := list.New()
	queue.PushBack(beginWord)

	visited := make(map[string]bool)
	visited[beginWord] = true
	step := 1
	for queue.Len() > 0 {
		size := queue.Len()
		for i := 0; i < size; i++ {

			front := queue.Front()
			queue.Remove(front)
			word := front.Value.(string)
			if word == endWord {
				return step
			}
			for j := range word {
				key := word[0:j] + "*" + word[j+1:]
				if nextWords, ok := allComboDict[key]; ok {
					for _, nextWord := range nextWords {
						if visited[nextWord] == false {
							queue.PushBack(nextWord)
							visited[nextWord] = true
						}
					}
				}
			}

		}
		step++
	}

	return 0
}

// 151. 翻转字符串里的单词
// 给你一个字符串 s ，逐个翻转字符串中的所有 单词 。
// 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
// 请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。
// 说明：
// 输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
// 翻转后单词间应当仅用一个空格分隔。
// 翻转后的字符串中不应包含额外的空格。
//
// 示例 1：
// 输入：s = "the sky is blue"
// 输出："blue is sky the"
//
// 示例 2：
// 输入：s = "  hello world  "
// 输出："world hello"
// 解释：输入字符串可以在前面或者后面包含多余的空格，但是翻转后的字符不能包括。
//
// 示例 3：
// 输入：s = "a good   example"
// 输出："example good a"
// 解释：如果两个单词间有多余的空格，将翻转后单词间的空格减少到只含一个。
//
// 示例 4：
// 输入：s = "  Bob    Loves  Alice   "
// 输出："Alice Loves Bob"
//
// 示例 5：
// 输入：s = "Alice does not even like bob"
// 输出："bob like even not does Alice"
//
// 提示：
// 1 <= s.length <= 104
// s 包含英文大小写字母、数字和空格 ' '
// s 中 至少存在一个 单词
//
// 进阶：
// 请尝试使用 O(1) 额外空间复杂度的原地解法。
func reverseWords(s string) string {
	strs := strings.Split(s, " ")

	var builder strings.Builder
	size := len(strs)
	for i := size - 1; i >= 0; i-- {
		if len(strs[i]) == 0 {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(strs[i])
	}

	return builder.String()
}

// 165. 比较版本号
// 给你两个版本号 version1 和 version2 ，请你比较它们。
// 版本号由一个或多个修订号组成，各修订号由一个 '.' 连接。每个修订号由 多位数字 组成，可能包含 前导零 。每个版本号至少包含一个字符。修订号从左到右编号，下标从 0 开始，最左边的修订号下标为 0 ，下一个修订号下标为 1 ，以此类推。例如，2.5.33 和 0.1 都是有效的版本号。
// 比较版本号时，请按从左到右的顺序依次比较它们的修订号。比较修订号时，只需比较 忽略任何前导零后的整数值 。也就是说，修订号 1 和修订号 001 相等 。如果版本号没有指定某个下标处的修订号，则该修订号视为 0 。例如，版本 1.0 小于版本 1.1 ，因为它们下标为 0 的修订号相同，而下标为 1 的修订号分别为 0 和 1 ，0 < 1 。
//
// 返回规则如下：
// 如果 version1 > version2 返回 1，
// 如果 version1 < version2 返回 -1，
// 除此之外返回 0。
//
// 示例 1：
// 输入：version1 = "1.01", version2 = "1.001" 输出：0
// 解释：忽略前导零，"01" 和 "001" 都表示相同的整数 "1"
//
// 示例 2：
// 输入：version1 = "1.0", version2 = "1.0.0" 输出：0
// 解释：version1 没有指定下标为 2 的修订号，即视为 "0"
//
// 示例 3：
// 输入：version1 = "0.1", version2 = "1.1" 输出：-1
// 解释：version1 中下标为 0 的修订号是 "0"，version2 中下标为 0 的修订号是 "1" 。0 < 1，所以 version1 < version2
//
// 示例 4：
// 输入：version1 = "1.0.1", version2 = "1" 输出：1
//
// 示例 5：
// 输入：version1 = "7.5.2.4", version2 = "7.5.3" 输出：-1
//
// 提示：
// 1 <= version1.length, version2.length <= 500
// version1 和 version2 仅包含数字和 '.'
// version1 和 version2 都是 有效版本号
// version1 和 version2 的所有修订号都可以存储在 32 位整数 中
func compareVersion(version1 string, version2 string) int {
	versionNums1, versionNums2 := strings.Split(version1, "."), strings.Split(version2, ".")
	size1, size2 := len(versionNums1), len(versionNums2)
	minSize := min(size1, size2)

	for i := 0; i < minSize; i++ {
		v1, _ := strconv.Atoi(versionNums1[i])
		v2, _ := strconv.Atoi(versionNums2[i])
		if v1 > v2 {
			return 1
		} else if v1 < v2 {
			return -1
		}
	}
	for i := minSize; i < size1; i++ {
		v1, _ := strconv.Atoi(versionNums1[i])
		if v1 > 0 {
			return 1
		}
	}
	for i := minSize; i < size2; i++ {
		v2, _ := strconv.Atoi(versionNums2[i])
		if v2 > 0 {
			return -1
		}
	}

	return 0
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// 179. 最大数
// 给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。
//
// 注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。
//
// 示例 1：
// 输入：nums = [10,2] 输出："210"
//
// 示例 2：
// 输入：nums = [3,30,34,5,9] 输出："9534330"
//
// 示例 3：
// 输入：nums = [1] 输出："1"
//
// 示例 4：
// 输入：nums = [10] 输出："10"
//
// 提示：
// 1 <= nums.length <= 100
// 0 <= nums[i] <= 109
func largestNumber(nums []int) string {
	n := len(nums)
	strs := make([]string, n)
	for i := range nums {
		strs[i] = strconv.Itoa(nums[i])
	}
	sort.Slice(strs, func(i, j int) bool {
		a, b := strs[i]+strs[j], strs[j]+strs[i]
		return b < a
	})
	if strs[0] == "0" {
		return "0"
	}
	var builder strings.Builder
	for i := range strs {
		builder.WriteString(strs[i])
	}

	return builder.String()
}

// 214. 最短回文串
// 给定一个字符串 s，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。
//
// 示例 1：
// 输入：s = "aacecaaa"
// 输出："aaacecaaa"
//
// 示例 2：
// 输入：s = "abcd"
// 输出："dcbabcd"
//
// 提示：
// 0 <= s.length <= 5 * 104
// s 仅由小写英文字母组成
func shortestPalindrome(s string) string {
	n := len(s)
	// KMP 算法
	fail := make([]int, n)
	for i := 0; i < n; i++ {
		fail[i] = -1
	}
	for i := 1; i < n; i++ {
		j := fail[i-1]
		for j != -1 && s[j+1] != s[i] {
			j = fail[j]
		}
		if s[j+1] == s[i] {
			fail[i] = j + 1
		}
	}
	best := -1
	// 反向匹配
	for i := n - 1; i >= 0; i-- {
		for best != -1 && s[best+1] != s[i] {
			best = fail[best]
		}
		if s[best+1] == s[i] {
			best++
		}
	}
	var add string
	if best == n-1 {
		add = ""
	} else {
		add = s[best+1:]
	}

	reverse := func(str string) string {
		size := len(str)

		bytes := []byte(str)
		for i, j := 0, size-1; i < j; {
			bytes[i], bytes[j] = bytes[j], bytes[i]
			i++
			j--
		}
		return string(bytes)
	}

	var builder strings.Builder
	builder.WriteString(reverse(add))
	builder.WriteString(s)
	return builder.String()

}

// 388. 文件的最长绝对路径
// 假设文件系统如下图所示：
//
// 这里将 dir 作为根目录中的唯一目录。dir 包含两个子目录 subdir1 和 subdir2 。subdir1 包含文件 file1.ext 和子目录 subsubdir1；subdir2 包含子目录 subsubdir2，该子目录下包含文件 file2.ext 。
//
// 在文本格式中，如下所示(⟶表示制表符)：
// dir
// ⟶ subdir1
// ⟶ ⟶ file1.ext
// ⟶ ⟶ subsubdir1
// ⟶ subdir2
// ⟶ ⟶ subsubdir2
// ⟶ ⟶ ⟶ file2.ext
// 如果是代码表示，上面的文件系统可以写为 "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" 。'\n' 和 '\t' 分别是换行符和制表符。
//
// 文件系统中的每个文件和文件夹都有一个唯一的 绝对路径 ，即必须打开才能到达文件/目录所在位置的目录顺序，所有路径用 '/' 连接。上面例子中，指向 file2.ext 的绝对路径是 "dir/subdir2/subsubdir2/file2.ext" 。每个目录名由字母、数字和/或空格组成，每个文件名遵循 name.extension 的格式，其中名称和扩展名由字母、数字和/或空格组成。
// 给定一个以上述格式表示文件系统的字符串 input ，返回文件系统中 指向文件的最长绝对路径 的长度。 如果系统中没有文件，返回 0。
//
// 示例 1：
// 输入：input = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
// 输出：20
// 解释：只有一个文件，绝对路径为 "dir/subdir2/file.ext" ，路径长度 20
// 路径 "dir/subdir1" 不含任何文件
//
// 示例 2：
// 输入：input = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
// 输出：32
// 解释：存在两个文件：
// "dir/subdir1/file1.ext" ，路径长度 21
// "dir/subdir2/subsubdir2/file2.ext" ，路径长度 32
// 返回 32 ，因为这是最长的路径
//
// 示例 3：
// 输入：input = "a"
// 输出：0
// 解释：不存在任何文件
//
// 示例 4：
// 输入：input = "file1.txt\nfile2.txt\nlongfile.txt"
// 输出：12
// 解释：根目录下有 3 个文件。
// 因为根目录中任何东西的绝对路径只是名称本身，所以答案是 "longfile.txt" ，路径长度为 12
//
// 提示：
// 1 <= input.length <= 104
// input 可能包含小写或大写的英文字母，一个换行符 '\n'，一个指表符 '\t'，一个点 '.'，一个空格 ' '，和数字。
func lengthLongestPath(input string) int {
	paths := strings.Split(input, "\n")

	getDirLevel := func(path string) int {
		level := 0
		for i := 0; i < len(path); i++ {
			if path[i] != '\t' {
				break
			}
			level++
		}
		return level
	}
	result := 0
	n := len(paths)
	pathLen, parentIndex := 0, -1
	pathLens := make([]int, n)
	for _, path := range paths {
		level := getDirLevel(path)
		if level <= 0 {
			pathLen = 0
		} else if level <= parentIndex {
			pathLen = pathLens[level-1]
		}

		pathLen += len(path) - level + 1
		if strings.Contains(path, ".") {
			result = max(result, pathLen)
		} else {
			pathLens[level] = pathLen
		}
		parentIndex = level
	}
	if result == 0 {
		return 0
	}

	return result - 1
}

// 424. 替换后的最长重复字符
// 给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。
//
// 注意：字符串长度 和 k 不会超过 104。
//
// 示例 1：
// 输入：s = "ABAB", k = 2
// 输出：4
// 解释：用两个'A'替换为两个'B',反之亦然。
//
// 示例 2：
// 输入：s = "AABABBA", k = 1
// 输出：4
// 解释：
// 将中间的一个'A'替换为'B',字符串变为 "AABBBBA"。
// 子串 "BBBB" 有最长重复字母, 答案为 4。
func characterReplacement(s string, k int) int {
	n := len(s)
	left, maxCount, result := 0, 0, 0
	letters := [26]int{}
	for right := 0; right < n; right++ {
		letters[s[right]-'A']++
		//
		maxCount = max(maxCount, letters[s[right]-'A'])
		if right-left+1-maxCount > k {
			letters[s[left]-'A']--
			left++
		}
		result = max(result, right-left+1)
	}
	return result
}

// 420. 强密码检验器
// 一个强密码应满足以下所有条件：
//
// 由至少6个，至多20个字符组成。
// 至少包含一个小写字母，一个大写字母，和一个数字。
// 同一字符不能连续出现三次 (比如 "...aaa..." 是不允许的, 但是 "...aa...a..." 是可以的)。
// 编写函数 strongPasswordChecker(s)，s 代表输入字符串，如果 s 已经符合强密码条件，则返回0；否则返回要将 s 修改为满足强密码条件的字符串所需要进行修改的最小步数。
//
// 插入、删除、替换任一字符都算作一次修改。
func strongPasswordChecker(password string) int {
	// 缺失个数
	missCounts := [3]int{1, 1, 1}
	// mod 3 求余个数
	modCounts := [3]int{0, 0, 0}

	modifyCount, n := 0, len(password)

	for i := 0; i < n; {
		c := password[i]
		if '0' <= c && c <= '9' {
			missCounts[0] = 0
		} else if 'a' <= c && c <= 'z' {
			missCounts[1] = 0
		} else if 'A' <= c && c <= 'Z' {
			missCounts[2] = 0
		}
		// 判断 连续
		start := i
		for i < n && password[i] == c {
			i++
		}
		subSize := i - start
		if subSize >= 3 {
			// 每3个替换1个，可保证不连续
			modifyCount += subSize / 3
			modCounts[subSize%3]++
		}
	}
	missCount := missCounts[0] + missCounts[1] + missCounts[2]
	// 字符串长度限制为 [6，20]

	// 1. 长度过短，仅考虑字符长度缺失和字符类型缺失
	if n < 6 {
		return max(6-n, missCount)
	}
	// 2. 长度合法，仅考虑连续字符串和字符类型缺失
	if n <= 20 {
		return max(modifyCount, missCount)
	}
	// 3. 长度过长，考虑删除过长的长度、连续字符串、字符类型缺失
	delCount := n - 20

	// 3n型子串，部分能通过删除一个转化成 3n + 2 型子串，每个子串删 1
	// aaa -> aa
	if delCount < modCounts[0] {
		modifyCount -= delCount
		return max(modifyCount, missCount) + n - 20
	}
	// 3n 型子串，全部都能通过删除转化成 3n + 2 型子串
	delCount -= modCounts[0]
	modifyCount -= modCounts[0]

	// 3n + 1 型子串，部分能通过删除转化成 3n + 2 型子串，每个子串删 2
	// aaa a  -> aa
	if delCount < modCounts[1]*2 {
		// 需要修改的 元素 每个 删除两个
		modifyCount -= delCount / 2
		return max(modifyCount, missCount) + n - 20
	}

	// 3n + 1 型子串，全部都能通过删除转化成 3n + 2 型子串
	delCount -= modCounts[1] << 1
	modifyCount -= modCounts[1]
	// 3n + 2型子串
	// (1) 删除 len - 20 个字符，使字符串长度降到合法长度
	// (2) 根据合法长度的公式，应为 Math.max(modifyCount, missCount);
	// (3) 由于删除时可以删除连续子串中的字符，减少 modifyCount
	//    aaa aaa aa  原需替换 2 次
	//    aaa aaa a   删除 1 次，仍需替换 2 次
	//    aaa aaa     删除 2 次，仍需替换 2 次
	//    aaa aa      删除 3 次，仍需替换 1 次
	// 即对于 3n + 2 型子串，删除 3 次可抵消替换 1 次
	// 其他型的子串可以转换成 3n + 2 型
	modifyCount -= delCount / 3

	return max(modifyCount, missCount) + n - 20
}

// 438. 找到字符串中所有字母异位词
// 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
//
// 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
//
// 示例 1:
// 输入: s = "cbaebabacd", p = "abc"
// 输出: [0,6]
// 解释:
// 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
// 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
//
// 示例 2:
// 输入: s = "abab", p = "ab"
// 输出: [0,1,2]
// 解释:
// 起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
// 起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
// 起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
//
// 提示:
// 1 <= s.length, p.length <= 3 * 104
// s 和 p 仅包含小写字母
func findAnagrams(s string, p string) []int {
	m, n := len(s), len(p)
	letters := [26]int{}

	for i := 0; i < n; i++ {
		letters[p[i]-'a']++
	}

	result := make([]int, 0)

	check := func() bool {
		for _, count := range letters {
			if count != 0 {
				return false
			}
		}
		return true
	}

	for i := 0; i < m; i++ {
		letters[s[i]-'a']--
		if i-n >= 0 {
			letters[s[i-n]-'a']++
		}
		if i-n+1 >= 0 && check() {
			result = append(result, i-n+1)
		}
	}

	return result
}

// 443. 压缩字符串
// 给你一个字符数组 chars ，请使用下述算法压缩：
//
// 从一个空字符串 s 开始。对于 chars 中的每组 连续重复字符 ：
//
// 如果这一组长度为 1 ，则将字符追加到 s 中。
// 否则，需要向 s 追加字符，后跟这一组的长度。
// 压缩后得到的字符串 s 不应该直接返回 ，需要转储到字符数组 chars 中。需要注意的是，如果组长度为 10 或 10 以上，则在 chars 数组中会被拆分为多个字符。
//
// 请在 修改完输入数组后 ，返回该数组的新长度。
//
// 你必须设计并实现一个只使用常量额外空间的算法来解决此问题。
//
// 示例 1：
// 输入：chars = ["a","a","b","b","c","c","c"]
// 输出：返回 6 ，输入数组的前 6 个字符应该是：["a","2","b","2","c","3"]
// 解释：
// "aa" 被 "a2" 替代。"bb" 被 "b2" 替代。"ccc" 被 "c3" 替代。
//
// 示例 2：
// 输入：chars = ["a"]
// 输出：返回 1 ，输入数组的前 1 个字符应该是：["a"]
// 解释：
// 没有任何字符串被替代。
//
// 示例 3：
// 输入：chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
// 输出：返回 4 ，输入数组的前 4 个字符应该是：["a","b","1","2"]。
// 解释：
// 由于字符 "a" 不重复，所以不会被压缩。"bbbbbbbbbbbb" 被 “b12” 替代。
// 注意每个数字在数组中都有它自己的位置。
//
// 提示：
// 1 <= chars.length <= 2000
// chars[i] 可以是小写英文字母、大写英文字母、数字或符号
func compress(chars []byte) int {
	n := len(chars)
	if n <= 1 {
		return n
	}
	count, index := 1, 1
	for i := 1; i <= n; i++ {
		if i == n || chars[i] != chars[index] {
			if count > 1 {
				num := strconv.Itoa(count)
				for j := 0; j < len(num); j++ {
					index++
					chars[index] = num[j]
				}
			}
			if i < n {
				chars[index] = chars[i]
			}
			count = 1
		} else {
			count++
		}
	}
	return index
}

// 468. 验证IP地址
// 编写一个函数来验证输入的字符串是否是有效的 IPv4 或 IPv6 地址。
//
// 如果是有效的 IPv4 地址，返回 "IPv4" ；
// 如果是有效的 IPv6 地址，返回 "IPv6" ；
// 如果不是上述类型的 IP 地址，返回 "Neither" 。
// IPv4 地址由十进制数和点来表示，每个地址包含 4 个十进制数，其范围为 0 - 255， 用(".")分割。比如，172.16.254.1；
//
// 同时，IPv4 地址内的数不会以 0 开头。比如，地址 172.16.254.01 是不合法的。
//
// IPv6 地址由 8 组 16 进制的数字来表示，每组表示 16 比特。这些组数字通过 (":")分割。比如,  2001:0db8:85a3:0000:0000:8a2e:0370:7334 是一个有效的地址。而且，我们可以加入一些以 0 开头的数字，字母可以使用大写，也可以是小写。所以， 2001:db8:85a3:0:0:8A2E:0370:7334 也是一个有效的 IPv6 address地址 (即，忽略 0 开头，忽略大小写)。
//
// 然而，我们不能因为某个组的值为 0，而使用一个空的组，以至于出现 (::) 的情况。 比如， 2001:0db8:85a3::8A2E:0370:7334 是无效的 IPv6 地址。
//
// 同时，在 IPv6 地址中，多余的 0 也是不被允许的。比如， 02001:0db8:85a3:0000:0000:8a2e:0370:7334 是无效的。
//
// 示例 1：
// 输入：IP = "172.16.254.1"
// 输出："IPv4"
// 解释：有效的 IPv4 地址，返回 "IPv4"
//
// 示例 2：
// 输入：IP = "2001:0db8:85a3:0:0:8A2E:0370:7334"
// 输出："IPv6"
// 解释：有效的 IPv6 地址，返回 "IPv6"
//
// 示例 3：
// 输入：IP = "256.256.256.256"
// 输出："Neither"
// 解释：既不是 IPv4 地址，又不是 IPv6 地址
//
// 示例 4：
// 输入：IP = "2001:0db8:85a3:0:0:8A2E:0370:7334:"
// 输出："Neither"
//
// 示例 5：
// 输入：IP = "1e1.4.5.6"
// 输出："Neither"
//
// 提示：
// IP 仅由英文字母，数字，字符 '.' 和 ':' 组成。
func validIPAddress(queryIP string) string {
	IpAddrOut := "Neither"

	validIPv4 := func(ip string) string {
		ips := strings.Split(ip, ".")
		n := len(ips)
		if n != 4 {
			return IpAddrOut
		}
		for _, str := range ips {
			l := len(str)
			if l > 3 || l == 0 {
				return IpAddrOut
			}
			if l > 1 && str[0] == '0' {
				return IpAddrOut
			}
			num := 0
			for i := 0; i < l; i++ {
				if !('0' <= str[i] && str[i] <= '9') {
					return IpAddrOut
				}
				num = num*10 + int(str[i]-'0')
			}
			if num < 0 || num > 255 {
				return IpAddrOut
			}

		}

		return "IPv4"
	}
	validIPv6 := func(ip string) string {
		ips := strings.Split(ip, ":")
		n := len(ips)
		if n != 8 {
			return IpAddrOut
		}
		for _, str := range ips {
			l := len(str)
			if l > 4 || l == 0 {
				return IpAddrOut
			}

			for i := 0; i < l; i++ {
				if !(('0' <= str[i] && str[i] <= '9') || ('a' <= str[i] && str[i] <= 'f') || ('A' <= str[i] && str[i] <= 'F')) {
					return IpAddrOut
				}
			}

		}
		return "IPv6"
	}

	if strings.Contains(queryIP, ".") {
		return validIPv4(queryIP)
	} else if strings.Contains(queryIP, ":") {
		return validIPv6(queryIP)
	}
	return IpAddrOut
}

// 481. 神奇字符串
// 神奇字符串 s 仅由 '1' 和 '2' 组成，并需要遵守下面的规则：
//
// 神奇字符串 s 的神奇之处在于，串联字符串中 '1' 和 '2' 的连续出现次数可以生成该字符串。
// s 的前几个元素是 s = "1221121221221121122……" 。如果将 s 中连续的若干 1 和 2 进行分组，可以得到 "1 22 11 2 1 22 1 22 11 2 11 22 ......" 。每组中 1 或者 2 的出现次数分别是 "1 2 2 1 1 2 1 2 2 1 2 2 ......" 。上面的出现次数正是 s 自身。
//
// 给你一个整数 n ，返回在神奇字符串 s 的前 n 个数字中 1 的数目。
//
// 示例 1：
// 输入：n = 6 输出：3
// 解释：神奇字符串 s 的前 6 个元素是 “122112”，它包含三个 1，因此返回 3 。
//
// 示例 2：
// 输入：n = 1 输出：1
//
// 提示：
// 1 <= n <= 105
func magicalString(n int) int {
	if n <= 3 {
		return 1
	}
	nums := make([]bool, n+2)
	// false 1  true 2
	nums[1], nums[2] = true, true
	end := 2
	for i := 2; i < n; i++ {
		if end >= n {
			break
		}
		endNum := nums[end]
		end++
		nums[end] = !endNum
		if nums[i] { // 2
			end++
			nums[end] = !endNum
		}
	}

	count := 0
	for i := 0; i < n; i++ {
		if !nums[i] {
			count++
		}
	}

	return count
}

// 488. 祖玛游戏
// 你正在参与祖玛游戏的一个变种。
// 在这个祖玛游戏变体中，桌面上有 一排 彩球，每个球的颜色可能是：红色 'R'、黄色 'Y'、蓝色 'B'、绿色 'G' 或白色 'W' 。你的手中也有一些彩球。
// 你的目标是 清空 桌面上所有的球。每一回合：
//
// 从你手上的彩球中选出 任意一颗 ，然后将其插入桌面上那一排球中：两球之间或这一排球的任一端。
// 接着，如果有出现 三个或者三个以上 且 颜色相同 的球相连的话，就把它们移除掉。
// 如果这种移除操作同样导致出现三个或者三个以上且颜色相同的球相连，则可以继续移除这些球，直到不再满足移除条件。
// 如果桌面上所有球都被移除，则认为你赢得本场游戏。
// 重复这个过程，直到你赢了游戏或者手中没有更多的球。
// 给你一个字符串 board ，表示桌面上最开始的那排球。另给你一个字符串 hand ，表示手里的彩球。请你按上述操作步骤移除掉桌上所有球，计算并返回所需的 最少 球数。如果不能移除桌上所有的球，返回 -1 。
//
// 示例 1：
// 输入：board = "WRRBBW", hand = "RB"
// 输出：-1
// 解释：无法移除桌面上的所有球。可以得到的最好局面是：
// - 插入一个 'R' ，使桌面变为 WRRRBBW 。WRRRBBW -> WBBW
// - 插入一个 'B' ，使桌面变为 WBBBW 。WBBBW -> WW
// 桌面上还剩着球，没有其他球可以插入。
//
// 示例 2：
// 输入：board = "WWRRBBWW", hand = "WRBRW"
// 输出：2
// 解释：要想清空桌面上的球，可以按下述步骤：
// - 插入一个 'R' ，使桌面变为 WWRRRBBWW 。WWRRRBBWW -> WWBBWW
// - 插入一个 'B' ，使桌面变为 WWBBBWW 。WWBBBWW -> WWWW -> empty
// 只需从手中出 2 个球就可以清空桌面。
//
// 示例 3：
// 输入：board = "G", hand = "GGGGG"
// 输出：2
// 解释：要想清空桌面上的球，可以按下述步骤：
// - 插入一个 'G' ，使桌面变为 GG 。
// - 插入一个 'G' ，使桌面变为 GGG 。GGG -> empty
// 只需从手中出 2 个球就可以清空桌面。
//
// 示例 4：
// 输入：board = "RBYYBBRRB", hand = "YRBGB"
// 输出：3
// 解释：要想清空桌面上的球，可以按下述步骤：
// - 插入一个 'Y' ，使桌面变为 RBYYYBBRRB 。RBYYYBBRRB -> RBBBRRB -> RRRB -> B
// - 插入一个 'B' ，使桌面变为 BB 。
// - 插入一个 'B' ，使桌面变为 BBB 。BBB -> empty
// 只需从手中出 3 个球就可以清空桌面。
//
// 提示：
// 1 <= board.length <= 16
// 1 <= hand.length <= 5
// board 和 hand 由字符 'R'、'Y'、'B'、'G' 和 'W' 组成
// 桌面上一开始的球中，不会有三个及三个以上颜色相同且连着的球
func findMinStep(board string, hand string) int {
	handBall := [26]int{}
	colors := []byte{'R', 'Y', 'B', 'G', 'W'}
	minStep := math.MaxInt32
	for i := 0; i < len(hand); i++ {
		handBall[hand[i]-'A']++
	}
	eliminate := func(s string) string {
		flag := true
		for flag {
			flag = false
			for i := 0; i < len(s); i++ {
				j := i + 1
				for j < len(s) && s[j] == s[i] {
					j++
				}
				if j-i >= 3 {
					s = s[:i] + s[j:]
					flag = true
				}
			}
		}
		return s
	}

	var back func(s string, step int)

	back = func(s string, step int) {
		if step >= minStep {
			return
		}
		l := len(s)
		if l == 0 {
			minStep = step
			return
		}
		for i := 0; i < l; i++ {
			c := s[i]
			j := i
			for j+1 < l && s[j+1] == c {
				j++
			}
			// 一个球
			if i == j && handBall[c-'A'] >= 2 {
				handBall[c-'A'] -= 2
				tmp := s[:i] + string(c) + string(c) + s[i:]
				back(eliminate(tmp), step+2)
				handBall[c-'A'] += 2
			} else if i+1 == j { // 两个球
				if handBall[c-'A'] >= 1 {
					tmp := s[:i] + string(c) + s[i:]
					handBall[c-'A']--
					back(eliminate(tmp), step+1)
					handBall[c-'A']++
				}
				for _, color := range colors {
					if color == c {
						continue
					}
					if handBall[color-'A'] >= 1 {
						tmp := s[:i+1] + string(color) + s[i+1:] // 尝试往这两个颜色相同且相邻的球中间插入一个颜色不同的球
						handBall[color-'A']--
						back(eliminate(tmp), step+1)
						handBall[color-'A']++
					}
				}
			}
		}

	}

	back(board, 0)
	if minStep == math.MaxInt32 {
		return -1
	}
	return minStep
}

// 522. 最长特殊序列 II
// 给定字符串列表，你需要从它们中找出最长的特殊序列。最长特殊序列定义如下：该序列为某字符串独有的最长子序列（即不能是其他字符串的子序列）。
//
// 子序列可以通过删去字符串中的某些字符实现，但不能改变剩余字符的相对顺序。空序列为所有字符串的子序列，任何字符串为其自身的子序列。
// 输入将是一个字符串列表，输出是最长特殊序列的长度。如果最长特殊序列不存在，返回 -1 。
//
// 示例：
// 输入: "aba", "cdc", "eae" 输出: 3
//
// 提示：
// 所有给定的字符串长度不会超过 10 。
// 给定字符串列表的长度将在 [2, 50 ] 之间。
func findLUSlengthII(strs []string) int {
	sort.Slice(strs, func(i, j int) bool {
		return len(strs[i]) > len(strs[j])
	})

	n := len(strs)
	for i := 0; i < n; i++ {
		flag := true

		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if isSubsequence(strs[i], strs[j]) {
				flag = false
				break
			}
		}

		if flag {
			return len(strs[i])
		}
	}

	return -1
}

// 判断 a 是否是 b 的子序列
func isSubsequence(a, b string) bool {
	if len(a) > len(b) {
		return false
	}
	if a == b {
		return true
	}
	index := 0
	for i := 0; i < len(b) && index < len(a); i++ {
		if a[index] == b[i] {
			index++
		}
	}
	return index == len(a)
}

// 859. 亲密字符串
// 给你两个字符串 s 和 goal ，只要我们可以通过交换 s 中的两个字母得到与 goal 相等的结果，就返回 true ；否则返回 false 。
// 交换字母的定义是：取两个下标 i 和 j （下标从 0 开始）且满足 i != j ，接着交换 s[i] 和 s[j] 处的字符。
// 例如，在 "abcd" 中交换下标 0 和下标 2 的元素可以生成 "cbad" 。
//
// 示例 1：
// 输入：s = "ab", goal = "ba"
// 输出：true
// 解释：你可以交换 s[0] = 'a' 和 s[1] = 'b' 生成 "ba"，此时 s 和 goal 相等。
// 示例 2：
//
// 输入：s = "ab", goal = "ab"
// 输出：false
// 解释：你只能交换 s[0] = 'a' 和 s[1] = 'b' 生成 "ba"，此时 s 和 goal 不相等。
//
// 示例 3：
// 输入：s = "aa", goal = "aa"
// 输出：true
// 解释：你可以交换 s[0] = 'a' 和 s[1] = 'a' 生成 "aa"，此时 s 和 goal 相等。
//
// 示例 4：
// 输入：s = "aaaaaaabc", goal = "aaaaaaacb"
// 输出：true
//
// 提示：
// 1 <= s.length, goal.length <= 2 * 104
// s 和 goal 由小写英文字母组成
func buddyStrings(s string, goal string) bool {
	if len(s) != len(goal) {
		return false
	}
	n := len(s)
	diff, twoCount := 0, false
	first, second := -1, -1
	letters := [26]int{}
	for i := 0; i < n; i++ {
		if s[i] == goal[i] {
			index := int(s[i] - 'a')
			letters[index]++
			if letters[index] >= 2 {
				twoCount = true
			}
		} else {
			diff++
			if diff == 1 {
				first = i
			}
			if diff == 2 {
				second = i
			}
			if diff > 2 {
				return false
			}
		}
	}
	if diff == 0 && twoCount {
		return true
	}
	if diff == 2 && s[first] == goal[second] && s[second] == goal[first] {
		return true
	}
	return false
}

// 524. 通过删除字母匹配到字典里最长单词
// 给你一个字符串 s 和一个字符串数组 dictionary ，找出并返回 dictionary 中最长的字符串，该字符串可以通过删除 s 中的某些字符得到。
//
// 如果答案不止一个，返回长度最长且字母序最小的字符串。如果答案不存在，则返回空字符串。
//
// 示例 1：
// 输入：s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
// 输出："apple"
//
// 示例 2：
// 输入：s = "abpcplea", dictionary = ["a","b","c"]
// 输出："a"
//
// 提示：
// 1 <= s.length <= 1000
// 1 <= dictionary.length <= 1000
// 1 <= dictionary[i].length <= 1000
// s 和 dictionary[i] 仅由小写英文字母组成
func findLongestWord(s string, dictionary []string) string {
	sort.Strings(dictionary)
	result := ""
	for _, word := range dictionary {
		if isSubsequence(word, s) {
			if len(word) > len(result) {
				result = word
			}
		}
	}
	return result
}

// 567. 字符串的排列
// 给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。
//
// 换句话说，s1 的排列之一是 s2 的 子串 。
//
// 示例 1：
// 输入：s1 = "ab" s2 = "eidbaooo"
// 输出：true
// 解释：s2 包含 s1 的排列之一 ("ba").
//
// 示例 2：
// 输入：s1= "ab" s2 = "eidboaoo"  输出：false
//
// 提示：
// 1 <= s1.length, s2.length <= 104
// s1 和 s2 仅包含小写字母
func checkInclusion(s1 string, s2 string) bool {
	// 滑动窗口
	m, n := len(s1), len(s2)
	if m > n {
		return false
	}
	letters1, letters2 := [26]int{}, [26]int{}

	for i := 0; i < m; i++ {
		letters1[s1[i]-'a']++
		letters2[s2[i]-'a']++
	}
	match := func() bool {
		for i := 0; i < 26; i++ {
			if letters1[i] != letters2[i] {
				return false
			}
		}
		return true
	}

	for i := m; i < n; i++ {
		if match() {
			return true
		}
		letters2[s2[i]-'a']++
		letters2[s2[i-m]-'a']--
	}
	return match()
}

// 564. 寻找最近的回文数
// 给定一个整数 n ，你需要找到与它最近的回文数（不包括自身）。
// “最近的”定义为两个整数差的绝对值最小。
//
// 示例 1:
// 输入: "123" 输出: "121"
// 注意:
// n 是由字符串表示的正整数，其长度不超过18。
// 如果有多个结果，返回最小的那个。
func nearestPalindromic(n string) string {
	m := len(n)
	num, _ := strconv.Atoi(n)
	if num == 11 {
		return "9"
	} else if num <= 10 {
		return strconv.Itoa(num - 1)
	}
	// 奇数
	odd := m&1 == 1
	midIdx := m >> 1
	var strPre string
	if odd {
		strPre = n[:midIdx+1]
	} else {
		strPre = n[:midIdx]
	}
	halfNum, _ := strconv.Atoi(strPre)
	if isTenMultiple(n) { // 1000 -> 999
		return strconv.Itoa(num - 1)
	} else if validPalindromeString(n) { // 本身是回文
		if n[midIdx] == '0' {
			// 10001 -> 9999
			if is101(n) {
				return strconv.Itoa(num - 2)
			}
		} else if n[midIdx] == '9' {
			// 9999 -> 10001
			if isAllNine(n) {
				return strconv.Itoa(num + 2)
			}
		}
		tmp1, tmp2 := getResult(odd, strconv.Itoa(halfNum-1)), getResult(odd, strconv.Itoa(halfNum+1))
		tmpNum1, _ := strconv.Atoi(tmp1)
		tmpNum2, _ := strconv.Atoi(tmp2)
		if num-tmpNum1 <= tmpNum2-num {
			return tmp1
		} else {
			return tmp2
		}
	} else {
		// 否则进行去中间数+1，-1，和中间数不变三个进行比较
		tmp1, tmp2, tmp3 := getResult(odd, strconv.Itoa(halfNum+1)), getResult(odd, strconv.Itoa(halfNum-1)),
			getResult(odd, strPre)
		tmpNum1, _ := strconv.Atoi(tmp1)
		tmpNum2, _ := strconv.Atoi(tmp2)
		tmpNum3, _ := strconv.Atoi(tmp3)

		if tmpNum1-num < num-tmpNum2 && tmpNum1-num < abs(tmpNum3-num) {
			return tmp1
		} else if num-tmpNum2 < tmpNum1-num && num-tmpNum2 <= abs(tmpNum3-num) {
			return tmp2
		} else {
			return tmp3
		}

	}
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

func getResult(odd bool, strPre string) string {
	strSuf := reverse(strPre)
	if odd {
		return strPre + strSuf[1:]
	}
	return strPre + strSuf
}

func validPalindromeString(s string) bool {
	left, right := 0, len(s)-1
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}
func isAllNine(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] != '9' {
			return false
		}
	}
	return true
}

func is101(s string) bool {
	n := len(s)
	if s[0] != '1' || s[n-1] != '1' {
		return false
	}
	for i := 1; i < n-1; i++ {
		if s[i] != '0' {
			return false
		}
	}
	return true
}

func isTenMultiple(s string) bool {
	if s[0] != '1' {
		return false
	}
	for i := 1; i < len(s); i++ {
		if s[i] != '0' {
			return false
		}
	}

	return true
}

func reverse(s string) string {
	bytes := []byte(s)
	left, right := 0, len(s)-1
	for left < right {
		bytes[left], bytes[right] = bytes[right], bytes[left]
		left++
		right--
	}
	return string(bytes)
}

// 1446. 连续字符
// 给你一个字符串 s ，字符串的「能量」定义为：只包含一种字符的最长非空子字符串的长度。
// 请你返回字符串的能量。
//
// 例 1：
// 输入：s = "leetcode" 输出：2
// 解释：子字符串 "ee" 长度为 2 ，只包含字符 'e' 。
//
// 示例 2：
// 输入：s = "abbcccddddeeeeedcba" 输出：5
// 解释：子字符串 "eeeee" 长度为 5 ，只包含字符 'e' 。
//
// 示例 3：
// 输入：s = "triplepillooooow" 输出：5
//
// 示例 4：
// 输入：s = "hooraaaaaaaaaaay" 输出：11
//
// 示例 5：
// 输入：s = "tourist" 输出：1
//
// 提示：
// 1 <= s.length <= 500
// s 只包含小写英文字母。
func maxPower(s string) int {
	n := len(s)
	result, count := 1, 1
	for i := 1; i < n; i++ {
		if s[i] == s[i-1] {
			count++
		} else {
			count = 1
		}
		result = max(result, count)
	}

	return result
}

// 1816. 截断句子
// 句子 是一个单词列表，列表中的单词之间用单个空格隔开，且不存在前导或尾随空格。每个单词仅由大小写英文字母组成（不含标点符号）。
//
// 例如，"Hello World"、"HELLO" 和 "hello world hello world" 都是句子。
// 给你一个句子 s 和一个整数 k ，请你将 s 截断,使截断后的句子仅含 前 k 个单词。返回 截断 s 后得到的句子。
//
// 示例 1：
// 输入：s = "Hello how are you Contestant", k = 4
// 输出："Hello how are you"
// 解释：
// s 中的单词为 ["Hello", "how" "are", "you", "Contestant"]
// 前 4 个单词为 ["Hello", "how", "are", "you"]
// 因此，应当返回 "Hello how are you"
//
// 示例 2：
// 输入：s = "What is the solution to this problem", k = 4
// 输出："What is the solution"
// 解释：
// s 中的单词为 ["What", "is" "the", "solution", "to", "this", "problem"]
// 前 4 个单词为 ["What", "is", "the", "solution"]
// 因此，应当返回 "What is the solution"
//
// 示例 3：
// 输入：s = "chopper is not a tanuki", k = 5
// 输出："chopper is not a tanuki"
//
// 提示：
// 1 <= s.length <= 500
// k 的取值范围是 [1,  s 中单词的数目]
// s 仅由大小写英文字母和空格组成
// s 中的单词之间由单个空格隔开
// 不存在前导或尾随空格
func truncateSentence(s string, k int) string {
	words := strings.Split(s, " ")
	var builder strings.Builder
	for i := 0; i < k; i++ {
		if i > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(words[i])
	}
	return builder.String()
}

// 686. 重复叠加字符串匹配
// 给定两个字符串 a 和 b，寻找重复叠加字符串 a 的最小次数，使得字符串 b 成为叠加后的字符串 a 的子串，如果不存在则返回 -1。
//
// 注意：字符串 "abc" 重复叠加 0 次是 ""，重复叠加 1 次是 "abc"，重复叠加 2 次是 "abcabc"。
//
// 示例 1：
// 输入：a = "abcd", b = "cdabcdab" 输出：3
// 解释：a 重复叠加三遍后为 "abcdabcdabcd", 此时 b 是其子串。
//
// 示例 2：
// 输入：a = "a", b = "aa" 输出：2
//
// 示例 3：
// 输入：a = "a", b = "a"  输出：1
//
// 示例 4：
// 输入：a = "abc", b = "wxyz" 输出：-1
//
// 提示：
// 1 <= a.length <= 104
// 1 <= b.length <= 104
// a 和 b 由小写英文字母组成
func repeatedStringMatch(a string, b string) int {
	m, n := len(a), len(b)
	count := n / m
	if m*count < n {
		count++
	}
	tmp := strings.Repeat(a, count)
	if strings.Index(tmp, b) >= 0 {
		return count
	}
	tmp += a
	if strings.Index(tmp, b) >= 0 {
		return count + 1
	}

	return -1
}

// 1576. 替换所有的问号
// 给你一个仅包含小写英文字母和 '?' 字符的字符串 s，请你将所有的 '?' 转换为若干小写字母，使最终的字符串不包含任何 连续重复 的字符。
//
// 注意：你 不能 修改非 '?' 字符。
// 题目测试用例保证 除 '?' 字符 之外，不存在连续重复的字符。
// 在完成所有转换（可能无需转换）后返回最终的字符串。如果有多个解决方案，请返回其中任何一个。可以证明，在给定的约束条件下，答案总是存在的。
//
// 示例 1：
// 输入：s = "?zs"
// 输出："azs"
// 解释：该示例共有 25 种解决方案，从 "azs" 到 "yzs" 都是符合题目要求的。只有 "z" 是无效的修改，因为字符串 "zzs" 中有连续重复的两个 'z' 。
//
// 示例 2：
// 输入：s = "ubv?w"
// 输出："ubvaw"
// 解释：该示例共有 24 种解决方案，只有替换成 "v" 和 "w" 不符合题目要求。因为 "ubvvw" 和 "ubvww" 都包含连续重复的字符。
//
// 示例 3：
// 输入：s = "j?qg??b"
// 输出："jaqgacb"
//
// 示例 4：
// 输入：s = "??yw?ipkj?"
// 输出："acywaipkja"
//
// 提示：
// 1 <= s.length <= 100
// s 仅包含小写英文字母和 '?' 字符
func modifyString(s string) string {

	getNext := func(c byte) byte {
		if c == 'z' {
			return 'a'
		}
		return c + 1
	}

	bytes := []byte(s)
	for i := 0; i < len(s); i++ {
		if bytes[i] != '?' {
			continue
		}
		var c1 byte
		if i == 0 {
			c1 = 'a'
		} else {
			c1 = getNext(bytes[i-1])
		}
		if i+1 < len(s) {
			for c1 == bytes[i+1] {
				c1 = getNext(c1)
			}
		}
		bytes[i] = c1
	}

	return string(bytes)
}

// 647. 回文子串
// 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
//
// 回文字符串 是正着读和倒过来读一样的字符串。
// 子字符串 是字符串中的由连续字符组成的一个序列。
// 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
//
// 示例 1：
// 输入：s = "abc"
// 输出：3
// 解释：三个回文子串: "a", "b", "c"
//
// 示例 2：
// 输入：s = "aaa"
// 输出：6
// 解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
//
// 提示：
// 1 <= s.length <= 1000
// s 由小写英文字母组成
func countSubstrings(s string) int {
	n := len(s)

	result := 0
	// 双指针
	countStr := func(left, right int) {
		for left >= 0 && right < n && s[left] == s[right] {
			result++
			left--
			right++
		}
	}

	for i := 0; i < n; i++ {
		countStr(i, i)
		countStr(i, i+1)
	}

	return result
}

// 159. 至多包含两个不同字符的最长子串
// 给定一个字符串 s ，找出 至多 包含两个不同字符的最长子串 t ，并返回该子串的长度。
//
// 示例 1:
// 输入: "eceba"
// 输出: 3
// 解释: t 是 "ece"，长度为3。
//
// 示例 2:
// 输入: "ccaabbb"
// 输出: 5
// 解释: t 是 "aabbb"，长度为5。
func lengthOfLongestSubstringTwoDistinct(s string) int {

	result := 0
	letterMap := make(map[byte]int)

	left, n := 0, len(s)
	for i := 0; i < n; i++ {
		c := s[i]
		letterMap[c]++
		for i+1 < n && s[i+1] == c {
			i++
			letterMap[c]++
		}
		for len(letterMap) > 2 {
			letterMap[s[left]]--
			if letterMap[s[left]] == 0 {
				delete(letterMap, s[left])
			}
			left++
		}
		result = max(result, i-left+1)
	}
	return result
}

// 161. 相隔为 1 的编辑距离
// 给定两个字符串 s 和 t，判断他们的编辑距离是否为 1。
//
// 注意：
// 满足编辑距离等于 1 有三种可能的情形：
// 往 s 中插入一个字符得到 t
// 从 s 中删除一个字符得到 t
// 在 s 中替换一个字符得到 t
//
// 示例 1：
// 输入: s = "ab", t = "acb"
// 输出: true
// 解释: 可以将 'c' 插入字符串 s 来得到 t。
//
// 示例 2:
// 输入: s = "cab", t = "ad"
// 输出: false
// 解释: 无法通过 1 步操作使 s 变为 t。
//
// 示例 3:
// 输入: s = "1203", t = "1213"
// 输出: true
// 解释: 可以将字符串 s 中的 '0' 替换为 '1' 来得到 t。
func isOneEditDistance(s string, t string) bool {
	m, n := len(s), len(t)
	if n > m {
		return isOneEditDistance(t, s)
	}
	if m-n > 1 {
		return false
	}
	diff := 0
	if m == n {
		for i := 0; i < m; i++ {
			if s[i] != t[i] {
				diff++
			}
		}
		return diff == 1
	}
	// m > n
	for i := 0; i < n; i++ {
		if s[i] != t[i] {
			return s[i+1:] == t[i:]
		}
	}
	return true
}

// 186. 翻转字符串里的单词 II
// 给定一个字符串，逐个翻转字符串中的每个单词。
//
// 示例：
// 输入: ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
// 输出: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]
// 注意：
// 单词的定义是不包含空格的一系列字符
// 输入字符串中不会包含前置或尾随的空格
// 单词与单词之间永远是以单个空格隔开的
// 进阶：使用 O(1) 额外空间复杂度的原地解法。
func reverseWordsII(s []byte) {

	n := len(s)
	// 全部反转
	left, right := 0, n-1
	for left < right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
	for i := 0; i < n; i++ {
		left = i
		for i+1 < n && s[i+1] != ' ' {
			i++
		}
		right = i
		i++
		// 按空格 每个反转
		for left < right {
			s[left], s[right] = s[right], s[left]
			left++
			right--
		}
	}

}

// 2047. 句子中的有效单词数
// 句子仅由小写字母（'a' 到 'z'）、数字（'0' 到 '9'）、连字符（'-'）、标点符号（'!'、'.' 和 ','）以及空格（' '）组成。每个句子可以根据空格分解成 一个或者多个 token ，这些 token 之间由一个或者多个空格 ' ' 分隔。
//
// 如果一个 token 同时满足下述条件，则认为这个 token 是一个有效单词：
//
// 仅由小写字母、连字符和/或标点（不含数字）。
// 至多一个 连字符 '-' 。如果存在，连字符两侧应当都存在小写字母（"a-b" 是一个有效单词，但 "-ab" 和 "ab-" 不是有效单词）。
// 至多一个 标点符号。如果存在，标点符号应当位于 token 的 末尾 。
// 这里给出几个有效单词的例子："a-b."、"afad"、"ba-c"、"a!" 和 "!" 。
//
// 给你一个字符串 sentence ，请你找出并返回 sentence 中 有效单词的数目 。
//
// 示例 1：
// 输入：sentence = "cat and  dog"
// 输出：3
// 解释：句子中的有效单词是 "cat"、"and" 和 "dog"
//
// 示例 2：
// 输入：sentence = "!this  1-s b8d!"
// 输出：0
// 解释：句子中没有有效单词
// "!this" 不是有效单词，因为它以一个标点开头
// "1-s" 和 "b8d" 也不是有效单词，因为它们都包含数字
//
// 示例 3：
// 输入：sentence = "alice and  bob are playing stone-game10"
// 输出：5
// 解释：句子中的有效单词是 "alice"、"and"、"bob"、"are" 和 "playing"
// "stone-game10" 不是有效单词，因为它含有数字
//
// 示例 4：
// 输入：sentence = "he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."
// 输出：6
// 解释：句子中的有效单词是 "he"、"bought"、"pencils,"、"erasers,"、"and" 和 "pencil-sharpener."
//
// 提示：
// 1 <= sentence.length <= 1000
// sentence 由小写英文字母、数字（0-9）、以及字符（' '、'-'、'!'、'.' 和 ','）组成
// 句子中至少有 1 个 token
func countValidWords(sentence string) int {
	count := 0

	valid := func(word string) bool {
		// 仅由小写字母、连字符和/或标点（不含数字）。
		// 至多一个 连字符 '-' 。如果存在，连字符两侧应当都存在小写字母（"a-b" 是一个有效单词，但 "-ab" 和 "ab-" 不是有效单词）。
		// 至多一个 标点符号。如果存在，标点符号应当位于 token 的 末尾 。
		n := len(word)
		a, b := 0, 0
		aIdx := 0
		for i, c := range word {
			if c >= '0' && c <= '9' {
				return false
			}

			if c == '-' {
				a++
				aIdx = i
				if a > 1 || i == 0 || i == n-1 {
					return false
				}
			}
			if c == '!' || c == '.' || c == ',' {
				b++
				if b > 1 || i != n-1 {
					return false
				}
			}
		}
		// 如果存在，连字符两侧应当都存在小写字母（"a-b" 是一个有效单词，但 "-ab" 和 "ab-" 不是有效单词）。
		if aIdx > 0 {
			left, right := 0, 0
			for i, c := range word {
				if c >= 'a' && c <= 'z' {
					if i < aIdx {
						left++
					}
					if i > aIdx {
						right++
					}
				}
			}
			return left > 0 && right > 0
		}
		return true
	}

	for _, word := range strings.Split(sentence, " ") {
		if len(word) == 0 {
			continue
		}
		fmt.Println(word)
		if valid(word) {
			count++
		}
	}

	return count
}

// 2000. 反转单词前缀
// 给你一个下标从 0 开始的字符串 word 和一个字符 ch 。找出 ch 第一次出现的下标 i ，反转 word 中从下标 0 开始、直到下标 i 结束（含下标 i ）的那段字符。如果 word 中不存在字符 ch ，则无需进行任何操作。
//
// 例如，如果 word = "abcdefd" 且 ch = "d" ，那么你应该 反转 从下标 0 开始、直到下标 3 结束（含下标 3 ）。结果字符串将会是 "dcbaefd" 。
// 返回 结果字符串 。
//
// 示例 1：
// 输入：word = "abcdefd", ch = "d"
// 输出："dcbaefd"
// 解释："d" 第一次出现在下标 3 。
// 反转从下标 0 到下标 3（含下标 3）的这段字符，结果字符串是 "dcbaefd" 。
//
// 示例 2：
// 输入：word = "xyxzxe", ch = "z"
// 输出："zxyxxe"
// 解释："z" 第一次也是唯一一次出现是在下标 3 。
// 反转从下标 0 到下标 3（含下标 3）的这段字符，结果字符串是 "zxyxxe" 。
//
// 示例 3：
// 输入：word = "abcd", ch = "z"
// 输出："abcd"
// 解释："z" 不存在于 word 中。
// 无需执行反转操作，结果字符串是 "abcd" 。
//
// 提示：
// 1 <= word.length <= 250
// word 由小写英文字母组成
// ch 是一个小写英文字母
func reversePrefix(word string, ch byte) string {
	chars := []byte(word)
	left, right := 0, strings.IndexByte(word, ch)

	for left < right {
		chars[left], chars[right] = chars[right], chars[left]
		left++
		right--
	}
	return string(chars)
}

// 917. 仅仅反转字母
// 给你一个字符串 s ，根据下述规则反转字符串：
//
// 所有非英文字母保留在原有位置。
// 所有英文字母（小写或大写）位置反转。
// 返回反转后的 s 。
//
// 示例 1：
// 输入：s = "ab-cd"
// 输出："dc-ba"
//
// 示例 2：
// 输入：s = "a-bC-dEf-ghIj"
// 输出："j-Ih-gfE-dCba"
//
// 示例 3：
// 输入：s = "Test1ng-Leet=code-Q!"
// 输出："Qedo1ct-eeLg=ntse-T!"
//
// 提示
// 1 <= s.length <= 100
// s 仅由 ASCII 值在范围 [33, 122] 的字符组成
// s 不含 '\"' 或 '\\'
func reverseOnlyLetters(s string) string {
	chars := []byte(s)
	n := len(s)
	left, right := 0, n-1
	for left < right {
		if !unicode.IsLetter(rune(chars[left])) {
			left++
			continue
		}
		if !unicode.IsLetter(rune(chars[right])) {
			right--
			continue
		}
		chars[left], chars[right] = chars[right], chars[left]
		left++
		right--
	}

	return string(chars)
}

// 722. 删除注释
// 给一个 C++ 程序，删除程序中的注释。这个程序source是一个数组，其中source[i]表示第 i 行源码。 这表示每行源码由 '\n' 分隔。
//
// 在 C++ 中有两种注释风格，行内注释和块注释。
//
// 字符串// 表示行注释，表示//和其右侧的其余字符应该被忽略。
// 字符串/* 表示一个块注释，它表示直到下一个（非重叠）出现的*/之间的所有字符都应该被忽略。（阅读顺序为从左到右）非重叠是指，字符串/*/并没有结束块注释，因为注释的结尾与开头相重叠。
// 第一个有效注释优先于其他注释。
//
// 如果字符串//出现在块注释中会被忽略。
// 同样，如果字符串/*出现在行或块注释中也会被忽略。
// 如果一行在删除注释之后变为空字符串，那么不要输出该行。即，答案列表中的每个字符串都是非空的。
//
// 样例中没有控制字符，单引号或双引号字符。
//
// 比如，source = "string s = "/* Not a comment. */";" 不会出现在测试样例里。
// 此外，没有其他内容（如定义或宏）会干扰注释。
//
// 我们保证每一个块注释最终都会被闭合， 所以在行或块注释之外的/*总是开始新的注释。
//
// 最后，隐式换行符可以通过块注释删除。 有关详细信息，请参阅下面的示例。
//
// 从源代码中删除注释后，需要以相同的格式返回源代码。
//
// 示例 1:
// 输入: source = ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", "   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]
// 输出: ["int main()","{ ","  ","int a, b, c;","a = b + c;","}"]
// 解释: 示例代码可以编排成这样:
// /*Test program */
// int main()
//
//	{
//	 // variable declaration
//
// int a, b, c;
// /* This is a test
//
//	multiline
//	comment for
//	testing */
//
// a = b + c;
// }
// 第 1 行和第 6-9 行的字符串 /* 表示块注释。第 4 行的字符串 // 表示行注释。
// 编排后:
// int main()
// {
// int a, b, c;
// a = b + c;
// }
//
// 示例 2:
// 输入: source = ["a/*comment", "line", "more_comment*/b"]
// 输出: ["ab"]
// 解释: 原始的 source 字符串是 "a/*comment\nline\nmore_comment*/b", 其中我们用粗体显示了换行符。删除注释后，隐含的换行符被删除，留下字符串 "ab" 用换行符分隔成数组时就是 ["ab"].
//
// 提示:
// 1 <= source.length <= 100
// 0 <= source[i].length <= 80
// source[i] 由可打印的 ASCII 字符组成。
// 每个块注释都会被闭合。
// 给定的源码中不会有单引号、双引号或其他控制字符。
func removeComments(source []string) []string {
	result := make([]string, 0)
	inBlock := false
	var text strings.Builder
	for _, line := range source {
		n := len(line)
		if !inBlock {
			text.Reset()
		}
		i := 0
		for i < n {
			if !inBlock && i+1 < n && line[i] == '/' && line[i+1] == '*' {
				inBlock = true
				i++
			} else if inBlock && i+1 < n && line[i] == '*' && line[i+1] == '/' {
				inBlock = false
				i++
			} else if !inBlock && i+1 < n && line[i] == '/' && line[i+1] == '/' {
				break
			} else if !inBlock {
				text.WriteByte(line[i])
			}
			i++
		}
		if !inBlock && text.Len() > 0 {
			result = append(result, text.String())
		}
	}

	return result
}

// 2024. 考试的最大困扰度
// 一位老师正在出一场由 n 道判断题构成的考试，每道题的答案为 true （用 'T' 表示）或者 false （用 'F' 表示）。老师想增加学生对自己做出答案的不确定性，方法是 最大化 有 连续相同 结果的题数。（也就是连续出现 true 或者连续出现 false）。
//
// 给你一个字符串 answerKey ，其中 answerKey[i] 是第 i 个问题的正确结果。除此以外，还给你一个整数 k ，表示你能进行以下操作的最多次数：
//
// 每次操作中，将问题的正确答案改为 'T' 或者 'F' （也就是将 answerKey[i] 改为 'T' 或者 'F' ）。
// 请你返回在不超过 k 次操作的情况下，最大 连续 'T' 或者 'F' 的数目。
//
// 示例 1：
// 输入：answerKey = "TTFF", k = 2
// 输出：4
// 解释：我们可以将两个 'F' 都变为 'T' ，得到 answerKey = "TTTT" 。
// 总共有四个连续的 'T' 。
//
// 示例 2：
// 输入：answerKey = "TFFT", k = 1
// 输出：3
// 解释：我们可以将最前面的 'T' 换成 'F' ，得到 answerKey = "FFFT" 。
// 或者，我们可以将第二个 'T' 换成 'F' ，得到 answerKey = "TFFF" 。
// 两种情况下，都有三个连续的 'F' 。
//
// 示例 3：
// 输入：answerKey = "TTFTTFTT", k = 1
// 输出：5
// 解释：我们可以将第一个 'F' 换成 'T' ，得到 answerKey = "TTTTTFTT" 。
// 或者我们可以将第二个 'F' 换成 'T' ，得到 answerKey = "TTFTTTTT" 。
// 两种情况下，都有五个连续的 'T' 。
//
// 提示：
// n == answerKey.length
// 1 <= n <= 5 * 104
// answerKey[i] 要么是 'T' ，要么是 'F'
// 1 <= k <= n
func maxConsecutiveAnswers(answerKey string, k int) int {
	n := len(answerKey)
	count, maxVal := 0, 0
	left, right := 0, 0
	// 滑动窗口
	for ; right < n; right++ {
		if answerKey[right] == 'F' {
			count++
		}
		for count > k {
			if answerKey[left] == 'F' {
				count--
			}
			left++
		}
		maxVal = max(maxVal, right+1-left)
	}
	left, right = 0, 0
	count = 0
	for ; right < n; right++ {
		if answerKey[right] == 'T' {
			count++
		}
		for count > k {
			if answerKey[left] == 'T' {
				count--
			}
			left++
		}
		maxVal = max(maxVal, right+1-left)
	}

	return maxVal
}

// 796. 旋转字符串
// 给定两个字符串, s 和 goal。如果在若干次旋转操作之后，s 能变成 goal ，那么返回 true 。
//
// s 的 旋转操作 就是将 s 最左边的字符移动到最右边。
//
// 例如, 若 s = 'abcde'，在旋转一次之后结果就是'bcdea' 。
//
// 示例 1:
// 输入: s = "abcde", goal = "cdeab"
// 输出: true
//
// 示例 2:
// 输入: s = "abcde", goal = "abced"
// 输出: false
//
// 提示:
// 1 <= s.length, goal.length <= 100
// s 和 goal 由小写英文字母组成
func rotateString(s string, goal string) bool {
	m, n := len(s), len(goal)
	if m != n {
		return false
	}
	if s == goal {
		return true
	}
	letters := [26]int{}
	for i := 0; i < n; i++ {
		c1, c2 := s[i], goal[i]
		letters[c1-'a']++
		letters[c2-'a']--
	}
	for i := 0; i < 26; i++ {
		if letters[i] != 0 {
			return false
		}
	}
	indexs := make([]int, 0)

	for i := 0; i < n; i++ {
		last := i - 1
		if i == 0 {
			last = n - 1
		}
		if goal[i] == s[0] && goal[last] == s[n-1] {
			indexs = append(indexs, i)
		}
	}
	match := func(idx int) bool {
		for i := 0; i < n; i++ {
			if s[i] != goal[idx] {
				return false
			}
			idx++
			if idx == n {
				idx = 0
			}
		}
		return true
	}

	for _, idx := range indexs {
		if match(idx) {
			return true
		}
	}

	return false
}

// 806. 写字符串需要的行数
// 我们要把给定的字符串 S 从左到右写到每一行上，每一行的最大宽度为100个单位，如果我们在写某个字母的时候会使这行超过了100 个单位，那么我们应该把这个字母写到下一行。我们给定了一个数组 widths ，这个数组 widths[0] 代表 'a' 需要的单位， widths[1] 代表 'b' 需要的单位，...， widths[25] 代表 'z' 需要的单位。
//
// 现在回答两个问题：至少多少行能放下S，以及最后一行使用的宽度是多少个单位？将你的答案作为长度为2的整数列表返回。
//
// 示例 1:
// 输入:
// widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
// S = "abcdefghijklmnopqrstuvwxyz"
// 输出: [3, 60]
// 解释:
// 所有的字符拥有相同的占用单位10。所以书写所有的26个字母，
// 我们需要2个整行和占用60个单位的一行。
//
// 示例 2:
// 输入:
// widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
// S = "bbbcccdddaaa"
// 输出: [2, 4]
// 解释:
// 除去字母'a'所有的字符都是相同的单位10，并且字符串 "bbbcccdddaa" 将会覆盖 9 * 10 + 2 * 4 = 98 个单位.
// 最后一个字母 'a' 将会被写到第二行，因为第一行只剩下2个单位了。
// 所以，这个答案是2行，第二行有4个单位宽度。
//
// 注:
// 字符串 S 的长度在 [1, 1000] 的范围。
// S 只包含小写字母。
// widths 是长度为 26的数组。
// widths[i] 值的范围在 [2, 10]。
func numberOfLines(widths []int, s string) []int {
	rows, count := 1, 0
	for _, c := range s {
		width := widths[c-'a']
		if width+count > 100 {
			rows++
			count = width
		} else {
			count += width
		}
	}

	return []int{rows, count}
}

// 824. 山羊拉丁文
// 给你一个由若干单词组成的句子 sentence ，单词间由空格分隔。每个单词仅由大写和小写英文字母组成。
//
// 请你将句子转换为 “山羊拉丁文（Goat Latin）”（一种类似于 猪拉丁文 - Pig Latin 的虚构语言）。山羊拉丁文的规则如下：
//
// 如果单词以元音开头（'a', 'e', 'i', 'o', 'u'），在单词后添加"ma"。
// 例如，单词 "apple" 变为 "applema" 。
// 如果单词以辅音字母开头（即，非元音字母），移除第一个字符并将它放到末尾，之后再添加"ma"。
// 例如，单词 "goat" 变为 "oatgma" 。
// 根据单词在句子中的索引，在单词最后添加与索引相同数量的字母'a'，索引从 1 开始。
// 例如，在第一个单词后添加 "a" ，在第二个单词后添加 "aa" ，以此类推。
// 返回将 sentence 转换为山羊拉丁文后的句子。
//
// 示例 1：
// 输入：sentence = "I speak Goat Latin"
// 输出："Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
//
// 示例 2：
// 输入：sentence = "The quick brown fox jumped over the lazy dog"
// 输出："heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"
//
// 提示：
// 1 <= sentence.length <= 150
// sentence 由英文字母和空格组成
// sentence 不含前导或尾随空格
// sentence 中的所有单词由单个空格分隔
func toGoatLatin(sentence string) string {
	vowel := func(c byte) bool {
		switch c {
		case 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U':
			return true
		}
		return false
	}
	words := strings.Split(sentence, " ")
	var builder strings.Builder
	for i, word := range words {
		c := word[0]
		if vowel(c) {
			builder.WriteString(word)
		} else {
			builder.WriteString(word[1:])
			builder.WriteByte(c)
		}
		builder.WriteString("ma")
		for j := 0; j <= i; j++ {
			builder.WriteByte('a')
		}
		if i < len(words)-1 {
			builder.WriteByte(' ')
		}
	}
	return builder.String()
}

// 944. 删列造序
// 给你由 n 个小写字母字符串组成的数组 strs，其中每个字符串长度相等。
//
// 这些字符串可以每个一行，排成一个网格。例如，strs = ["abc", "bce", "cae"] 可以排列为：
// abc
// bce
// cae
// 你需要找出并删除 不是按字典序升序排列的 列。在上面的例子（下标从 0 开始）中，列 0（'a', 'b', 'c'）和列 2（'c', 'e', 'e'）都是按升序排列的，而列 1（'b', 'c', 'a'）不是，所以要删除列 1 。
//
// 返回你需要删除的列数。
//
// 示例 1：
// 输入：strs = ["cba","daf","ghi"]
// 输出：1
// 解释：网格示意如下：
//
//	cba
//	daf
//	ghi
//
// 列 0 和列 2 按升序排列，但列 1 不是，所以只需要删除列 1 。
//
// 示例 2：
// 输入：strs = ["a","b"]
// 输出：0
// 解释：网格示意如下：
//
//	a
//	b
//
// 只有列 0 这一列，且已经按升序排列，所以不用删除任何列。
//
// 示例 3：
// 输入：strs = ["zyx","wvu","tsr"]
// 输出：3
// 解释：网格示意如下：
//
//	zyx
//	wvu
//	tsr
//
// 所有 3 列都是非升序排列的，所以都要删除。
//
// 提示：
// n == strs.length
// 1 <= n <= 100
// 1 <= strs[i].length <= 1000
// strs[i] 由小写英文字母组成
func minDeletionSize(strs []string) int {
	rows, cols := len(strs), len(strs[0])

	result := 0
	for j := 0; j < cols; j++ {
		for i := 1; i < rows; i++ {
			if strs[i-1][j] > strs[i][j] {
				result++
				break
			}
		}
	}

	return result
}

// 1108. IP 地址无效化
// 给你一个有效的 IPv4 地址 address，返回这个 IP 地址的无效化版本。
//
// 所谓无效化 IP 地址，其实就是用 "[.]" 代替了每个 "."。
//
// 示例 1：
// 输入：address = "1.1.1.1"
// 输出："1[.]1[.]1[.]1"
//
// 示例 2：
// 输入：address = "255.100.50.0"
// 输出："255[.]100[.]50[.]0"
//
// 提示：
// 给出的 address 是一个有效的 IPv4 地址
func defangIPaddr(address string) string {
	return strings.ReplaceAll(address, ".", "[.]")
}

// 1374. 生成每种字符都是奇数个的字符串
// 给你一个整数 n，请你返回一个含 n 个字符的字符串，其中每种字符在该字符串中都恰好出现 奇数次 。
//
// 返回的字符串必须只含小写英文字母。如果存在多个满足题目要求的字符串，则返回其中任意一个即可。
//
// 示例 1：
// 输入：n = 4
// 输出："pppz"
// 解释："pppz" 是一个满足题目要求的字符串，因为 'p' 出现 3 次，且 'z' 出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ohhh" 和 "love"。
//
// 示例 2：
// 输入：n = 2
// 输出："xy"
// 解释："xy" 是一个满足题目要求的字符串，因为 'x' 和 'y' 各出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ag" 和 "ur"。
//
// 示例 3：
// 输入：n = 7
// 输出："holasss"
//
// 提示：
// 1 <= n <= 500
func generateTheString(n int) string {
	var builder strings.Builder

	if n&1 == 0 {
		// 偶数
		builder.WriteByte('a')
		n--
	}
	for i := 0; i < n; i++ {
		builder.WriteByte('b')
	}

	return builder.String()
}

// 899. 有序队列
// 给定一个字符串 s 和一个整数 k 。你可以从 s 的前 k 个字母中选择一个，并把它加到字符串的末尾。
//
// 返回 在应用上述步骤的任意数量的移动后，字典上最小的字符串 。
//
// 示例 1：
// 输入：s = "cba", k = 1
// 输出："acb"
// 解释：
// 在第一步中，我们将第一个字符（“c”）移动到最后，获得字符串 “bac”。
// 在第二步中，我们将第一个字符（“b”）移动到最后，获得最终结果 “acb”。
//
// 示例 2：
// 输入：s = "baaca", k = 3
// 输出："aaabc"
// 解释：
// 在第一步中，我们将第一个字符（“b”）移动到最后，获得字符串 “aacab”。
// 在第二步中，我们将第三个字符（“c”）移动到最后，获得最终结果 “aaabc”。
//
// 提示：
// 1 <= k <= S.length <= 1000
// s 只由小写字母组成。
func orderlyQueue(s string, k int) string {
	n := len(s)
	if k == 1 {
		result := s
		for i := 1; i < n; i++ {
			str := s[i:] + s[:i]
			if str < result {
				result = str
			}
		}
		return result
	}
	t := []byte(s)
	sort.Slice(t, func(i, j int) bool {
		return t[i] < t[j]
	})
	return string(t)
}

// 1408. 数组中的字符串匹配
// 给你一个字符串数组 words ，数组中的每个字符串都可以看作是一个单词。请你按 任意 顺序返回 words 中是其他单词的子字符串的所有单词。
//
// 如果你可以删除 words[j] 最左侧和/或最右侧的若干字符得到 word[i] ，那么字符串 words[i] 就是 words[j] 的一个子字符串。
//
// 示例 1：
// 输入：words = ["mass","as","hero","superhero"]
// 输出：["as","hero"]
// 解释："as" 是 "mass" 的子字符串，"hero" 是 "superhero" 的子字符串。
// ["hero","as"] 也是有效的答案。
//
// 示例 2：
// 输入：words = ["leetcode","et","code"]
// 输出：["et","code"]
// 解释："et" 和 "code" 都是 "leetcode" 的子字符串。
//
// 示例 3：
// 输入：words = ["blue","green","bu"]
// 输出：[]
//
// 提示：
// 1 <= words.length <= 100
// 1 <= words[i].length <= 30
// words[i] 仅包含小写英文字母。
// 题目数据 保证 每个 words[i] 都是独一无二的。
func stringMatching(words []string) []string {
	sort.Slice(words, func(i, j int) bool {
		return len(words[i]) < len(words[j])
	})
	n := len(words)
	result := make([]string, 0)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if strings.Contains(words[j], words[i]) {
				result = append(result, words[i])
				break
			}
		}
	}
	return result
}

// 761. 特殊的二进制序列
// 特殊的二进制序列是具有以下两个性质的二进制序列：
//
// 0 的数量与 1 的数量相等。
// 二进制序列的每一个前缀码中 1 的数量要大于等于 0 的数量。
// 给定一个特殊的二进制序列 S，以字符串形式表示。定义一个操作 为首先选择 S 的两个连续且非空的特殊的子串，然后将它们交换。（两个子串为连续的当且仅当第一个子串的最后一个字符恰好为第二个子串的第一个字符的前一个字符。)
//
// 在任意次数的操作之后，交换后的字符串按照字典序排列的最大的结果是什么？
//
// 示例 1:
// 输入: S = "11011000"
// 输出: "11100100"
// 解释:
// 将子串 "10" （在S[1]出现） 和 "1100" （在S[3]出现）进行交换。
// 这是在进行若干次操作后按字典序排列最大的结果。
//
// 说明:
// S 的长度不超过 50。
// S 保证为一个满足上述定义的特殊 的二进制序列。
func makeLargestSpecial(s string) string {
	n := len(s)
	if n == 0 {
		return ""
	}
	strs := make([]string, 0)
	count, start := 0, 0
	for i := 0; i < n; i++ {
		if s[i] == '1' {
			count++
		} else {
			count--
		}
		if count == 0 {
			str := "1" + makeLargestSpecial(s[start+1:i]) + "0"
			strs = append(strs, str)
			start = i + 1
		}
	}
	sort.Slice(strs, func(i, j int) bool {
		return strs[i] > strs[j]
	})
	var builder strings.Builder
	for _, str := range strs {
		builder.WriteString(str)
	}

	return builder.String()
}

// 1417. 重新格式化字符串
// 给你一个混合了数字和字母的字符串 s，其中的字母均为小写英文字母。
//
// 请你将该字符串重新格式化，使得任意两个相邻字符的类型都不同。也就是说，字母后面应该跟着数字，而数字后面应该跟着字母。
//
// 请你返回 重新格式化后 的字符串；如果无法按要求重新格式化，则返回一个 空字符串 。
//
// 示例 1：
// 输入：s = "a0b1c2"
// 输出："0a1b2c"
// 解释："0a1b2c" 中任意两个相邻字符的类型都不同。 "a0b1c2", "0a1b2c", "0c2a1b" 也是满足题目要求的答案。
//
// 示例 2：
// 输入：s = "leetcode"
// 输出：""
// 解释："leetcode" 中只有字母，所以无法满足重新格式化的条件。
//
// 示例 3：
// 输入：s = "1229857369"
// 输出：""
// 解释："1229857369" 中只有数字，所以无法满足重新格式化的条件。
//
// 示例 4：
// 输入：s = "covid2019"
// 输出："c2o0v1i9d"
//
// 示例 5：
// 输入：s = "ab123"
// 输出："1a2b3"
//
// 提示：
// 1 <= s.length <= 500
// s 仅由小写英文字母和/或数字组成。
func reformat(s string) string {
	n := len(s)
	if n == 1 {
		return s
	}
	nums, letters := make([]byte, n), make([]byte, n)
	count1, count2 := 0, 0
	for i := range s {
		if s[i] >= '0' && s[i] <= '9' {
			nums[count1] = s[i]
			count1++
		} else {
			letters[count2] = s[i]
			count2++
		}
	}
	if abs(count1-count2) > 1 {
		return ""
	}
	result := make([]byte, n)
	i, j, index := 0, 0, 0
	flag := count1 > count2

	for i < count1 || j < count2 {
		if flag {
			result[index] = nums[i]
			i++
		} else {
			result[index] = letters[j]
			j++
		}
		index++
		flag = !flag
	}

	return string(result)
}

// 756. 金字塔转换矩阵
// 你正在把积木堆成金字塔。每个块都有一个颜色，用一个字母表示。每一行的块比它下面的行 少一个块 ，并且居中。
//
// 为了使金字塔美观，只有特定的 三角形图案 是允许的。一个三角形的图案由 两个块 和叠在上面的 单个块 组成。模式是以三个字母字符串的列表形式 allowed 给出的，其中模式的前两个字符分别表示左右底部块，第三个字符表示顶部块。
//
// 例如，"ABC" 表示一个三角形图案，其中一个 “C” 块堆叠在一个 'A' 块(左)和一个 'B' 块(右)之上。请注意，这与 "BAC" 不同，"B" 在左下角，"A" 在右下角。
// 你从底部的一排积木 bottom 开始，作为一个单一的字符串，你 必须 使用作为金字塔的底部。
//
// 在给定 bottom 和 allowed 的情况下，如果你能一直构建到金字塔顶部，使金字塔中的 每个三角形图案 都是允许的，则返回 true ，否则返回 false 。
//
// 示例 1：
// 输入：bottom = "BCD", allowed = ["BCG", "CDE", "GEA", "FFF"]
// 输出：true
// 解释：允许的三角形模式显示在右边。
// 从最底层(第3层)开始，我们可以在第2层构建“CE”，然后在第1层构建“E”。
// 金字塔中有三种三角形图案，分别是“BCC”、“CDE”和“CEA”。都是允许的。
//
// 示例 2：
// 输入：bottom = "AABA", allowed = ["AAA", "AAB", "ABA", "ABB", "BAC"]
// 输出：false
// 解释：允许的三角形模式显示在右边。
// 从最底层(游戏邦注:即第4个关卡)开始，创造第3个关卡有多种方法，但如果尝试所有可能性，你便会在创造第1个关卡前陷入困境。
//
// 提示：
// 2 <= bottom.length <= 6
// 0 <= allowed.length <= 216
// allowed[i].length == 3
// 所有输入字符串中的字母来自集合 {'A', 'B', 'C', 'D', 'E', 'F', 'G'}。
// allowed 中所有值都是 唯一的

func pyramidTransition(bottom string, allowed []string) bool {
	allowMap := make([][]int, 7)
	for i := 0; i < 7; i++ {
		allowMap[i] = make([]int, 7)
	}
	for _, allow := range allowed {
		c1, c2, c3 := allow[0], allow[1], allow[2]
		allowMap[c1-'A'][c2-'A'] |= 1 << (c3 - 'A')
	}
	n := len(bottom)
	pyramid := make([][]int, n)
	for i := 0; i < n; i++ {
		pyramid[i] = make([]int, n)
	}
	for i := 0; i < n; i++ {
		pyramid[n-1][i] = int(bottom[i] - 'A')
	}
	visited := make(map[int64]bool)
	// 深度优先遍历
	var dfs func(row, col int, num int64) bool
	dfs = func(row, col int, num int64) bool {
		fmt.Printf("r:%d c:%d num:%d", row, col, num)
		fmt.Println()
		if row == 1 && col == 1 {
			return true
		} else if row == col {
			if visited[num] {
				return false
			}
			visited[num] = true
			// 上一层
			return dfs(row-1, 0, 0)
		} else {
			c1, c2 := pyramid[row][col], pyramid[row][col+1]
			w := allowMap[c1][c2]
			// 所有 允许的上层
			for i := 0; i < 7; i++ {
				if w&(1<<i) != 0 {
					// 上一层
					pyramid[row-1][col] = i
					if dfs(row, col+1, num*8+int64(i)+1) {
						return true
					}
				}
			}
			return false
		}

	}

	return dfs(n-1, 0, 0)
}

// 767. 重构字符串
// 给定一个字符串 s ，检查是否能重新排布其中的字母，使得两相邻的字符不同。
//
// 返回 s 的任意可能的重新排列。若不可行，返回空字符串 "" 。
//
// 示例 1:
// 输入: s = "aab"
// 输出: "aba"
//
// 示例 2:
// 输入: s = "aaab"
// 输出: ""
//
// 提示:
// 1 <= s.length <= 500
// s 只包含小写字母
func reorganizeString(s string) string {
	n := len(s)
	half := (n + 1) >> 1
	letters := make([]int, 26)
	for i := 0; i < 26; i++ {
		letters[i] = i
	}
	for _, c := range s {
		letters[c-'a'] += 100
	}
	sort.Ints(letters)
	index := 0
	result := make([]byte, n)
	for i := 25; i >= 0; i-- {
		count, c := letters[i]/100, byte('a'+letters[i]%100)
		if count == 0 {
			continue
		}
		if count > half {
			// 超过一半 不可行
			return ""
		}
		for count > 0 {
			// 先放置 偶数位  然后放置奇数索引
			if index >= n {
				index = 1
			}
			result[index] = c
			count--
			index += 2
		}
	}

	return string(result)
}

// 777. 在LR字符串中交换相邻字符
// 在一个由 'L' , 'R' 和 'X' 三个字符组成的字符串（例如"RXXLRXRXL"）中进行移动操作。
// 一次移动操作指用一个"LX"替换一个"XL"，或者用一个"XR"替换一个"RX"。现给定起始字符串start和结束字符串end，请编写代码，当且仅当存在一系列移动操作使得start可以转换成end时， 返回True。
//
// 示例 :
// 输入: start = "RXXLRXRXL", end = "XRLXXRRLX"
// 输出: True
// 解释:
// 我们可以通过以下几步将start转换成end:
// RXXLRXRXL ->
// XRXLRXRXL ->
// XRLXRXRXL ->
// XRLXXRRXL ->
// XRLXXRRLX
//
// 提示：
// 1 <= len(start) = len(end) <= 10000。
// start和end中的字符串仅限于'L', 'R'和'X'。
func canTransform(start string, end string) bool {
	// 有 X 才能移动 移除X后 两个字符串相同
	if strings.ReplaceAll(start, "X", "") != strings.ReplaceAll(end, "X", "") {
		return false
	}
	j := 0
	for i, c := range start {
		if c != 'X' {
			// c = L 或者 R
			for end[j] == 'X' {
				j++
			}
			if i != j && c == 'L' != (i > j) {
				return false
			}
			j++
		}
	}
	return true
}

// 828. 统计子串中的唯一字符
// 我们定义了一个函数 countUniqueChars(s) 来统计字符串 s 中的唯一字符，并返回唯一字符的个数。
//
// 例如：s = "LEETCODE" ，则其中 "L", "T","C","O","D" 都是唯一字符，因为它们只出现一次，所以 countUniqueChars(s) = 5 。
//
// 本题将会给你一个字符串 s ，我们需要返回 countUniqueChars(t) 的总和，其中 t 是 s 的子字符串。输入用例保证返回值为 32 位整数。
//
// 注意，某些子字符串可能是重复的，但你统计时也必须算上这些重复的子字符串（也就是说，你必须统计 s 的所有子字符串中的唯一字符）。
//
// 示例 1：
// 输入: s = "ABC"
// 输出: 10
// 解释: 所有可能的子串为："A","B","C","AB","BC" 和 "ABC"。
//
//	其中，每一个子串都由独特字符构成。
//	所以其长度总和为：1 + 1 + 1 + 2 + 2 + 3 = 10
//
// 示例 2：
// 输入: s = "ABA"
// 输出: 8
// 解释: 除了 countUniqueChars("ABA") = 1 之外，其余与示例 1 相同。
//
// 示例 3：
// 输入：s = "LEETCODE"
// 输出：92
//
// 提示：
// 1 <= s.length <= 10^5
// s 只包含大写英文字符
func uniqueLetterString(s string) int {
	// "LEETCODE"  len=8
	// 对于字符'L'，在区间[0,7]只出现一次，为答案贡献8(在该区间中,'L'可以存在于8个子串中)
	// 对于字符'E'，在区间[0,1]只出现一次，为答案贡献2
	// 对于字符'E'，在区间[2,6]只出现一次，为答案贡献5
	// 对于字符'T'，在区间[0,7]只出现一次，为答案贡献20 左边 4 右边5
	// 对于字符'C'，在区间[0,7]只出现一次，为答案贡献20 左边 5 右边4
	// 对于字符'O'，在区间[0,7]只出现一次，为答案贡献18
	// 对于字符'D'，在区间[0,7]只出现一次，为答案贡献14
	// 对于字符'E'，在区间[3,7]只出现一次，为答案贡献5
	// ans=8+2+5+20+20+18+14+5=92
	// 以每个字符为中心，向两边扩展到不重复为止
	n := len(s)
	result := 0
	for i := 0; i < n; i++ {
		c := s[i]
		left, right := i-1, i+1
		for left >= 0 && s[left] != c {
			left--
		}
		for right < n && s[right] != c {
			right++
		}
		result += (i - left) * (right - i)
	}

	return result
}

// 1592. 重新排列单词间的空格
// 给你一个字符串 text ，该字符串由若干被空格包围的单词组成。每个单词由一个或者多个小写英文字母组成，并且两个单词之间至少存在一个空格。题目测试用例保证 text 至少包含一个单词 。
//
// 请你重新排列空格，使每对相邻单词之间的空格数目都 相等 ，并尽可能 最大化 该数目。如果不能重新平均分配所有空格，请 将多余的空格放置在字符串末尾 ，这也意味着返回的字符串应当与原 text 字符串的长度相等。
// 返回 重新排列空格后的字符串 。
//
// 示例 1：
// 输入：text = "  this   is  a sentence "
// 输出："this   is   a   sentence"
// 解释：总共有 9 个空格和 4 个单词。可以将 9 个空格平均分配到相邻单词之间，相邻单词间空格数为：9 / (4-1) = 3 个。
//
// 示例 2：
// 输入：text = " practice   makes   perfect"
// 输出："practice   makes   perfect "
// 解释：总共有 7 个空格和 3 个单词。7 / (3-1) = 3 个空格加上 1 个多余的空格。多余的空格需要放在字符串的末尾。
//
// 示例 3：
// 输入：text = "hello   world"
// 输出："hello   world"
//
// 示例 4：
// 输入：text = "  walks  udp package   into  bar a"
// 输出："walks  udp  package  into  bar  a "
//
// 示例 5：
// 输入：text = "a"
// 输出："a"
//
// 提示：
// 1 <= text.length <= 100
// text 由小写英文字母和 ' ' 组成
// text 中至少包含一个单词
func reorderSpaces(text string) string {
	count := 0
	for _, c := range text {
		if c == ' ' {
			count++
		}
	}
	strs := strings.Split(text, " ")

	words := make([]string, 0)
	for _, str := range strs {
		if len(str) > 0 {
			words = append(words, str)
		}
	}
	n := len(words)
	var num int
	if n == 1 {
		num = 0
	} else {
		num = count / (n - 1)
	}
	left := count - num*(n-1)
	var builder strings.Builder
	for i, word := range words {
		builder.WriteString(word)
		var blankSize int
		if i == n-1 {
			blankSize = left
		} else {
			blankSize = num
		}
		for j := 0; j < blankSize; j++ {
			builder.WriteByte(' ')
		}
	}

	return builder.String()
}

// 1598. 文件夹操作日志搜集器
// 每当用户执行变更文件夹操作时，LeetCode 文件系统都会保存一条日志记录。
//
// 下面给出对变更操作的说明：
//
// "../" ：移动到当前文件夹的父文件夹。如果已经在主文件夹下，则 继续停留在当前文件夹 。
// "./" ：继续停留在当前文件夹。
// "x/" ：移动到名为 x 的子文件夹中。题目数据 保证总是存在文件夹 x 。
// 给你一个字符串列表 logs ，其中 logs[i] 是用户在 ith 步执行的操作。
//
// 文件系统启动时位于主文件夹，然后执行 logs 中的操作。
//
// 执行完所有变更文件夹操作后，请你找出 返回主文件夹所需的最小步数 。
//
// 示例 1：
// 输入：logs = ["d1/","d2/","../","d21/","./"]
// 输出：2
// 解释：执行 "../" 操作变更文件夹 2 次，即可回到主文件夹
//
// 示例 2：
// 输入：logs = ["d1/","d2/","./","d3/","../","d31/"]
// 输出：3
//
// 示例 3：
// 输入：logs = ["d1/","../","../","../"]
// 输出：0
//
// 提示：
// 1 <= logs.length <= 103
// 2 <= logs[i].length <= 10
// logs[i] 包含小写英文字母，数字，'.' 和 '/'
// logs[i] 符合语句中描述的格式
// 文件夹名称由小写英文字母和数字组成
func minOperations(logs []string) int {
	num := 0

	for _, log := range logs {
		if log == "../" {
			if num > 0 {
				num--
			}
		} else if log != "./" {
			num++
		}
	}

	return abs(num)
}

// 1624. 两个相同字符之间的最长子字符串
// 给你一个字符串 s，请你返回 两个相同字符之间的最长子字符串的长度 ，计算长度时不含这两个字符。如果不存在这样的子字符串，返回 -1 。
// 子字符串 是字符串中的一个连续字符序列。
//
// 示例 1：
// 输入：s = "aa"
// 输出：0
// 解释：最优的子字符串是两个 'a' 之间的空子字符串。
//
// 示例 2：
// 输入：s = "abca"
// 输出：2
// 解释：最优的子字符串是 "bc" 。
//
// 示例 3：
// 输入：s = "cbzxy"
// 输出：-1
// 解释：s 中不存在出现出现两次的字符，所以返回 -1 。
//
// 示例 4：
// 输入：s = "cabbac"
// 输出：4
// 解释：最优的子字符串是 "abba" ，其他的非最优解包括 "bb" 和 "" 。
//
// 提示：
// 1 <= s.length <= 300
// s 只含小写英文字母
func maxLengthBetweenEqualCharacters(s string) int {
	letterIndex := make([]int, 26)
	for i := 0; i < 26; i++ {
		letterIndex[i] = -1
	}
	result := -1
	for i, c := range s {
		if letterIndex[c-'a'] == -1 {
			letterIndex[c-'a'] = i
		} else {
			result = max(result, i-letterIndex[c-'a']-1)
		}
	}
	return result
}

// 面试题 01.02. 判定是否互为字符重排
// 给定两个字符串 s1 和 s2，请编写一个程序，确定其中一个字符串的字符重新排列后，能否变成另一个字符串。
//
// 示例 1：
// 输入: s1 = "abc", s2 = "bca"
// 输出: true
//
// 示例 2：
// 输入: s1 = "abc", s2 = "bad"
// 输出: false
//
// 说明：
// 0 <= len(s1) <= 100
// 0 <= len(s2) <= 100
func checkPermutation(s1 string, s2 string) bool {
	letters := make([]int, 26)
	for _, c := range s1 {
		letters[c-'a']++
	}
	for _, c := range s2 {
		letters[c-'a']--
	}
	for i := 0; i < 26; i++ {
		if letters[i] != 0 {
			return false
		}
	}

	return true
}

// 面试题 01.09. 字符串轮转
// 字符串轮转。给定两个字符串s1和s2，请编写代码检查s2是否为s1旋转而成（比如，waterbottle是erbottlewat旋转后的字符串）。
//
// 示例1:
// 输入：s1 = "waterbottle", s2 = "erbottlewat"
// 输出：True
//
// 示例2:
// 输入：s1 = "aa", s2 = "aba"
// 输出：False
//
// 提示：
// 字符串长度在[0, 100000]范围内。
// 说明:
// 你能只调用一次检查子串的方法吗？
func isFlipedString(s1 string, s2 string) bool {
	m, n := len(s1), len(s2)
	if m != n {
		return false
	}

	return strings.Contains(s1+s1, s2)
}

// 1694. 重新格式化电话号码
// 给你一个字符串形式的电话号码 number 。number 由数字、空格 ' '、和破折号 '-' 组成。
//
// 请你按下述方式重新格式化电话号码。
// 首先，删除 所有的空格和破折号。
// 其次，将数组从左到右 每 3 个一组 分块，直到 剩下 4 个或更少数字。剩下的数字将按下述规定再分块：
// 2 个数字：单个含 2 个数字的块。
// 3 个数字：单个含 3 个数字的块。
// 4 个数字：两个分别含 2 个数字的块。
// 最后用破折号将这些块连接起来。注意，重新格式化过程中 不应该 生成仅含 1 个数字的块，并且 最多 生成两个含 2 个数字的块。
//
// 返回格式化后的电话号码。
//
// 示例 1：
// 输入：number = "1-23-45 6"
// 输出："123-456"
// 解释：数字是 "123456"
// 步骤 1：共有超过 4 个数字，所以先取 3 个数字分为一组。第 1 个块是 "123" 。
// 步骤 2：剩下 3 个数字，将它们放入单个含 3 个数字的块。第 2 个块是 "456" 。
// 连接这些块后得到 "123-456" 。
//
// 示例 2：
// 输入：number = "123 4-567"
// 输出："123-45-67"
// 解释：数字是 "1234567".
// 步骤 1：共有超过 4 个数字，所以先取 3 个数字分为一组。第 1 个块是 "123" 。
// 步骤 2：剩下 4 个数字，所以将它们分成两个含 2 个数字的块。这 2 块分别是 "45" 和 "67" 。
// 连接这些块后得到 "123-45-67" 。
//
// 示例 3：
// 输入：number = "123 4-5678"
// 输出："123-456-78"
// 解释：数字是 "12345678" 。
// 步骤 1：第 1 个块 "123" 。
// 步骤 2：第 2 个块 "456" 。
// 步骤 3：剩下 2 个数字，将它们放入单个含 2 个数字的块。第 3 个块是 "78" 。
// 连接这些块后得到 "123-456-78" 。
//
// 示例 4：
// 输入：number = "12"
// 输出："12"
//
// 示例 5：
// 输入：number = "--17-5 229 35-39475 "
// 输出："175-229-353-94-75"
//
// 提示：
// 2 <= number.length <= 100
// number 由数字和字符 '-' 及 ' ' 组成。
// number 中至少含 2 个数字。
func reformatNumber(number string) string {
	bytes := make([]byte, 0)
	for i := range number {
		if number[i] >= '0' && number[i] <= '9' {
			bytes = append(bytes, number[i])
		}

	}
	n := len(bytes)
	left := n % 3
	if left == 1 {
		left = 4
	}
	var builder strings.Builder
	for i := 0; i < n-left; i += 3 {
		if builder.Len() > 0 {
			builder.WriteByte('-')
		}
		builder.WriteByte(bytes[i])
		builder.WriteByte(bytes[i+1])
		builder.WriteByte(bytes[i+2])
	}
	for i := n - left; i < n; i += 2 {
		if builder.Len() > 0 {
			builder.WriteByte('-')
		}
		builder.WriteByte(bytes[i])
		builder.WriteByte(bytes[i+1])
	}
	return builder.String()
}

// 1784. 检查二进制字符串字段
// 给你一个二进制字符串 s ，该字符串 不含前导零 。
//
// 如果 s 包含 零个或一个由连续的 '1' 组成的字段 ，返回 true​​​ 。否则，返回 false 。
//
// 如果 s 中 由连续若干个 '1' 组成的字段 数量不超过 1，返回 true​​​ 。否则，返回 false 。
//
// 示例 1：
// 输入：s = "1001"
// 输出：false
// 解释：由连续若干个 '1' 组成的字段数量为 2，返回 false
//
// 示例 2：
// 输入：s = "110"
// 输出：true
//
// 提示：
// 1 <= s.length <= 100
// s[i] 为 '0' 或 '1'
// s[0] 为 '1'
func checkOnesSegment(s string) bool {
	n := len(s)
	// 第一个0的位置
	idx := 0
	for idx < n && s[idx] == '1' {
		idx++
	}
	for i := idx; i < n; i++ {
		if s[i] == '1' {
			return false
		}
	}

	return true
}

// 1790. 仅执行一次字符串交换能否使两个字符串相等
// 给你长度相等的两个字符串 s1 和 s2 。一次 字符串交换 操作的步骤如下：选出某个字符串中的两个下标（不必不同），并交换这两个下标所对应的字符。
//
// 如果对 其中一个字符串 执行 最多一次字符串交换 就可以使两个字符串相等，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：s1 = "bank", s2 = "kanb"
// 输出：true
// 解释：例如，交换 s2 中的第一个和最后一个字符可以得到 "bank"
//
// 示例 2：
// 输入：s1 = "attack", s2 = "defend"
// 输出：false
// 解释：一次字符串交换无法使两个字符串相等
//
// 示例 3：
// 输入：s1 = "kelb", s2 = "kelb"
// 输出：true
// 解释：两个字符串已经相等，所以不需要进行字符串交换
//
// 示例 4：
// 输入：s1 = "abcd", s2 = "dcba"
// 输出：false
//
// 提示：
// 1 <= s1.length, s2.length <= 100
// s1.length == s2.length
// s1 和 s2 仅由小写英文字母组成
func areAlmostEqual(s1 string, s2 string) bool {
	letters := make([]int, 26)
	n := len(s1)
	diff := 0
	for i := 0; i < n; i++ {
		if s1[i] == s2[i] {
			continue
		}
		diff++
		letters[s1[i]-'a']++
		letters[s2[i]-'a']--
	}
	if diff == 0 {
		return true
	}
	if diff != 2 {
		return false
	}
	for i := 0; i < 26; i++ {
		if letters[i] != 0 {
			return false
		}
	}

	return true
}

var MOD = 1_000_000_007

// 940. 不同的子序列 II
// 给定一个字符串 s，计算 s 的 不同非空子序列 的个数。因为结果可能很大，所以返回答案需要对 10^9 + 7 取余 。
//
// 字符串的 子序列 是经由原字符串删除一些（也可能不删除）字符但不改变剩余字符相对位置的一个新字符串。
//
// 例如，"ace" 是 "abcde" 的一个子序列，但 "aec" 不是。
//
// 示例 1：
// 输入：s = "abc"
// 输出：7
// 解释：7 个不同的子序列分别是 "a", "b", "c", "ab", "ac", "bc", 以及 "abc"。
//
// 示例 2：
// 输入：s = "aba"
// 输出：6
// 解释：6 个不同的子序列分别是 "a", "b", "ab", "ba", "aa" 以及 "aba"。
//
// 示例 3：
// 输入：s = "aaa"
// 输出：3
// 解释：3 个不同的子序列分别是 "a", "aa" 以及 "aaa"。
//
// 提示：
// 1 <= s.length <= 2000
// s 仅由小写英文字母组成
func distinctSubseqII(s string) int {
	// 思路 根据组合数公式 n个不同的数 组合为 2^n （包括 空集合） 中
	// "a" - "" "a" 2
	// "ab" - "" "a" "b"("" + "b") "ab"("a" + "b")  4
	// "abc" - "" "a" "b" "ab" "c"(""+"c") "ac"("a"+"c") "bc"("b"+"c")"abc"("ab"+"c")  8
	// "aba" - "" "a" "b" "ab" (""+"a" 舍弃) "aa"("a"+"a") "ba"("b"+"a")"aba"("ab"+"a")  7
	// "abaa" - "" "a" "b" "ab" "aa" "ba" "aba"
	//  (""+"a" 舍弃)("a"+"a" 舍弃)("b"+"a" 舍弃)("ab"+"a" 舍弃) "aaa"("aa"+"a")
	// "baa"("b"+"aa")"abaa"("ab"a+"a") 10(7*2-4)
	// num[i+1] = num[i] * 2 - 上一个a的前一个字符的个数
	n := len(s)
	nums := make([]int, n+1)
	nums[0] = 1
	letterIndex := make([]int, 26)
	for i := 0; i < 26; i++ {
		letterIndex[i] = -1
	}
	for i, c := range s {
		index := letterIndex[c-'a']
		nums[i+1] = nums[i] * 2 % MOD
		if index != -1 {
			nums[i+1] -= nums[index]
		}
		nums[i+1] %= MOD
		letterIndex[c-'a'] = i
	}
	result := nums[n] - 1
	if result < 0 {
		result += MOD
	}
	return result
}

// 1768. 交替合并字符串
// 给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。
//
// 返回 合并后的字符串 。
//
// 示例 1：
// 输入：word1 = "abc", word2 = "pqr"
// 输出："apbqcr"
// 解释：字符串合并情况如下所示：
// word1：  a   b   c
// word2：    p   q   r
// 合并后：  a p b q c r
//
// 示例 2：
// 输入：word1 = "ab", word2 = "pqrs"
// 输出："apbqrs"
// 解释：注意，word2 比 word1 长，"rs" 需要追加到合并后字符串的末尾。
// word1：  a   b
// word2：    p   q   r   s
// 合并后：  a p b q   r   s
//
// 示例 3：
// 输入：word1 = "abcd", word2 = "pq"
// 输出："apbqcd"
// 解释：注意，word1 比 word2 长，"cd" 需要追加到合并后字符串的末尾。
// word1：  a   b   c   d
// word2：    p   q
// 合并后：  a p b q c   d
//
// 提示：
// 1 <= word1.length, word2.length <= 100
// word1 和 word2 由小写英文字母组成
func mergeAlternately(word1 string, word2 string) string {
	m, n := len(word1), len(word2)
	var builder strings.Builder
	tmpLen := min(m, n)
	for i := 0; i < tmpLen; i++ {
		builder.WriteByte(word1[i])
		builder.WriteByte(word2[i])
	}
	if m > tmpLen {
		builder.WriteString(word1[tmpLen:])
	}
	if n > tmpLen {
		builder.WriteString(word2[tmpLen:])
	}

	return builder.String()
}

// 1773. 统计匹配检索规则的物品数量
// 给你一个数组 items ，其中 items[i] = [typei, colori, namei] ，描述第 i 件物品的类型、颜色以及名称。
//
// 另给你一条由两个字符串 ruleKey 和 ruleValue 表示的检索规则。
//
// 如果第 i 件物品能满足下述条件之一，则认为该物品与给定的检索规则 匹配 ：
//
// ruleKey == "type" 且 ruleValue == typei 。
// ruleKey == "color" 且 ruleValue == colori 。
// ruleKey == "name" 且 ruleValue == namei 。
// 统计并返回 匹配检索规则的物品数量 。
//
// 示例 1：
// 输入：items = [["phone","blue","pixel"],["computer","silver","lenovo"],["phone","gold","iphone"]], ruleKey = "color", ruleValue = "silver"
// 输出：1
// 解释：只有一件物品匹配检索规则，这件物品是 ["computer","silver","lenovo"] 。
//
// 示例 2：
// 输入：items = [["phone","blue","pixel"],["computer","silver","phone"],["phone","gold","iphone"]], ruleKey = "type", ruleValue = "phone"
// 输出：2
// 解释：只有两件物品匹配检索规则，这两件物品分别是 ["phone","blue","pixel"] 和 ["phone","gold","iphone"] 。注意，["computer","silver","phone"] 未匹配检索规则。
//
// 提示：
// 1 <= items.length <= 104
// 1 <= typei.length, colori.length, namei.length, ruleValue.length <= 10
// ruleKey 等于 "type"、"color" 或 "name"
// 所有字符串仅由小写字母组成
func countMatches(items [][]string, ruleKey string, ruleValue string) int {

	var getRuleKeyIndex = func(key string) int {
		switch key {
		case "type":
			return 0
		case "color":
			return 1
		case "name":
			return 2
		}
		return -1
	}
	result := 0
	index := getRuleKeyIndex(ruleKey)
	for _, item := range items {
		if item[index] == ruleValue {
			result++
		}
	}
	return result
}

// 1662. 检查两个字符串数组是否相等
// 给你两个字符串数组 word1 和 word2 。如果两个数组表示的字符串相同，返回 true ；否则，返回 false 。
//
// 数组表示的字符串 是由数组中的所有元素 按顺序 连接形成的字符串。
//
// 示例 1：
// 输入：word1 = ["ab", "c"], word2 = ["a", "bc"]
// 输出：true
// 解释：
// word1 表示的字符串为 "ab" + "c" -> "abc"
// word2 表示的字符串为 "a" + "bc" -> "abc"
// 两个字符串相同，返回 true
//
// 示例 2：
// 输入：word1 = ["a", "cb"], word2 = ["ab", "c"]
// 输出：false
//
// 示例 3：
// 输入：word1  = ["abc", "d", "defg"], word2 = ["abcddefg"]
// 输出：true
//
// 提示：
// 1 <= word1.length, word2.length <= 103
// 1 <= word1[i].length, word2[i].length <= 103
// 1 <= sum(word1[i].length), sum(word2[i].length) <= 103
// word1[i] 和 word2[i] 由小写字母组成
func arrayStringsAreEqual(word1 []string, word2 []string) bool {
	var builder1, builder2 strings.Builder
	for _, word := range word1 {
		builder1.WriteString(word)
	}
	for _, word := range word2 {
		builder2.WriteString(word)
	}
	return builder1.String() == builder2.String()
}

// 1668. 最大重复子字符串
// 给你一个字符串 sequence ，如果字符串 word 连续重复 k 次形成的字符串是 sequence 的一个子字符串，那么单词 word 的 重复值为 k 。单词 word 的 最大重复值 是单词 word 在 sequence 中最大的重复值。如果 word 不是 sequence 的子串，那么重复值 k 为 0 。
//
// 给你一个字符串 sequence 和 word ，请你返回 最大重复值 k 。
//
// 示例 1：
// 输入：sequence = "ababc", word = "ab"
// 输出：2
// 解释："abab" 是 "ababc" 的子字符串。
//
// 示例 2：
// 输入：sequence = "ababc", word = "ba"
// 输出：1
// 解释："ba" 是 "ababc" 的子字符串，但 "baba" 不是 "ababc" 的子字符串。
//
// 示例 3：
// 输入：sequence = "ababc", word = "ac"
// 输出：0
// 解释："ac" 不是 "ababc" 的子字符串。
//
// 提示：
// 1 <= sequence.length <= 100
// 1 <= word.length <= 100
// sequence 和 word 都只包含小写英文字母。
func maxRepeating(sequence string, word string) int {
	m, n := len(sequence), len(word)
	if m < n {
		return 0
	}
	getRepeat := func(start int) int {
		count := 0
	out:
		for i := start; i <= m-n; i += n {
			for j := 0; j < n; j++ {
				if sequence[i+j] != word[j] {
					break out
				}
			}
			count++
		}

		return count
	}

	result := 0
	for i := 0; i < m; i++ {
		if sequence[i] != word[0] {
			continue
		}
		count := getRepeat(i)
		result = max(result, count)
		//i += count * n
	}
	return result
}

// 791. 自定义字符串排序
// 给定两个字符串 order 和 s 。order 的所有单词都是 唯一 的，并且以前按照一些自定义的顺序排序。
//
// 对 s 的字符进行置换，使其与排序的 order 相匹配。更具体地说，如果在 order 中的字符 x 出现字符 y 之前，那么在排列后的字符串中， x 也应该出现在 y 之前。
//
// 返回 满足这个性质的 s 的任意排列 。
//
// 示例 1:
// 输入: order = "cba", s = "abcd"
// 输出: "cbad"
// 解释:
// “a”、“b”、“c”是按顺序出现的，所以“a”、“b”、“c”的顺序应该是“c”、“b”、“a”。
// 因为“d”不是按顺序出现的，所以它可以在返回的字符串中的任何位置。“dcba”、“cdba”、“cbda”也是有效的输出。
//
// 示例 2:
// 输入: order = "cbafg", s = "abcd"
// 输出: "cbad"
//
// 提示:
// 1 <= order.length <= 26
// 1 <= s.length <= 200
// order 和 s 由小写英文字母组成
// order 中的所有字符都 不同
func customSortString(order string, s string) string {
	var builder strings.Builder
	letters := make([]int, 26)
	for _, c := range s {
		letters[c-'a']++
	}
	// 先排 order
	for i := range order {
		c := order[i]
		for j := 0; j < letters[c-'a']; j++ {
			builder.WriteByte(c)
		}
		letters[c-'a'] = 0
	}
	for i := 0; i < 26; i++ {
		for j := 0; j < letters[i]; j++ {
			builder.WriteByte(byte('a' + i))
		}
	}
	return builder.String()
}

// 816. 模糊坐标
// 我们有一些二维坐标，如 "(1, 3)" 或 "(2, 0.5)"，然后我们移除所有逗号，小数点和空格，得到一个字符串S。返回所有可能的原始字符串到一个列表中。
//
// 原始的坐标表示法不会存在多余的零，所以不会出现类似于"00", "0.0", "0.00", "1.0", "001", "00.01"或一些其他更小的数来表示坐标。此外，一个小数点前至少存在一个数，所以也不会出现“.1”形式的数字。
//
// 最后返回的列表可以是任意顺序的。而且注意返回的两个数字中间（逗号之后）都有一个空格。
//
// 示例 1:
// 输入: "(123)"
// 输出: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]
//
// 示例 2:
// 输入: "(00011)"
// 输出:  ["(0.001, 1)", "(0, 0.011)"]
// 解释:
// 0.0, 00, 0001 或 00.01 是不被允许的。
//
// 示例 3:
// 输入: "(0123)"
// 输出: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)", "(0.12, 3)"]
//
// 示例 4:
// 输入: "(100)"
// 输出: [(10, 0)]
// 解释:
// 1.0 是不被允许的。
//
// 提示:
// 4 <= S.length <= 12.
// S[0] = "(", S[S.length - 1] = ")", 且字符串 S 中的其他元素都是数字。
func ambiguousCoordinates(s string) []string {

	getCoordinates := func(str string) []string {
		strs := make([]string, 0)
		m := len(str)
		if m == 1 {
			strs = append(strs, str)
			return strs
		}
		if str[0] == '0' {
			if str[m-1] != '0' {
				strs = append(strs, "0."+str[1:])
			}
			return strs
		}
		if str[m-1] == '0' {
			strs = append(strs, str)
			return strs
		}
		strs = append(strs, str)

		for i := 1; i < m; i++ {
			strs = append(strs, str[:i]+"."+str[i:])
		}

		return strs
	}

	n := len(s)
	result := make([]string, 0)
	for i := 2; i < n-1; i++ {
		for _, left := range getCoordinates(s[1:i]) {
			for _, right := range getCoordinates(s[i : n-1]) {
				result = append(result, "("+left+", "+right+")")
			}
		}
	}

	return result
}

// 1684. 统计一致字符串的数目
// 给你一个由不同字符组成的字符串 allowed 和一个字符串数组 words 。如果一个字符串的每一个字符都在 allowed 中，就称这个字符串是 一致字符串 。
//
// 请你返回 words 数组中 一致字符串 的数目。
//
// 示例 1：
// 输入：allowed = "ab", words = ["ad","bd","aaab","baa","badab"]
// 输出：2
// 解释：字符串 "aaab" 和 "baa" 都是一致字符串，因为它们只包含字符 'a' 和 'b' 。
//
// 示例 2：
// 输入：allowed = "abc", words = ["a","b","c","ab","ac","bc","abc"]
// 输出：7
// 解释：所有字符串都是一致的。
//
// 示例 3：
// 输入：allowed = "cad", words = ["cc","acd","b","ba","bac","bad","ac","d"]
// 输出：4
// 解释：字符串 "cc"，"acd"，"ac" 和 "d" 是一致字符串。
//
// 提示：
// 1 <= words.length <= 104
// 1 <= allowed.length <= 26
// 1 <= words[i].length <= 10
// allowed 中的字符 互不相同 。
// words[i] 和 allowed 只包含小写英文字母。
func countConsistentStrings(allowed string, words []string) int {
	result := 0
	letters := make([]bool, 26)
	for _, c := range allowed {
		letters[c-'a'] = true
	}
out:
	for _, word := range words {
		for _, c := range word {
			if !letters[c-'a'] {
				continue out
			}
		}
		result++
	}

	return result
}

// 1704. 判断字符串的两半是否相似
// 给你一个偶数长度的字符串 s 。将其拆分成长度相同的两半，前一半为 a ，后一半为 b 。
//
// 两个字符串 相似 的前提是它们都含有相同数目的元音（'a'，'e'，'i'，'o'，'u'，'A'，'E'，'I'，'O'，'U'）。注意，s 可能同时含有大写和小写字母。
//
// 如果 a 和 b 相似，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：s = "book"
// 输出：true
// 解释：a = "bo" 且 b = "ok" 。a 中有 1 个元音，b 也有 1 个元音。所以，a 和 b 相似。
//
// 示例 2：
// 输入：s = "textbook"
// 输出：false
// 解释：a = "text" 且 b = "book" 。a 中有 1 个元音，b 中有 2 个元音。因此，a 和 b 不相似。
// 注意，元音 o 在 b 中出现两次，记为 2 个。
//
// 提示：
// 2 <= s.length <= 1000
// s.length 是偶数
// s 由 大写和小写 字母组成
func halvesAreAlike(s string) bool {
	n := len(s) >> 1
	count := 0
	for i := 0; i < n; i++ {
		c1, c2 := s[i], s[i+n]
		if isVowel(c1) {
			count++
		}
		if isVowel(c2) {
			count--
		}
	}

	return count == 0
}

// 1796. 字符串中第二大的数字
// 给你一个混合字符串 s ，请你返回 s 中 第二大 的数字，如果不存在第二大的数字，请你返回 -1 。
// 混合字符串 由小写英文字母和数字组成。
//
// 示例 1：
// 输入：s = "dfa12321afd"
// 输出：2
// 解释：出现在 s 中的数字包括 [1, 2, 3] 。第二大的数字是 2 。
//
// 示例 2：
// 输入：s = "abc1111"
// 输出：-1
// 解释：出现在 s 中的数字只包含 [1] 。没有第二大的数字。
//
// 提示：
// 1 <= s.length <= 500
// s 只包含小写英文字母和（或）数字。
func secondHighest(s string) int {
	maxVal, second := -1, -1
	for _, c := range s {
		if c >= '0' && c <= '9' {
			num := int(c - '0')
			if num > maxVal {
				second = maxVal
				maxVal = num
			} else if num < maxVal && num > second {
				second = num
			}
		}
	}

	return second
}

// 809. 情感丰富的文字
// 有时候人们会用重复写一些字母来表示额外的感受，比如 "hello" -> "heeellooo", "hi" -> "hiii"。我们将相邻字母都相同的一串字符定义为相同字母组，例如："h", "eee", "ll", "ooo"。
//
// 对于一个给定的字符串 S ，如果另一个单词能够通过将一些字母组扩张从而使其和 S 相同，我们将这个单词定义为可扩张的（stretchy）。扩张操作定义如下：选择一个字母组（包含字母 c ），然后往其中添加相同的字母 c 使其长度达到 3 或以上。
//
// 例如，以 "hello" 为例，我们可以对字母组 "o" 扩张得到 "hellooo"，但是无法以同样的方法得到 "helloo" 因为字母组 "oo" 长度小于 3。此外，我们可以进行另一种扩张 "ll" -> "lllll" 以获得 "helllllooo"。如果 S = "helllllooo"，那么查询词 "hello" 是可扩张的，因为可以对它执行这两种扩张操作使得 query = "hello" -> "hellooo" -> "helllllooo" = S。
//
// 输入一组查询单词，输出其中可扩张的单词数量。
//
// 示例：
// 输入：
// S = "heeellooo"
// words = ["hello", "hi", "helo"]
// 输出：1
// 解释：
// 我们能通过扩张 "hello" 的 "e" 和 "o" 来得到 "heeellooo"。
// 我们不能通过扩张 "helo" 来得到 "heeellooo" 因为 "ll" 的长度小于 3 。
//
// 提示：
// 0 <= len(S) <= 100。
// 0 <= len(words) <= 100。
// 0 <= len(words[i]) <= 100。
// S 和所有在 words 中的单词都只由小写字母组成。
func expressiveWords(s string, words []string) int {
	n := len(s)
	var canExpressWord = func(word string) bool {
		m := len(word)
		i, j := 0, 0
		for i < n && j < m {
			c1, c2 := s[i], word[j]
			if c1 != c2 {
				return false
			}
			start1, start2 := i, j
			i++
			j++
			for i < n && s[i] == c1 {
				i++
			}
			for j < m && word[j] == c2 {
				j++
			}
			if i-start1 == j-start2 {
				continue
			}
			if j-start2 > i-start1 || i-start1 < 3 {
				return false
			}
		}
		return i == n && j == m
	}

	count := 0
	for _, word := range words {
		if canExpressWord(word) {
			count++
		}
	}

	return count
}

// 1758. 生成交替二进制字符串的最少操作数
// 给你一个仅由字符 '0' 和 '1' 组成的字符串 s 。一步操作中，你可以将任一 '0' 变成 '1' ，或者将 '1' 变成 '0' 。
//
// 交替字符串 定义为：如果字符串中不存在相邻两个字符相等的情况，那么该字符串就是交替字符串。例如，字符串 "010" 是交替字符串，而字符串 "0100" 不是。
//
// 返回使 s 变成 交替字符串 所需的 最少 操作数。
//
// 示例 1：
// 输入：s = "0100"
// 输出：1
// 解释：如果将最后一个字符变为 '1' ，s 就变成 "0101" ，即符合交替字符串定义。
//
// 示例 2：
// 输入：s = "10"
// 输出：0
// 解释：s 已经是交替字符串。
//
// 示例 3：
// 输入：s = "1111"
// 输出：2
// 解释：需要 2 步操作得到 "0101" 或 "1010" 。
//
// 提示：
// 1 <= s.length <= 104
// s[i] 是 '0' 或 '1'
func minOperationsI(s string) int {
	count1, count2 := 0, 0
	n := len(s)
	for i := 0; i < n; i++ {
		if i&1 == 0 {
			// 偶数
			if s[i] == '1' {
				count1++
			} else {
				count2++
			}
		} else {
			// 奇数
			if s[i] == '0' {
				count1++
			} else {
				count2++
			}
		}
	}
	return min(count1, count2)
}

// 1805. 字符串中不同整数的数目
// 给你一个字符串 word ，该字符串由数字和小写英文字母组成。
//
// 请你用空格替换每个不是数字的字符。例如，"a123bc34d8ef34" 将会变成 " 123  34 8  34" 。注意，剩下的这些整数为（相邻彼此至少有一个空格隔开）："123"、"34"、"8" 和 "34" 。
//
// 返回对 word 完成替换后形成的 不同 整数的数目。
//
// 只有当两个整数的 不含前导零 的十进制表示不同， 才认为这两个整数也不同。
//
// 示例 1：
// 输入：word = "a123bc34d8ef34"
// 输出：3
// 解释：不同的整数有 "123"、"34" 和 "8" 。注意，"34" 只计数一次。
//
// 示例 2：
// 输入：word = "leet1234code234"
// 输出：2
//
// 示例 3：
// 输入：word = "a1b01c001"
// 输出：1
// 解释："1"、"01" 和 "001" 视为同一个整数的十进制表示，因为在比较十进制值时会忽略前导零的存在。
//
// 提示：
// 1 <= word.length <= 1000
// word 由数字和小写英文字母组成
func numDifferentIntegers(word string) int {
	numMap := make(map[string]bool)
	n := len(word)
	for i := 0; i < n; i++ {
		for i < n && !isNumber(word[i]) {
			i++
		}
		if i == n {
			break
		}
		start := i
		for i < n && isNumber(word[i]) {
			i++
		}
		end := i
		// 去掉前导0
		for start < end-1 && word[start] == '0' {
			start++
		}

		num := word[start:end]
		numMap[num] = true
	}
	return len(numMap)
}

// 1812. 判断国际象棋棋盘中一个格子的颜色
// 给你一个坐标 coordinates ，它是一个字符串，表示国际象棋棋盘中一个格子的坐标。下图是国际象棋棋盘示意图。
//
// 如果所给格子的颜色是白色，请你返回 true，如果是黑色，请返回 false 。
//
// 给定坐标一定代表国际象棋棋盘上一个存在的格子。坐标第一个字符是字母，第二个字符是数字。
//
// 示例 1：
// 输入：coordinates = "a1"
// 输出：false
// 解释：如上图棋盘所示，"a1" 坐标的格子是黑色的，所以返回 false 。
//
// 示例 2：
// 输入：coordinates = "h3"
// 输出：true
// 解释：如上图棋盘所示，"h3" 坐标的格子是白色的，所以返回 true 。
//
// 示例 3：
// 输入：coordinates = "c7"
// 输出：false
//
// 提示：
// coordinates.length == 2
// 'a' <= coordinates[0] <= 'h'
// '1' <= coordinates[1] <= '8'
func squareIsWhite(coordinates string) bool {
	b1 := ((coordinates[0] - 'a') & 1) == 1
	b2 := ((coordinates[1] - '1') & 1) == 1

	return b1 != b2
}

// 2351. 第一个出现两次的字母
// 给你一个由小写英文字母组成的字符串 s ，请你找出并返回第一个出现 两次 的字母。
//
// 注意：
// 如果 a 的 第二次 出现比 b 的 第二次 出现在字符串中的位置更靠前，则认为字母 a 在字母 b 之前出现两次。
// s 包含至少一个出现两次的字母。
//
// 示例 1：
// 输入：s = "abccbaacz"
// 输出："c"
// 解释：
// 字母 'a' 在下标 0 、5 和 6 处出现。
// 字母 'b' 在下标 1 和 4 处出现。
// 字母 'c' 在下标 2 、3 和 7 处出现。
// 字母 'z' 在下标 8 处出现。
// 字母 'c' 是第一个出现两次的字母，因为在所有字母中，'c' 第二次出现的下标是最小的。
//
// 示例 2：
// 输入：s = "abcdd"
// 输出："d"
// 解释：
// 只有字母 'd' 出现两次，所以返回 'd' 。
//
// 提示：
// 2 <= s.length <= 100
// s 由小写英文字母组成
// s 包含至少一个重复字母
func repeatedCharacter(s string) byte {
	letters := make([]int, 26)
	for i := 0; i < len(s); i++ {
		c := s[i]
		letters[c-'a']++
		if letters[c-'a'] == 2 {
			return c
		}
	}
	return ' '
}

// 804. 唯一摩尔斯密码词
// 国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 比如:
//
// 'a' 对应 ".-" ，
// 'b' 对应 "-..." ，
// 'c' 对应 "-.-." ，以此类推。
// 为了方便，所有 26 个英文字母的摩尔斯密码表如下：
//
// [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
// 给你一个字符串数组 words ，每个单词可以写成每个字母对应摩尔斯密码的组合。
//
// 例如，"cab" 可以写成 "-.-..--..." ，(即 "-.-." + ".-" + "-..." 字符串的结合)。我们将这样一个连接过程称作 单词翻译 。
// 对 words 中所有单词进行单词翻译，返回不同 单词翻译 的数量。
//
// 示例 1：
// 输入: words = ["gin", "zen", "gig", "msg"]
// 输出: 2
// 解释:
// 各单词翻译如下:
// "gin" -> "--...-."
// "zen" -> "--...-."
// "gig" -> "--...--."
// "msg" -> "--...--."
// 共有 2 种不同翻译, "--...-." 和 "--...--.".
//
// 示例 2：
// 输入：words = ["a"]
// 输出：1
//
// 提示：
// 1 <= words.length <= 100
// 1 <= words[i].length <= 12
// words[i] 由小写英文字母组成
func uniqueMorseRepresentations(words []string) int {
	morses := []string{
		".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..",
		"--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-",
		"-.--", "--..",
	}
	set := make(map[string]bool)

	var builder strings.Builder
	for _, word := range words {
		builder.Reset()
		for _, c := range word {
			index := c - 'a'
			builder.WriteString(morses[index])
		}
		set[builder.String()] = true
	}
	return len(set)
}

// 821. 字符的最短距离
// 给你一个字符串 s 和一个字符 c ，且 c 是 s 中出现过的字符。
// 返回一个整数数组 answer ，其中 answer.length == s.length 且 answer[i] 是 s 中从下标 i 到离它 最近 的字符 c 的 距离 。
// 两个下标 i 和 j 之间的 距离 为 abs(i - j) ，其中 abs 是绝对值函数。
//
// 示例 1：
// 输入：s = "loveleetcode", c = "e"
// 输出：[3,2,1,0,1,0,0,1,2,2,1,0]
// 解释：字符 'e' 出现在下标 3、5、6 和 11 处（下标从 0 开始计数）。
// 距下标 0 最近的 'e' 出现在下标 3 ，所以距离为 abs(0 - 3) = 3 。
// 距下标 1 最近的 'e' 出现在下标 3 ，所以距离为 abs(1 - 3) = 2 。
// 对于下标 4 ，出现在下标 3 和下标 5 处的 'e' 都离它最近，但距离是一样的 abs(4 - 3) == abs(4 - 5) = 1 。
// 距下标 8 最近的 'e' 出现在下标 6 ，所以距离为 abs(8 - 6) = 2 。
//
// 示例 2：
// 输入：s = "aaab", c = "b"
// 输出：[3,2,1,0]
//
// 提示：
// 1 <= s.length <= 104
// s[i] 和 c 均为小写英文字母
// 题目数据保证 c 在 s 中至少出现一次
func shortestToChar(s string, c byte) []int {
	n := len(s)
	result := make([]int, n)
	// 前后遍历两次
	index := math.MaxInt32
	for i := 0; i < n; i++ {
		if s[i] == c {
			index = 0
		}
		result[i] = index
		if index != math.MaxInt32 {
			index++
		}
	}
	index = math.MaxInt32
	for i := n - 1; i >= 0; i-- {
		if s[i] == c {
			index = 0
		}
		if index != math.MaxInt32 {
			result[i] = min(result[i], index)
			index++
		}
	}
	return result
}

// 830. 较大分组的位置
// 在一个由小写字母构成的字符串 s 中，包含由一些连续的相同字符所构成的分组。
// 例如，在字符串 s = "abbxxxxzyy" 中，就含有 "a", "bb", "xxxx", "z" 和 "yy" 这样的一些分组。
// 分组可以用区间 [start, end] 表示，其中 start 和 end 分别表示该分组的起始和终止位置的下标。上例中的 "xxxx" 分组用区间表示为 [3,6] 。
// 我们称所有包含大于或等于三个连续字符的分组为 较大分组 。
// 找到每一个 较大分组 的区间，按起始位置下标递增顺序排序后，返回结果。
//
// 示例 1：
// 输入：s = "abbxxxxzzy"
// 输出：[[3,6]]
// 解释："xxxx" 是一个起始于 3 且终止于 6 的较大分组。
//
// 示例 2：
// 输入：s = "abc"
// 输出：[]
// 解释："a","b" 和 "c" 均不是符合要求的较大分组。
//
// 示例 3：
// 输入：s = "abcdddeeeeaabbbcd"
// 输出：[[3,5],[6,9],[12,14]]
// 解释：较大分组为 "ddd", "eeee" 和 "bbb"
//
// 示例 4：
// 输入：s = "aba"
// 输出：[]
//
// 提示：
// 1 <= s.length <= 1000
// s 仅含小写英文字母
func largeGroupPositions(s string) [][]int {
	n := len(s)
	start := 0
	result := make([][]int, 0)
	for i := 1; i <= n; i++ {
		if i == n || s[i-1] != s[i] {
			if i-start >= 3 {
				tmp := make([]int, 2)
				tmp[0], tmp[1] = start, i-1
				result = append(result, tmp)
			}
			start = i
		}
	}
	return result
}

// 925. 长按键入
// 你的朋友正在使用键盘输入他的名字 name。偶尔，在键入字符 c 时，按键可能会被长按，而字符可能被输入 1 次或多次。
// 你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），那么就返回 True。
//
// 示例 1：
// 输入：name = "alex", typed = "aaleex"
// 输出：true
// 解释：'alex' 中的 'a' 和 'e' 被长按。
//
// 示例 2：
// 输入：name = "saeed", typed = "ssaaedd"
// 输出：false
// 解释：'e' 一定需要被键入两次，但在 typed 的输出中不是这样。
//
// 提示：
// 1 <= name.length, typed.length <= 1000
// name 和 typed 的字符都是小写字母
func isLongPressedName(name string, typed string) bool {
	m, n := len(name), len(typed)
	index := 0
	for i := 0; i < n; i++ {
		c := typed[i]
		if index < m && c == name[index] {
			index++
		} else if index > 0 && c == name[index-1] {

		} else {
			return false
		}
	}
	return index == m
}
