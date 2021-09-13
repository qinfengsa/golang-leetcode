package string

import (
	"container/list"
	"fmt"
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
//
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
//「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：
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
//    "1",
//    "2",
//    "Fizz",
//    "4",
//    "Buzz",
//    "Fizz",
//    "7",
//    "8",
//    "Fizz",
//    "Buzz",
//    "11",
//    "Fizz",
//    "13",
//    "14",
//    "FizzBuzz"
// ]
func fizzBuzz(n int) []string {
	list := make([]string, n)
	for i := 1; i <= n; i++ {
		b1, b2 := i%3 == 0, i%5 == 0
		if b1 && b2 {
			list[i] = "FizzBuzz"
		} else if b1 {
			list[i] = "Fizz"
		} else if b2 {
			list[i] = "Buzz"
		} else {
			list[i] = fmt.Sprintf("%d", i)
		}
	}
	return list
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
//     注意，两个额外的破折号需要删掉。
//
// 示例 2：
// 输入：S = "2-5g-3-J", K = 2 输出："2-5G-3J"
// 解释：字符串 S 被分成了 3 个部分，按照前面的规则描述，第一部分的字符可以少于给定的数量，其余部分皆为 2 个字符。
//
// 提示:
// S 的长度可能很长，请按需分配大小。K 为正整数。
// S 只包含字母数字（a-z，A-Z，0-9）以及破折号'-'
// S 非空
func licenseKeyFormatting(S string, K int) string {
	size := len(S)
	clen := size + size/K + 1
	chars := make([]byte, clen)
	index, count := clen-1, 0
	for i := size - 1; i >= 0; i-- {
		c := S[i]
		if count == K {
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
	size := len(word)
	if word[0] >= 'a' {
		for i := 2; i < size; i++ {
			if word[i] <= 'Z' {
				return false
			}
		}
	} else {
		// 首字母大写
		if word[size-1] >= 'a' {
			for i := 2; i < size-1; i++ {
				if word[i] <= 'Z' {
					return false
				}
			}
		} else { // 全大写
			for i := 2; i < size-1; i++ {
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
//「最长特殊序列」定义如下：该序列为某字符串独有的最长子序列（即不能是其他字符串的子序列）。
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
func reverseWords(s string) string {
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
//
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
//     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
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
//
// 示例 1：
// 输入：s = "42" 输出：42
// 解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
// 第 1 步："42"（当前没有读入字符，因为没有前导空格）
//         ^
// 第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
//         ^
// 第 3 步："42"（读入 "42"）
//           ^
// 解析得到整数 42 。
// 由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
//
// 示例 2：
// 输入：s = "   -42" 输出：-42
// 解释：
// 第 1 步："   -42"（读入前导空格，但忽视掉）
//            ^
// 第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
//             ^
// 第 3 步："   -42"（读入 "42"）
//               ^
// 解析得到整数 -42 。
// 由于 "-42" 在范围 [-231, 231 - 1] 内，最终结果为 -42 。
//
// 示例 3：
// 输入：s = "4193 with words" 输出：4193
// 解释：
// 第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
//         ^
// 第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
//         ^
// 第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
//             ^
// 解析得到整数 4193 。
// 由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。
//
// 示例 4：
// 输入：s = "words and 987" 输出：0
// 解释：
// 第 1 步："words and 987"（当前没有读入字符，因为没有前导空格）
//         ^
// 第 2 步："words and 987"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
//         ^
// 第 3 步："words and 987"（由于当前字符 'w' 不是一个数字，所以读入停止）
//         ^
// 解析得到整数 0 ，因为没有读入任何数字。
// 由于 0 在范围 [-231, 231 - 1] 内，最终结果为 0 。
//
// 示例 5：
// 输入：s = "-91283472332" 输出：-2147483648
// 解释：
// 第 1 步："-91283472332"（当前没有读入字符，因为没有前导空格）
//         ^
// 第 2 步："-91283472332"（读入 '-' 字符，所以结果应该是负数）
//          ^
// 第 3 步："-91283472332"（读入 "91283472332"）
//                     ^
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
//  下述格式之一：
//  至少一位数字，后面跟着一个点 '.'
//  至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
//  一个点 '.' ，后面跟着至少一位数字
//  整数（按顺序）可以分成以下几个部分：
//
// （可选）一个符号字符（'+' 或 '-'）
//   至少一位数字
//   部分有效数字列举如下：
//
//   ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
//  部分无效数字列举如下：
//
//  ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]
//  给你一个字符串 s ，如果 s 是一个 有效数字 ，请返回 true 。
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
//   "This    is    an",
//   "example  of text",
//   "justification.  "
// ]
// 示例 2:
// 输入: words = ["What","must","be","acknowledgment","shall","be"] maxWidth = 16
// 输出:
// [
//  "What   must   be",
//  "acknowledgment  ",
//  "shall be        "
// ]
// 解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
//     因为最后一行应为左对齐，而不是左右两端对齐。
//     第二行同样为左对齐，这是因为这行只包含一个单词。
//
// 示例 3:
// 输入: words = ["Science","is","what","we","understand","well","enough","to","explain",
//         "to","a","computer.","Art","is","everything","else","we","do"] maxWidth = 20
// 输出:
// [
//  "Science  is  what we",
//  "understand      well",
//  "enough to explain to",
//  "a  computer.  Art is",
//  "everything  else  we",
//  "do                  "
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
