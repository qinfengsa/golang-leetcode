package hash

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// 205. 同构字符串
// 给定两个字符串 s 和 t，判断它们是否是同构的。
//
// 如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。
//
// 所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。
//
// 示例 1: 输入: s = "egg", t = "add" 输出: true
// 示例 2: 输入: s = "foo", t = "bar" 输出: false
// 示例 3: 输入: s = "paper", t = "title" 输出: true
// 说明:
// 你可以假设 s 和 t 具有相同的长度。
func isIsomorphic(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	map1, map2 := make(map[byte]byte), make(map[byte]byte)

	for i := 0; i < len(s); i++ {
		c1, c2 := s[i], t[i]

		if v1, ok := map1[c1]; ok {
			if c2 != v1 {
				return false
			}
		} else {
			map1[c1] = c2
		}

		if v2, ok := map2[c2]; ok {
			if c1 != v2 {
				return false
			}

		} else {
			map2[c2] = c1
		}
	}

	return true
}

// 217. 存在重复元素
// 给定一个整数数组，判断是否存在重复元素。
//
// 如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。
// 示例 1:
// 输入: [1,2,3,1] 输出: true
// 示例 2:
// 输入: [1,2,3,4] 输出: false
// 示例 3:
// 输入: [1,1,1,3,3,4,3,2,4,2] 输出: true
func containsDuplicate(nums []int) bool {
	numMap := make(map[int]bool)
	for _, num := range nums {
		if _, ok := numMap[num]; ok {
			return true
		} else {
			numMap[num] = true
		}
	}
	return false
}

// 219. 存在重复元素 II
// 给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且 i 和 j 的差的 绝对值 至多为 k。
// 示例 1:
// 输入: nums = [1,2,3,1], k = 3 输出: true
// 示例 2:
// 输入: nums = [1,0,1,1], k = 1 输出: true
// 示例 3:
// 输入: nums = [1,2,3,1,2,3], k = 2 输出: false
func containsNearbyDuplicate(nums []int, k int) bool {
	indexMap := make(map[int]int)
	for i, num := range nums {
		if v, ok := indexMap[num]; ok {
			if i-v <= k {
				return true
			}
		}
		indexMap[num] = i
	}
	return false
}

// 242. 有效的字母异位词
// 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
//
// 示例 1:
//
// 输入: s = "anagram", t = "nagaram" 输出: true
// 示例 2:
//
// 输入: s = "rat", t = "car" 输出: false
// 说明:
// 你可以假设字符串只包含小写字母。
//
// 进阶:
// 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？
func isAnagram(s string, t string) bool {
	len1, len2 := len(s), len(t)
	if len1 != len2 {
		return false
	}
	counts := [128]int{}
	for i := 0; i < len1; i++ {
		counts[s[i]]++
		counts[t[i]]--
	}
	for _, count := range counts {
		if count != 0 {
			return false
		}
	}

	return true
}

// 290. 单词规律
// 给定一种规律 pattern 和一个字符串 str ，判断 str 是否遵循相同的规律。
//
// 这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 str 中的每个非空单词之间存在着双向连接的对应规律。
//
// 示例1:
// 输入: pattern = "abba", str = "dog cat cat dog" 输出: true
// 示例 2:
// 输入:pattern = "abba", str = "dog cat cat fish" 输出: false
// 示例 3:
// 输入: pattern = "aaaa", str = "dog cat cat dog" 输出: false
// 示例 4:
// 输入: pattern = "abba", str = "dog dog dog dog" 输出: false
// 说明:
// 你可以假设 pattern 只包含小写字母， str 包含了由单个空格分隔的小写字母。
func wordPattern(pattern string, s string) bool {
	patternMap, strMap := make(map[int32]string), map[string]int32{}
	strList := strings.Split(s, " ")
	if len(pattern) != len(strList) {
		return false
	}
	for i, c := range pattern {
		str := strList[i]
		if v, ok := patternMap[c]; ok {
			if strings.Compare(v, str) != 0 {
				return false
			}
		} else {
			patternMap[c] = str
		}

		if v, ok := strMap[str]; ok {
			if v != c {
				return false
			}

		} else {
			strMap[str] = c
		}

	}
	return true
}

// 349. 两个数组的交集
// 给定两个数组，编写一个函数来计算它们的交集。
//
// 示例 1：
//
// 输入：nums1 = [1,2,2,1], nums2 = [2,2]  输出：[2]
// 示例 2：
//
// 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4] 输出：[9,4]
//
// 说明：
//
// 输出结果中的每个元素一定是唯一的。
// 我们可以不考虑输出结果的顺序。
func intersection(nums1 []int, nums2 []int) []int {
	intMap := make(map[int]bool)
	var result []int
	for _, num := range nums1 {
		intMap[num] = true
	}
	for _, num := range nums2 {
		if intMap[num] == true {
			result = append(result, num)
			intMap[num] = false
		}
	}
	return result
}

// 350. 两个数组的交集 II
// 给定两个数组，编写一个函数来计算它们的交集。
//
// 示例 1：
//
// 输入：nums1 = [1,2,2,1], nums2 = [2,2] 输出：[2,2]
// 示例 2:
//
// 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4] 输出：[4,9]
//
// 说明：
//
// 输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
// 我们可以不考虑输出结果的顺序。
// 进阶：
//
// 如果给定的数组已经排好序呢？你将如何优化你的算法？
// 如果 nums1 的大小比 nums2 小很多，哪种方法更优？
// 如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？
func intersect(nums1 []int, nums2 []int) []int {
	intMap := make(map[int]int)
	var result []int
	for _, num := range nums1 {
		intMap[num]++
	}
	for _, num := range nums2 {
		if intMap[num] > 0 {
			result = append(result, num)
			intMap[num]--
		}
	}
	return result
}

// 1207. 独一无二的出现次数
// 给你一个整数数组 arr，请你帮忙统计数组中每个数的出现次数。
//
// 如果每个数的出现次数都是独一无二的，就返回 true；否则返回 false。
//
// 示例 1：
// 输入：arr = [1,2,2,1,1,3] 输出：true
// 解释：在该数组中，1 出现了 3 次，2 出现了 2 次，3 只出现了 1 次。没有两个数的出现次数相同。
//
// 示例 2：
// 输入：arr = [1,2] 输出：false
//
// 示例 3：
// 输入：arr = [-3,0,1,-3,1,1,1,-3,10,0] 输出：true
//
// 提示：
// 1 <= arr.length <= 1000
// -1000 <= arr[i] <= 1000
func uniqueOccurrences(arr []int) bool {

	numMap := map[int]int{}
	for _, num := range arr {
		numMap[num]++
	}

	countMap := map[int]bool{}
	for _, v := range numMap {
		if _, ok := countMap[v]; ok {
			return false
		} else {
			countMap[v] = true
		}
	}

	return true
}

// 409. 最长回文串
// 给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。
//
// 在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。
//
// 注意:
// 假设字符串的长度不会超过 1010。
//
// 示例 1:
// 输入: "abccccdd" 输出: 7
//
// 解释:
// 我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
func longestPalindrome(s string) int {
	letters := [128]int{}
	for _, c := range s {
		letters[c]++
	}
	count := 0
	flag := false
	for _, num := range letters {
		if (num & 1) == 1 {
			count += num - 1
			flag = true
		} else {
			count += num
		}
	}
	if flag {
		count++
	}

	return count
}

// 500. 键盘行
// 给定一个单词列表，只返回可以使用在键盘同一行的字母打印出来的单词。键盘如下图所示。
//
// American keyboard
// 示例：
//
// 输入: ["Hello", "Alaska", "Dad", "Peace"]
// 输出: ["Alaska", "Dad"]
//
// 注意：
//
// 你可以重复使用键盘上同一字符。
// 你可以假设输入的字符串将只包含字母。
func findWords(words []string) []string {
	dict := map[byte]byte{
		'q': 0, 'w': 0, 'e': 0, 'r': 0, 't': 0, 'y': 0, 'u': 0, 'i': 0, 'o': 0, 'p': 0,
		'Q': 0, 'W': 0, 'E': 0, 'R': 0, 'T': 0, 'Y': 0, 'U': 0, 'I': 0, 'O': 0, 'P': 0,
		'a': 1, 's': 1, 'd': 1, 'f': 1, 'g': 1, 'h': 1, 'j': 1, 'k': 1, 'l': 1,
		'A': 1, 'S': 1, 'D': 1, 'F': 1, 'G': 1, 'H': 1, 'J': 1, 'K': 1, 'L': 1,
		'z': 2, 'x': 2, 'c': 2, 'v': 2, 'b': 2, 'n': 2, 'm': 2,
		'Z': 2, 'X': 2, 'C': 2, 'V': 2, 'B': 2, 'N': 2, 'M': 2,
	}
	var result []string
out:
	for _, word := range words {

		first := dict[word[0]]
		for i := 1; i < len(word); i++ {
			if first != dict[word[i]] {
				continue out
			}
		}
		result = append(result, word)
	}

	return result
}

// 575. 分糖果
// 给定一个偶数长度的数组，其中不同的数字代表着不同种类的糖果，每一个数字代表一个糖果。你需要把这些糖果平均分给一个弟弟和一个妹妹。返回妹妹可以获得的最大糖果的种类数。
//
// 示例 1:
// 输入: candies = [1,1,2,2,3,3] 输出: 3
// 解析: 一共有三种种类的糖果，每一种都有两个。
//     最优分配方案：妹妹获得[1,2,3],弟弟也获得[1,2,3]。这样使妹妹获得糖果的种类数最多。
//
// 示例 2 :
// 输入: candies = [1,1,2,3] 输出: 2
// 解析: 妹妹获得糖果[2,3],弟弟获得糖果[1,1]，妹妹有两种不同的糖果，弟弟只有一种。这样使得妹妹可以获得的糖果种类数最多。
//
// 注意:
// 数组的长度为[2, 10,000]，并且确定为偶数。
// 数组中数字的大小在范围[-100,000, 100,000]内。
func distributeCandies(candies []int) int {
	half := len(candies) >> 1
	candyMap := map[int]bool{}
	for _, candy := range candies {
		candyMap[candy] = true
	}
	size := len(candyMap)
	if size >= half {
		return half
	}
	return size
}

// 599. 两个列表的最小索引总和
// 假设Andy和Doris想在晚餐时选择一家餐厅，并且他们都有一个表示最喜爱餐厅的列表，每个餐厅的名字用字符串表示。
//
// 你需要帮助他们用最少的索引和找出他们共同喜爱的餐厅。 如果答案不止一个，则输出所有答案并且不考虑顺序。 你可以假设总是存在一个答案。
//
// 示例 1:
// 输入: ["Shogun", "Tapioca Express", "Burger King", "KFC"]
// ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"]
// 输出: ["Shogun"]
// 解释: 他们唯一共同喜爱的餐厅是“Shogun”。
//
// 示例 2:
// 输入:
// ["Shogun", "Tapioca Express", "Burger King", "KFC"]
// ["KFC", "Shogun", "Burger King"]
// 输出: ["Shogun"]
// 解释: 他们共同喜爱且具有最小索引和的餐厅是“Shogun”，它有最小的索引和1(0+1)。
// 提示:
//
// 两个列表的长度范围都在 [1, 1000]内。
// 两个列表中的字符串的长度将在[1，30]的范围内。
// 下标从0开始，到列表的长度减1。
// 两个列表都没有重复的元素。
func findRestaurant(list1 []string, list2 []string) []string {
	minVal := 2000
	var result []string
	restMap := map[string]int{}
	for i, name := range list1 {
		restMap[name] = i
	}
	for i, name := range list2 {
		if idx, ok := restMap[name]; ok {
			sum := i + idx
			if sum < minVal {
				minVal = sum
				result = []string{name}
			} else if sum == minVal {
				result = append(result, name)
			}
		}
	}
	return result
}

type Employee struct {
	Id           int
	Importance   int
	Subordinates []int
}

// 690. 员工的重要性
// 给定一个保存员工信息的数据结构，它包含了员工 唯一的 id ，重要度 和 直系下属的 id 。
//
// 比如，员工 1 是员工 2 的领导，员工 2 是员工 3 的领导。他们相应的重要度为 15 , 10 , 5 。那么员工 1 的数据结构是 [1, 15, [2]] ，员工 2的 数据结构是 [2, 10, [3]] ，员工 3 的数据结构是 [3, 5, []] 。注意虽然员工 3 也是员工 1 的一个下属，但是由于 并不是直系 下属，因此没有体现在员工 1 的数据结构中。
//
// 现在输入一个公司的所有员工信息，以及单个员工 id ，返回这个员工和他所有下属的重要度之和。
//
// 示例：
// 输入：[[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
// 输出：11
// 解释：
// 员工 1 自身的重要度是 5 ，他有两个直系下属 2 和 3 ，而且 2 和 3 的重要度均为 3 。因此员工 1 的总重要度是 5 + 3 + 3 = 11 。
//
// 提示：
// 一个员工最多有一个 直系 领导，但是可以有多个 直系 下属
// 员工数量不超过 2000 。
func getImportance(employees []*Employee, id int) int {
	empMap := map[int]*Employee{}
	for _, emp := range employees {
		empMap[emp.Id] = emp
	}
	result := 0
	var dfs func(*Employee)
	dfs = func(employe *Employee) {
		result += employe.Importance
		for _, child := range employe.Subordinates {
			dfs(empMap[child])
		}
	}
	dfs(empMap[id])
	return result
}

// 451. 根据字符出现频率排序
// 给定一个字符串，请将字符串里的字符按照出现的频率降序排列。
//
// 示例 1:
// 输入: "tree" 输出: "eert"
// 解释:
// 'e'出现两次，'r'和't'都只出现一次。 因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
//
// 示例 2:
// 输入: "cccaaa" 输出: "cccaaa"
// 解释: 'c'和'a'都出现三次。此外，"aaaccc"也是有效的答案。
// 注意"cacaca"是不正确的，因为相同的字母必须放在一起。
//
// 示例 3:
// 输入: "Aabb" 输出: "bbAa"
// 解释:
// 此外，"bbaA"也是一个有效的答案，但"Aabb"是不正确的。
// 注意'A'和'a'被认为是两种不同的字符。
func frequencySort(s string) string {

	size := len(s)
	counts, bytes := make([]int, 128), make([]byte, size)

	for _, c := range s {
		counts[c]++
	}
	for i := range counts {
		counts[i] = counts[i]*1000 + i
	}
	sort.Ints(counts)
	idx := 0
	for i := len(counts) - 1; i >= 0; i-- {
		num := counts[i]
		if num == 0 {
			break
		}
		cnt, b := num/1000, num%1000
		for j := 0; j < cnt; j++ {
			bytes[idx] = byte(b)
			idx++
		}
	}

	return string(bytes)
}

// 36. 有效的数独
// 请你判断一个 9x9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。
//
// 数字 1-9 在每一行只能出现一次。
// 数字 1-9 在每一列只能出现一次。
// 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
// 数独部分空格内已填入了数字，空白格用 '.' 表示。
//
// 注意：
// 一个有效的数独（部分已被填充）不一定是可解的。
// 只需要根据以上规则，验证已经填入的数字是否有效即可。
//
// 示例 1：
// 输入：board =
// [["5","3",".",".","7",".",".",".","."]
// ,["6",".",".","1","9","5",".",".","."]
// ,[".","9","8",".",".",".",".","6","."]
// ,["8",".",".",".","6",".",".",".","3"]
// ,["4",".",".","8",".","3",".",".","1"]
// ,["7",".",".",".","2",".",".",".","6"]
// ,[".","6",".",".",".",".","2","8","."]
// ,[".",".",".","4","1","9",".",".","5"]
// ,[".",".",".",".","8",".",".","7","9"]]
// 输出：true
//
// 示例 2：
// 输入：board =
// [["8","3",".",".","7",".",".",".","."]
// ,["6",".",".","1","9","5",".",".","."]
// ,[".","9","8",".",".",".",".","6","."]
// ,["8",".",".",".","6",".",".",".","3"]
// ,["4",".",".","8",".","3",".",".","1"]
// ,["7",".",".",".","2",".",".",".","6"]
// ,[".","6",".",".",".",".","2","8","."]
// ,[".",".",".","4","1","9",".",".","5"]
// ,[".",".",".",".","8",".",".","7","9"]]
// 输出：false
// 解释：除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。 但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
//
// 提示：
//
// board.length == 9
// board[i].length == 9
// board[i][j] 是一位数字或者 '.'
func isValidSudoku(board [][]byte) bool {
	hashRows, hashCols := make([]int, 9), make([]int, 9)

	hashBoards := make([]int, 9)
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				continue
			}
			num := board[i][j] - '0'
			state := 1 << num
			if hashRows[i]&state > 0 || hashCols[j]&state > 0 {
				return false
			}
			hashRows[i] |= state
			hashCols[j] |= state
			// 计算9宫格位置
			idx := i/3*3 + j/3
			if hashBoards[idx]&state > 0 {
				return false
			}
			hashBoards[idx] |= state

		}
	}

	return true
}

// 49. 字母异位词分组
// 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
//
// 示例:
// 输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
// 输出:
// [
//  ["ate","eat","tea"],
//  ["nat","tan"],
//  ["bat"]
// ]
// 说明：
// 所有输入均为小写字母。
// 不考虑答案输出的顺序。
func groupAnagrams(strs []string) [][]string {
	result := make([][]string, 0)

	getAnagramsKey := func(str string) string {
		letters := make([]int, 26)
		for _, c := range str {
			letters[c-'a']++
		}
		var builder strings.Builder
		for _, num := range letters {
			builder.WriteString(strconv.Itoa(num))
			builder.WriteByte(',')
		}
		return builder.String()
	}

	strMap := make(map[string][]string)

	for i := range strs {
		key := getAnagramsKey(strs[i])
		if _, ok := strMap[key]; ok {
			strMap[key] = append(strMap[key], strs[i])
		} else {
			strMap[key] = []string{strs[i]}
		}
	}

	for _, v := range strMap {
		result = append(result, v)
	}

	return result
}

// 187. 重复的DNA序列
// 所有 DNA 都由一系列缩写为 'A'，'C'，'G' 和 'T' 的核苷酸组成，例如："ACGAATTCCG"。在研究 DNA 时，识别 DNA 中的重复序列有时会对研究非常有帮助。
//
// 编写一个函数来找出所有目标子串，目标子串的长度为 10，且在 DNA 字符串 s 中出现次数超过一次。
//
// 示例 1：
// 输入：s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
// 输出：["AAAAACCCCC","CCCCCAAAAA"]
//
// 示例 2：
// 输入：s = "AAAAAAAAAAAAA"
// 输出：["AAAAAAAAAA"]
//
// 提示：
// 0 <= s.length <= 105
// s[i] 为 'A'、'C'、'G' 或 'T'
func findRepeatedDnaSequences(s string) []string {
	countMap := make(map[string]int)
	result := make([]string, 0)
	for i := 0; i <= len(s)-10; i++ {
		str := s[i : i+10]
		if count, ok := countMap[str]; ok && count == 1 {
			result = append(result, str)
		}
		countMap[str]++
	}
	return result
}

func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

// 220. 存在重复元素 III
// 给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，同时又满足 abs(i - j) <= k 。
//
// 如果存在则返回 true，不存在返回 false。
//
// 示例 1：
// 输入：nums = [1,2,3,1], k = 3, t = 0
// 输出：true
//
// 示例 2：
// 输入：nums = [1,0,1,1], k = 1, t = 2
// 输出：true
//
// 示例 3：
// 输入：nums = [1,5,9,1,5,9], k = 2, t = 3
// 输出：false
//
// 提示：
// 0 <= nums.length <= 2 * 104
// -231 <= nums[i] <= 231 - 1
// 0 <= k <= 104
// 0 <= t <= 231 - 1
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	// 利用桶, 把 num 分成 t + 1 份
	getBucketId := func(x int) int {
		if x >= 0 {
			return x / (t + 1)
		}
		return ((x + 1) / (t + 1)) - 1
	}
	hashBucket := make(map[int]int)
	for i, num := range nums {
		if i-k-1 >= 0 {
			delete(hashBucket, getBucketId(nums[i-k-1]))
		}
		bucketId := getBucketId(num)
		if _, ok := hashBucket[bucketId]; ok {
			return true
		}
		if v, ok := hashBucket[bucketId-1]; ok {
			if abs(num-v) <= t {
				return true
			}
		}
		if v, ok := hashBucket[bucketId+1]; ok {
			if abs(num-v) <= t {
				return true
			}
		}
		hashBucket[bucketId] = num
	}
	return false
}

// 299. 猜数字游戏
// 你在和朋友一起玩 猜数字（Bulls and Cows）游戏，该游戏规则如下：
//
// 你写出一个秘密数字，并请朋友猜这个数字是多少。
// 朋友每猜测一次，你就会给他一个提示，告诉他的猜测数字中有多少位属于数字和确切位置都猜对了（称为“Bulls”, 公牛），有多少位属于数字猜对了但是位置不对（称为“Cows”, 奶牛）。
// 朋友根据提示继续猜，直到猜出秘密数字。
// 请写出一个根据秘密数字和朋友的猜测数返回提示的函数，返回字符串的格式为 xAyB ，x 和 y 都是数字，A 表示公牛，用 B 表示奶牛。
//
// xA 表示有 x 位数字出现在秘密数字中，且位置都与秘密数字一致。
// yB 表示有 y 位数字出现在秘密数字中，但位置与秘密数字不一致。
// 请注意秘密数字和朋友的猜测数都可能含有重复数字，每位数字只能统计一次。
//
// 示例 1:
// 输入: secret = "1807", guess = "7810"
// 输出: "1A3B"
// 解释: 1 公牛和 3 奶牛。公牛是 8，奶牛是 0, 1 和 7。
//
// 示例 2:
// 输入: secret = "1123", guess = "0111"
// 输出: "1A1B"
// 解释: 朋友猜测数中的第一个 1 是公牛，第二个或第三个 1 可被视为奶牛。
//
// 说明: 你可以假设秘密数字和朋友的猜测数都只包含数字，并且它们的长度永远相等。
func getHint(secret string, guess string) string {
	bucket := make([]int, 10)
	countA, countB, n := 0, 0, len(secret)

	for i := 0; i < n; i++ {
		bucket[secret[i]-'0']++
	}
	for i := 0; i < n; i++ {
		if secret[i] == guess[i] {
			countA++
		}
		if bucket[guess[i]-'0'] > 0 {
			countB++
			bucket[guess[i]-'0']--
		}
	}
	return fmt.Sprintf("%dA%dB", countA, countB-countA)
}

// 336. 回文对
// 给定一组 互不相同 的单词， 找出所有 不同 的索引对 (i, j)，使得列表中的两个单词， words[i] + words[j] ，可拼接成回文串。
//
// 示例 1：
// 输入：words = ["abcd","dcba","lls","s","sssll"]
// 输出：[[0,1],[1,0],[3,2],[2,4]]
// 解释：可拼接成的回文串为 ["dcbaabcd","abcddcba","slls","llssssll"]
//
// 示例 2：
// 输入：words = ["bat","tab","cat"]
// 输出：[[0,1],[1,0]]
// 解释：可拼接成的回文串为 ["battab","tabbat"]
//
// 示例 3：
// 输入：words = ["a",""]
// 输出：[[0,1],[1,0]]
//
// 提示：
// 1 <= words.length <= 5000
// 0 <= words[i].length <= 300
// words[i] 由小写英文字母组成
func palindromePairs(words []string) [][]int {
	indexMap := make(map[string]int)
	for i, word := range words {
		indexMap[reverseString(word)] = i
	}

	checkPalindrome := func(word string, left, right int) bool {
		for left < right {
			if word[left] != word[right] {
				return false
			}
			left++
			right--

		}
		return true
	}

	result := make([][]int, 0)
	for i, word := range words {
		n := len(word)
		if n == 0 {
			continue
		}
		for j := 0; j <= n; j++ {
			// 截取前缀 0:j 判断 j~n-1 是否回文
			if checkPalindrome(word, j, n-1) {
				if v, ok := indexMap[word[0:j]]; ok {
					if v != i {
						result = append(result, []int{i, v})
					}
				}
			}
			// 截取后缀 j:n 判断 0~j-1 是否回文
			if j != 0 && checkPalindrome(word, 0, j-1) {
				if v, ok := indexMap[word[j:]]; ok {
					if v != i {
						result = append(result, []int{v, i})
					}
				}
			}
		}
	}

	return result
}

func reverseString(s string) string {
	arr := []byte(s)
	left, right := 0, len(arr)-1
	for left < right {
		arr[left], arr[right] = arr[right], arr[left]
		left++
		right--
	}
	return string(arr)
}

// 347. 前 K 个高频元素
// 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
//
// 示例 1:
// 输入: nums = [1,1,1,2,2,3], k = 2
// 输出: [1,2]
//
// 示例 2:
// 输入: nums = [1], k = 1
// 输出: [1]
//
// 提示：
// 1 <= nums.length <= 105
// k 的取值范围是 [1, 数组中不相同的元素的个数]
// 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的
//
// 进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。
func topKFrequent(nums []int, k int) []int {
	countMap := make(map[int]int)
	for _, num := range nums {
		countMap[num]++
	}
	// topK 用堆 或者 快速排序
	h := hp{}

	for key, v := range countMap {
		heap.Push(&h, element{num: key, count: v})
		if h.Len() > k {
			heap.Pop(&h)
		}
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[k-i-1] = h[i].num
	}

	return result
}

type element struct {
	word  string
	num   int
	count int
}

type hp []element

func (h hp) Len() int {
	return len(h)
}

func (h hp) Less(i, j int) bool {
	return h[i].count < h[j].count
}

func (h hp) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *hp) Push(x interface{}) {
	*h = append(*h, x.(element))
}

func (h *hp) Pop() interface{} {
	tmp := *h
	v := tmp[len(tmp)-1]
	*h = tmp[:len(tmp)-1]
	return v
}

// 395. 至少有 K 个重复字符的最长子串
// 给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。
//
// 示例 1：
// 输入：s = "aaabb", k = 3
// 输出：3
// 解释：最长子串为 "aaa" ，其中 'a' 重复了 3 次。
//
// 示例 2：
// 输入：s = "ababbc", k = 2
// 输出：5
// 解释：最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
//
// 提示：
// 1 <= s.length <= 104
// s 仅由小写英文字母组成
// 1 <= k <= 105
func longestSubstring(s string, k int) int {
	n := len(s)
	var longestCount func(left, right int) int

	longestCount = func(left, right int) int {
		if right-left+1 < 0 {
			return 0
		}
		// 统计 所有 字母出现的频率
		var letters [26]int
		for i := left; i <= right; i++ {
			letters[s[i]-'a']++
		}
		// 移动 左右指针，保证 right 和 left 的 频率 >= k
		for right-left+1 >= k && letters[s[left]-'a'] < k {
			left++
		}
		for right-left+1 >= k && letters[s[right]-'a'] < k {
			right--
		}
		if right-left+1 < 0 {
			return 0
		}
		for i := left; i <= right; i++ {
			// 分割
			if letters[s[i]-'a'] < k {
				return max(
					longestCount(left, i-1), longestCount(i+1, right))
			}
		}

		return right - left + 1
	}
	return longestCount(0, n-1)
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 433. 最小基因变化
// 一条基因序列由一个带有8个字符的字符串表示，其中每个字符都属于 "A", "C", "G", "T"中的任意一个。
// 假设我们要调查一个基因序列的变化。一次基因变化意味着这个基因序列中的一个字符发生了变化。
// 例如，基因序列由"AACCGGTT" 变化至 "AACCGGTA" 即发生了一次基因变化。
// 与此同时，每一次基因变化的结果，都需要是一个合法的基因串，即该结果属于一个基因库。
// 现在给定3个参数 — start, end, bank，分别代表起始基因序列，目标基因序列及基因库，请找出能够使起始基因序列变化为目标基因序列所需的最少变化次数。如果无法实现目标变化，请返回 -1。
//
// 注意：
// 起始基因序列默认是合法的，但是它并不一定会出现在基因库中。
// 如果一个起始基因序列需要多次变化，那么它每一次变化之后的基因序列都必须是合法的。
// 假定起始基因序列与目标基因序列是不一样的。
//
// 示例 1：
// start: "AACCGGTT"
// end:   "AACCGGTA"
// bank: ["AACCGGTA"]
// 返回值: 1
//
// 示例 2：
// start: "AACCGGTT"
// end:   "AAACGGTA"
// bank: ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
// 返回值: 2
//
// 示例 3：
// start: "AAAAACCC"
// end:   "AACCCCCC"
// bank: ["AAAACCCC", "AAACCCCC", "AACCCCCC"]
// 返回值: 3
func minMutation(start string, end string, bank []string) int {
	bankMap := make(map[string]bool)
	for _, str := range bank {
		bankMap[str] = true
	}
	if _, ok := bankMap[end]; !ok {
		return -1
	}
	factors := [4]byte{'A', 'C', 'G', 'T'}
	visited := make(map[string]bool)

	var bfs func(s string) int

	bfs = func(s string) int {
		if s == end {
			return 0
		}
		chars := []byte(s)
		minCount := math.MaxInt32
		for i := 0; i < len(s); i++ {
			c := chars[i]
			for _, factor := range factors {
				if c == factor {
					continue
				}
				chars[i] = factor
				str := string(chars)
				if !bankMap[str] {
					continue
				}
				if visited[str] {
					continue
				}
				visited[str] = true
				count := bfs(str)
				if count != -1 {
					minCount = min(minCount, count+1)
				}

				visited[str] = false
			}

			chars[i] = c
		}
		if minCount == math.MaxInt32 {
			return -1
		}
		return minCount
	}

	return bfs(start)
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

// 447. 回旋镖的数量
// 给定平面上 n 对 互不相同 的点 points ，其中 points[i] = [xi, yi] 。回旋镖 是由点 (i, j, k) 表示的元组 ，其中 i 和 j 之间的距离和 i 和 k 之间的欧式距离相等（需要考虑元组的顺序）。
//
// 返回平面上所有回旋镖的数量。
//
// 示例 1：
// 输入：points = [[0,0],[1,0],[2,0]]
// 输出：2
// 解释：两个回旋镖为 [[1,0],[0,0],[2,0]] 和 [[1,0],[2,0],[0,0]]
//
// 示例 2：
// 输入：points = [[1,1],[2,2],[3,3]]
// 输出：2
//
// 示例 3：
// 输入：points = [[1,1]]
// 输出：0
//
// 提示：
// n == points.length
// 1 <= n <= 500
// points[i].length == 2
// -104 <= xi, yi <= 104
// 所有点都 互不相同
func numberOfBoomerangs(points [][]int) int {

	getDistance := func(point1, point2 []int) int {
		x, y := point1[0]-point2[0], point1[1]-point2[1]

		return x*x + y*y
	}
	n, result := len(points), 0
	for i := 0; i < n; i++ {
		// 到 i 点的距离 -> 数量
		distanceMap := make(map[int]int)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			distance := getDistance(points[i], points[j])
			if count, ok := distanceMap[distance]; ok {
				distanceMap[distance]++
				// 可以更换顺序
				result += 2 * count
			} else {
				distanceMap[distance] = 1
			}

		}
	}
	return result
}

// 454. 四数相加 II
// 给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：
//
// 0 <= i, j, k, l < n
// nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
//
// 示例 1：
// 输入：nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
// 输出：2
// 解释：
// 两个元组如下：
// 1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
// 2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
//
// 示例 2：
// 输入：nums1 = [0], nums2 = [0], nums3 = [0], nums4 = [0]
// 输出：1
//
// 提示：
// n == nums1.length
// n == nums2.length
// n == nums3.length
// n == nums4.length
// 1 <= n <= 200
// -2^28 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 2^28
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	numMap := make(map[int]int)
	result := 0
	for _, num1 := range nums1 {
		for _, num2 := range nums2 {
			numMap[num1+num2]++
		}
	}

	for _, num3 := range nums3 {
		for _, num4 := range nums4 {
			num := -num3 - num4
			if count, ok := numMap[num]; ok {
				result += count
			}
		}
	}

	return result
}

// 523. 连续的子数组和
// 给你一个整数数组 nums 和一个整数 k ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：
//
// 子数组大小 至少为 2 ，且
// 子数组元素总和为 k 的倍数。
// 如果存在，返回 true ；否则，返回 false 。
//
// 如果存在一个整数 n ，令整数 x 符合 x = n * k ，则称 x 是 k 的一个倍数。0 始终视为 k 的一个倍数。
//
// 示例 1：
// 输入：nums = [23,2,4,6,7], k = 6
// 输出：true
// 解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。
//
// 示例 2：
// 输入：nums = [23,2,6,4,7], k = 6
// 输出：true
// 解释：[23, 2, 6, 4, 7] 是大小为 5 的子数组，并且和为 42 。
// 42 是 6 的倍数，因为 42 = 7 * 6 且 7 是一个整数。
//
// 示例 3：
// 输入：nums = [23,2,6,4,7], k = 13
// 输出：false
//
// 提示：
// 1 <= nums.length <= 105
// 0 <= nums[i] <= 109
// 0 <= sum(nums[i]) <= 231 - 1
// 1 <= k <= 231 - 1
func checkSubarraySum(nums []int, k int) bool {
	// 前缀和
	sum := 0
	sumMap := map[int]int{
		0: -1,
	}
	for i, num := range nums {
		sum += num
		sum %= k
		if v, ok := sumMap[sum]; ok {
			if i-v > 1 {
				return true
			}
		} else {
			sumMap[sum] = i
		}
	}

	return false
}

// 532. 数组中的 k-diff 数对
// 给定一个整数数组和一个整数 k，你需要在数组里找到 不同的 k-diff 数对，并返回不同的 k-diff 数对 的数目。
//
// 这里将 k-diff 数对定义为一个整数对 (nums[i], nums[j])，并满足下述全部条件：
// 0 <= i < j < nums.length
// |nums[i] - nums[j]| == k
// 注意，|val| 表示 val 的绝对值。
//
// 示例 1：
// 输入：nums = [3, 1, 4, 1, 5], k = 2
// 输出：2
// 解释：数组中有两个 2-diff 数对, (1, 3) 和 (3, 5)。
// 尽管数组中有两个1，但我们只应返回不同的数对的数量。
//
// 示例 2：
// 输入：nums = [1, 2, 3, 4, 5], k = 1
// 输出：4
// 解释：数组中有四个 1-diff 数对, (1, 2), (2, 3), (3, 4) 和 (4, 5)。
//
// 示例 3：
// 输入：nums = [1, 3, 1, 5, 4], k = 0
// 输出：1
// 解释：数组中只有一个 0-diff 数对，(1, 1)。
//
// 示例 4：
// 输入：nums = [1,2,4,4,3,3,0,9,2,3], k = 3
// 输出：2
//
// 示例 5：
// 输入：nums = [-1,-2,-3], k = 1
// 输出：2
//
// 提示：
// 1 <= nums.length <= 104
// -107 <= nums[i] <= 107
// 0 <= k <= 107
func findPairs(nums []int, k int) int {
	numMap := make(map[int]int)
	result := 0
	for _, num := range nums {
		numMap[num]++
	}
	if k == 0 {
		for _, v := range numMap {
			if v >= 2 {
				result++
			}
		}
	} else {
		for key := range numMap {
			if numMap[key+k] > 0 {
				result++
			}
		}
	}
	return result
}

// 554. 砖墙
// 你的面前有一堵矩形的、由 n 行砖块组成的砖墙。这些砖块高度相同（也就是一个单位高）但是宽度不同。每一行砖块的宽度之和相等。
//
// 你现在要画一条 自顶向下 的、穿过 最少 砖块的垂线。如果你画的线只是从砖块的边缘经过，就不算穿过这块砖。你不能沿着墙的两个垂直边缘之一画线，这样显然是没有穿过一块砖的。
//
// 给你一个二维数组 wall ，该数组包含这堵墙的相关信息。其中，wall[i] 是一个代表从左至右每块砖的宽度的数组。你需要找出怎样画才能使这条线 穿过的砖块数量最少 ，并且返回 穿过的砖块数量 。
//
// 示例 1：
// 输入：wall = [[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]]
// 输出：2
//
// 示例 2：
// 输入：wall = [[1],[1],[1]]
// 输出：3
//
// 提示：
// n == wall.length
// 1 <= n <= 104
// 1 <= wall[i].length <= 104
// 1 <= sum(wall[i].length) <= 2 * 104
// 对于每一行 i ，sum(wall[i]) 是相同的
// 1 <= wall[i][j] <= 231 - 1
func leastBricks(wall [][]int) int {
	n := len(wall)
	endMap := make(map[int]int)
	for _, w := range wall {
		end := 0
		for i := 0; i < len(w)-1; i++ {
			end += w[i]
			endMap[end]++
		}
	}

	maxCount := 0
	for _, count := range endMap {
		maxCount = max(count, maxCount)
	}
	return n - maxCount
}

// 560. 和为 K 的子数组
// 给你一个整数数组 nums 和一个整数 k ，请你统计并返回该数组中和为 k 的连续子数组的个数。
//
// 示例 1：
// 输入：nums = [1,1,1], k = 2 输出：2
//
// 示例 2：
// 输入：nums = [1,2,3], k = 3 输出：2
//
// 提示：
// 1 <= nums.length <= 2 * 104
// -1000 <= nums[i] <= 1000
// -107 <= k <= 107
// 通过次数163,391提交次数3
func subarraySum(nums []int, k int) int {
	sum := 0
	result := 0
	numMap := map[int]int{0: 1}
	// 前缀和
	for _, num := range nums {
		sum += num
		if v, ok := numMap[sum-k]; ok {
			result += v
		}
		numMap[sum]++
	}
	return result
}

// 748. 最短补全词
// 给你一个字符串 licensePlate 和一个字符串数组 words ，请你找出并返回 words 中的 最短补全词 。
//
// 补全词 是一个包含 licensePlate 中所有的字母的单词。在所有补全词中，最短的那个就是 最短补全词 。
// 在匹配 licensePlate 中的字母时：
// 忽略 licensePlate 中的 数字和空格 。
// 不区分大小写。
// 如果某个字母在 licensePlate 中出现不止一次，那么该字母在补全词中的出现次数应当一致或者更多。
// 例如：licensePlate = "aBc 12c"，那么它的补全词应当包含字母 'a'、'b' （忽略大写）和两个 'c' 。可能的 补全词 有 "abccdef"、"caaacab" 以及 "cbca" 。
//
// 请你找出并返回 words 中的 最短补全词 。题目数据保证一定存在一个最短补全词。当有多个单词都符合最短补全词的匹配条件时取 words 中 最靠前的 那个。
//
// 示例 1：
// 输入：licensePlate = "1s3 PSt", words = ["step", "steps", "stripe", "stepple"]
// 输出："steps"
// 解释：最短补全词应该包括 "s"、"p"、"s"（忽略大小写） 以及 "t"。
// "step" 包含 "t"、"p"，但只包含一个 "s"，所以它不符合条件。
// "steps" 包含 "t"、"p" 和两个 "s"。
// "stripe" 缺一个 "s"。
// "stepple" 缺一个 "s"。
// 因此，"steps" 是唯一一个包含所有字母的单词，也是本例的答案。
//
// 示例 2：
// 输入：licensePlate = "1s3 456", words = ["looks", "pest", "stew", "show"]
// 输出："pest"
// 解释：licensePlate 只包含字母 "s" 。所有的单词都包含字母 "s" ，其中 "pest"、"stew"、和 "show" 三者最短。答案是 "pest" ，因为它是三个单词中在 words 里最靠前的那个。
//
// 示例 3：
// 输入：licensePlate = "Ah71752", words = ["suggest","letter","of","husband","easy","education","drug","prevent","writer","old"]
// 输出："husband"
//
// 示例 4：
// 输入：licensePlate = "OgEu755", words = ["enough","these","play","wide","wonder","box","arrive","money","tax","thus"]
// 输出："enough"
//
// 示例 5：
// 输入：licensePlate = "iMSlpe4", words = ["claim","consumer","student","camera","public","never","wonder","simple","thought","use"]
// 输出："simple"
//
// 提示：
// 1 <= licensePlate.length <= 7
// licensePlate 由数字、大小写字母或空格 ' ' 组成
// 1 <= words.length <= 1000
// 1 <= words[i].length <= 15
// words[i] 由小写英文字母组成
func shortestCompletingWord(licensePlate string, words []string) string {
	letters := [26]int{}
	for _, c := range licensePlate {
		if c >= 'a' && c <= 'z' {
			letters[c-'a']++
		}
		if c >= 'A' && c <= 'Z' {
			letters[c-'A']++
		}
	}
	compareWord := func(word string) bool {
		wordLetters := [26]int{}
		for _, c := range word {
			wordLetters[c-'a']++
		}
		for i := 0; i < 26; i++ {
			if letters[i] > wordLetters[i] {
				return false
			}
		}

		return true
	}
	result := ""
	for _, word := range words {
		if compareWord(word) && (len(result) == 0 || len(word) < len(result)) {
			result = word
		}
	}
	return result
}

// 609. 在系统中查找重复文件
// 给定一个目录信息列表，包括目录路径，以及该目录中的所有包含内容的文件，您需要找到文件系统中的所有重复文件组的路径。一组重复的文件至少包括二个具有完全相同内容的文件。
//
// 输入列表中的单个目录信息字符串的格式如下：
//
// "root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"
//
// 这意味着有 n 个文件（f1.txt, f2.txt ... fn.txt 的内容分别是 f1_content, f2_content ... fn_content）在目录 root/d1/d2/.../dm 下。注意：n>=1 且 m>=0。如果 m=0，则表示该目录是根目录。
//
// 该输出是重复文件路径组的列表。对于每个组，它包含具有相同内容的文件的所有文件路径。文件路径是具有下列格式的字符串：
//
// "directory_path/file_name.txt"
//
// 示例 1：
// 输入：
// ["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
// 输出：
// [["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]]
//
// 注：
// 最终输出不需要顺序。
// 您可以假设目录名、文件名和文件内容只有字母和数字，并且文件内容的长度在 [1，50] 的范围内。
// 给定的文件数量在 [1，20000] 个范围内。
// 您可以假设在同一目录中没有任何文件或目录共享相同的名称。
// 您可以假设每个给定的目录信息代表一个唯一的目录。目录路径和文件信息用一个空格分隔。
//
// 超越竞赛的后续行动：
//
// 假设您有一个真正的文件系统，您将如何搜索文件？广度搜索还是宽度搜索？
// 如果文件内容非常大（GB级别），您将如何修改您的解决方案？
// 如果每次只能读取 1 kb 的文件，您将如何修改解决方案？
// 修改后的解决方案的时间复杂度是多少？其中最耗时的部分和消耗内存的部分是什么？如何优化？
// 如何确保您发现的重复文件不是误报？
func findDuplicate(paths []string) [][]string {
	contentMap := make(map[string][]string)

	for _, path := range paths {
		dirs := strings.Split(path, " ")
		root := dirs[0]
		for i := 1; i < len(dirs); i++ {
			dir := dirs[i]
			idx := strings.Index(dir, "(")
			fileName := dir[:idx]
			content := dir[idx+1 : len(dir)-1]
			contentMap[content] = append(contentMap[content], root+"/"+fileName)
		}
	}
	result := make([][]string, 0)
	for _, v := range contentMap {
		if len(v) > 1 {
			result = append(result, v)
		}
	}
	return result
}

// 692. 前K个高频单词
// 给一非空的单词列表，返回前 k 个出现次数最多的单词。
// 返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。
//
// 示例 1：
// 输入: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
// 输出: ["i", "love"]
// 解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。
//    注意，按字母顺序 "i" 在 "love" 之前。
//
// 示例 2：
// 输入: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
// 输出: ["the", "is", "sunny", "day"]
// 解析: "the", "is", "sunny" 和 "day" 是出现次数最多的四个单词，
//    出现次数依次为 4, 3, 2 和 1 次。
//
// 注意：
// 假定 k 总为有效值， 1 ≤ k ≤ 集合元素数。
// 输入的单词均由小写字母组成。
//
// 扩展练习：
// 尝试以 O(n log k) 时间复杂度和 O(n) 空间复杂度解决。
func topKFrequentII(words []string, k int) []string {

	countMap := make(map[string]int)
	for _, word := range words {
		countMap[word]++
	}

	wordList := make([]string, 0)
	for key := range countMap {
		wordList = append(wordList, key)
	}
	// 排序
	sort.Slice(wordList, func(i, j int) bool {
		count1, count2 := countMap[wordList[i]], countMap[wordList[j]]
		if count1 == count2 {
			return wordList[i] < wordList[j]
		}
		return count2 < count1
	})
	return wordList[:k]
}

// 953. 验证外星语词典
// 某种外星语也使用英文小写字母，但可能顺序 order 不同。字母表的顺序（order）是一些小写字母的排列。
//
// 给定一组用外星语书写的单词 words，以及其字母表的顺序 order，只有当给定的单词在这种外星语中按字典序排列时，返回 true；否则，返回 false。
//
// 示例 1：
// 输入：words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
// 输出：true
// 解释：在该语言的字母表中，'h' 位于 'l' 之前，所以单词序列是按字典序排列的。
//
// 示例 2：
// 输入：words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
// 输出：false
// 解释：在该语言的字母表中，'d' 位于 'l' 之后，那么 words[0] > words[1]，因此单词序列不是按字典序排列的。
//
// 示例 3：
// 输入：words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
// 输出：false
// 解释：当前三个字符 "app" 匹配时，第二个字符串相对短一些，然后根据词典编纂规则 "apple" > "app"，因为 'l' > '∅'，其中 '∅' 是空白字符，定义为比任何其他字符都小（更多信息）。
//
// 提示：
// 1 <= words.length <= 100
// 1 <= words[i].length <= 20
// order.length == 26
// 在 words[i] 和 order 中的所有字符都是英文小写字母。
func isAlienSorted(words []string, order string) bool {
	letters := make([]int, 26)
	for i, c := range order {
		letters[c-'a'] = i
	}

	compareWord := func(word1, word2 string) int {
		m, n := len(word1), len(word2)
		idx := 0
		for idx < m && idx < n {
			c1, c2 := word1[idx], word2[idx]
			num1, num2 := letters[c1-'a'], letters[c2-'a']
			if num1 < num2 {
				return -1
			} else if num1 > num2 {
				return 1
			}
			idx++
		}
		return m - n
	}

	n := len(words)

	for i := 1; i < n; i++ {
		if compareWord(words[i-1], words[i]) > 0 {
			return false
		}
	}

	return true
}

// 1224. 最大相等频率
// 给你一个正整数数组 nums，请你帮忙从该数组中找出能满足下面要求的 最长 前缀，并返回该前缀的长度：
//
// 从前缀中 恰好删除一个 元素后，剩下每个数字的出现次数都相同。
// 如果删除这个元素后没有剩余元素存在，仍可认为每个数字都具有相同的出现次数（也就是 0 次）。
//
// 示例 1：
// 输入：nums = [2,2,1,1,5,3,3,5]
// 输出：7
// 解释：对于长度为 7 的子数组 [2,2,1,1,5,3,3]，如果我们从中删去 nums[4] = 5，就可以得到 [2,2,1,1,3,3]，里面每个数字都出现了两次。
//
// 示例 2：
// 输入：nums = [1,1,1,2,2,2,3,3,3,4,4,4,5]
// 输出：13
//
// 提示：
// 2 <= nums.length <= 105
// 1 <= nums[i] <= 105
func maxEqualFreq(nums []int) int {
	n := len(nums)
	// num 的数量
	numCount := make([]int, 100001)
	// count 的个数
	countNum := make([]int, n+1)
	maxCount, result := 0, 0
	for i, num := range nums {
		numCount[num]++
		count := numCount[num]
		countNum[count-1]--
		countNum[count]++
		maxCount = max(count, maxCount)
		// count 全是1
		if countNum[1] == i+1 {
			result = i + 1
		}
		// count == i + 1 只有一个num
		if count == i+1 {
			result = i + 1
		}
		// count = 1 的 存在且只有1个
		if countNum[1] == 1 && countNum[maxCount]*maxCount == i {
			result = i + 1
		}
		// maxCount-1 有 n 个  maxCount   只有一个
		if countNum[maxCount] == 1 && countNum[maxCount-1]*(maxCount-1)+maxCount == i+1 {
			result = i + 1
		}
	}
	return result
}

// 1460. 通过翻转子数组使两个数组相等
// 给你两个长度相同的整数数组 target 和 arr 。每一步中，你可以选择 arr 的任意 非空子数组 并将它翻转。你可以执行此过程任意次。
//
// 如果你能让 arr 变得与 target 相同，返回 True；否则，返回 False 。
//
// 示例 1：
// 输入：target = [1,2,3,4], arr = [2,4,1,3]
// 输出：true
// 解释：你可以按照如下步骤使 arr 变成 target：
// 1- 翻转子数组 [2,4,1] ，arr 变成 [1,4,2,3]
// 2- 翻转子数组 [4,2] ，arr 变成 [1,2,4,3]
// 3- 翻转子数组 [4,3] ，arr 变成 [1,2,3,4]
// 上述方法并不是唯一的，还存在多种将 arr 变成 target 的方法。
//
// 示例 2：
// 输入：target = [7], arr = [7]
// 输出：true
// 解释：arr 不需要做任何翻转已经与 target 相等。
//
// 示例 3：
// 输入：target = [3,7,9], arr = [3,7,11]
// 输出：false
// 解释：arr 没有数字 9 ，所以无论如何也无法变成 target 。
//
// 提示：
// target.length == arr.length
// 1 <= target.length <= 1000
// 1 <= target[i] <= 1000
// 1 <= arr[i] <= 1000
func canBeEqual(target []int, arr []int) bool {
	nums := make([]int, 1001)
	for _, num := range target {
		nums[num]++
	}
	for _, num := range arr {
		nums[num]--
	}
	for _, num := range nums {
		if num != 0 {
			return false
		}
	}
	return true
}
