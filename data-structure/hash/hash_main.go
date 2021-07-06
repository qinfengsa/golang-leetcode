package hash

import (
	"sort"
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
		} else {
			indexMap[num] = i
		}
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
	result := []string{}
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
	helf := len(candies) >> 1
	candyMap := map[int]bool{}
	for _, cand := range candies {
		candyMap[cand] = true
	}
	size := len(candyMap)
	if size >= helf {
		return helf
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
	min := 2000
	result := []string{}
	restMap := map[string]int{}
	for i, name := range list1 {
		restMap[name] = i
	}
	for i, name := range list2 {
		if idx, ok := restMap[name]; ok {
			sum := i + idx
			if sum < min {
				min = sum
				result = []string{name}
			} else if sum == min {
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
