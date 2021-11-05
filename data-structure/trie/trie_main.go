package trie

import "sort"

// Trie
// 208. 实现 Trie (前缀树)
// Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。
//
// 请你实现 Trie 类：
// Trie() 初始化前缀树对象。
// void insert(String word) 向前缀树中插入字符串 word 。
// boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
// boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
//
// 示例：
// 输入
// ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
// [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
// 输出
// [null, null, true, false, true, null, true]
// 解释
// Trie trie = new Trie();
// trie.insert("apple");
// trie.search("apple");   // 返回 True
// trie.search("app");     // 返回 False
// trie.startsWith("app"); // 返回 True
// trie.insert("app");
// trie.search("app");     // 返回 True
//
// 提示：
// 1 <= word.length, prefix.length <= 2000
// word 和 prefix 仅由小写英文字母组成
// insert、search 和 startsWith 调用次数 总计 不超过 3 * 104 次
type Trie struct {
	root *Node
}

type Node struct {
	children []*Node
	isEnd    bool
	word     string
}

var SIZE = 26

func Constructor() Trie {
	return Trie{root: newNode()}
}

func newNode() *Node {
	return &Node{
		children: make([]*Node, SIZE),
		isEnd:    false,
	}
}

func (this *Trie) Insert(word string) {

	if len(word) == 0 {
		return
	}
	node := this.root
	for _, c := range word {
		if node.children[c-'a'] == nil {
			node.children[c-'a'] = newNode()
		}
		node = node.children[c-'a']
	}
	node.isEnd = true
	node.word = word
}

func (this *Trie) Search(word string) bool {
	n := len(word)
	if n == 0 {
		return false
	}
	node := this.root
	for i, c := range word {
		if node.children[c-'a'] == nil {
			return false
		}
		node = node.children[c-'a']
		if i == n-1 && node.isEnd {
			return true
		}
	}
	return false
}

func (this *Trie) StartsWith(prefix string) bool {

	if len(prefix) == 0 {
		return false
	}
	node := this.root
	for _, c := range prefix {
		if node.children[c-'a'] == nil {
			return false
		}
		node = node.children[c-'a']
	}
	return true
}

var (
	DirCol = []int{1, -1, 0, 0}
	DirRow = []int{0, 0, 1, -1}
)

func inArea(row, col, rows, cols int) bool {
	return row >= 0 && row < rows && col >= 0 && col < cols
}

// 212. 单词搜索 II
// 给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words，找出所有同时在二维网格和字典中出现的单词。
//
// 单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。
//
// 示例 1：
// 输入：board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
// 输出：["eat","oath"]
//
// 示例 2：
// 输入：board = [["a","b"],["c","d"]], words = ["abcb"]
// 输出：[]
//
// 提示：
// m == board.length
// n == board[i].length
// 1 <= m, n <= 12
// board[i][j] 是一个小写英文字母
// 1 <= words.length <= 3 * 104
// 1 <= words[i].length <= 10
// words[i] 由小写英文字母组成
// words 中的所有字符串互不相同
func findWords(board [][]byte, words []string) []string {
	result := make([]string, 0)
	m, n := len(board), len(board[0])
	trie := &Trie{root: newNode()}

	for _, word := range words {
		trie.Insert(word)
	}
	resultMap := make(map[string]bool)
	var back func(row, col int, node *Node)

	back = func(row, col int, node *Node) {
		c := board[row][col]
		if c == '#' {
			return
		}
		nextNode := node.children[c-'a']
		if nextNode == nil {
			return
		}
		if nextNode.isEnd {
			resultMap[nextNode.word] = true
		}
		board[row][col] = '#'
		for k := 0; k < 4; k++ {
			nextRow, nextCol := row+DirRow[k], col+DirCol[k]
			if !inArea(nextRow, nextCol, m, n) {
				continue
			}
			back(nextRow, nextCol, nextNode)

		}
		board[row][col] = c
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			back(i, j, trie.root)
		}
	}

	for k := range resultMap {
		result = append(result, k)
	}

	return result
}

// 472. 连接词
// 给定一个 不含重复 单词的字符串数组 words ，编写一个程序，返回 words 中的所有 连接词 。
//
// 连接词 的定义为：一个字符串完全是由至少两个给定数组中的单词组成的。
//
// 示例 1：
// 输入：words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
// 输出：["catsdogcats","dogcatsdog","ratcatdogcat"]
// 解释："catsdogcats"由"cats", "dog" 和 "cats"组成;
//     "dogcatsdog"由"dog", "cats"和"dog"组成;
//     "ratcatdogcat"由"rat", "cat", "dog"和"cat"组成。
//
// 示例 2：
// 输入：words = ["cat","dog","catdog"]
// 输出：["catdog"]
//
// 提示：
// 1 <= words.length <= 104
// 0 <= words[i].length <= 1000
// words[i] 仅由小写字母组成
// 0 <= sum(words[i].length) <= 105
func findAllConcatenatedWordsInADict(words []string) []string {

	sort.Slice(words, func(i, j int) bool {
		return len(words[i]) < len(words[j])
	})
	result := make([]string, 0)
	trie := &Trie{root: newNode()}

	var dfs func(word string, start int) bool

	dfs = func(word string, start int) bool {
		node := trie.root
		for i := start; i < len(word); i++ {
			c := word[i]
			if node.children[c-'a'] == nil {
				return false
			}
			node = node.children[c-'a']
			if node.isEnd && dfs(word, i+1) {
				return true
			}
		}
		return node.isEnd && start != 0
	}
	for _, word := range words {

		if len(word) > 0 && dfs(word, 0) {
			result = append(result, word)
		} else {
			trie.Insert(word)
		}
	}

	return result
}
