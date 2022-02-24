package trie

import (
	"container/list"
	"sort"
	"strings"
)

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

// 648. 单词替换
// 在英语中，我们有一个叫做 词根(root)的概念，它可以跟着其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。
// 现在，给定一个由许多词根组成的词典和一个句子。你需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。
// 你需要输出替换之后的句子。
//
// 示例 1：
// 输入：dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
// 输出："the cat was rat by the bat"
//
// 示例 2：
// 输入：dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"
// 输出："a a b c"
//
// 示例 3：
// 输入：dictionary = ["a", "aa", "aaa", "aaaa"], sentence = "a aa a aaaa aaa aaa aaa aaaaaa bbb baba ababa"
// 输出："a a a a a a a a bbb baba a"
//
// 示例 4：
// 输入：dictionary = ["catt","cat","bat","rat"], sentence = "the cattle was rattled by the battery"
// 输出："the cat was rat by the bat"
//
// 示例 5：
// 输入：dictionary = ["ac","ab"], sentence = "it is abnormal that this solution is accepted"
// 输出："it is ab that this solution is ac"
//
// 提示：
// 1 <= dictionary.length <= 1000
// 1 <= dictionary[i].length <= 100
// dictionary[i] 仅由小写字母组成。
// 1 <= sentence.length <= 10^6
// sentence 仅由小写字母和空格组成。
// sentence 中单词的总量在范围 [1, 1000] 内。
// sentence 中每个单词的长度在范围 [1, 1000] 内。
// sentence 中单词之间由一个空格隔开。
// sentence 没有前导或尾随空格。
func replaceWords(dictionary []string, sentence string) string {

	var builder strings.Builder
	trie := &Trie{root: newNode()}
	for _, word := range dictionary {
		trie.Insert(word)
	}
	for _, word := range strings.Split(sentence, " ") {
		if builder.Len() > 0 {
			builder.WriteString(" ")
		}
		builder.WriteString(trie.GetReplaceWord(word))
	}

	return builder.String()
}

func (this *Trie) GetReplaceWord(word string) string {

	node := this.root
	for _, c := range word {
		if node.children[c-'a'] == nil {
			break
		}
		node = node.children[c-'a']
		if node.isEnd {
			break
		}
	}
	if node.isEnd {
		return node.word
	}
	return word
}

// 720. 词典中最长的单词
// 给出一个字符串数组 words 组成的一本英语词典。返回 words 中最长的一个单词，该单词是由 words 词典中其他单词逐步添加一个字母组成。
//
// 若其中有多个可行的答案，则返回答案中字典序最小的单词。若无答案，则返回空字符串。
//
// 示例 1：
// 输入：words = ["w","wo","wor","worl", "world"]
// 输出："world"
// 解释： 单词"world"可由"w", "wo", "wor", 和 "worl"逐步添加一个字母组成。
//
// 示例 2：
// 输入：words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
// 输出："apple"
// 解释："apply" 和 "apple" 都能由词典中的单词组成。但是 "apple" 的字典序小于 "apply"
//
// 提示：
// 1 <= words.length <= 1000
// 1 <= words[i].length <= 30
// 所有输入的字符串 words[i] 都只包含小写字母。
func longestWord(words []string) string {
	trie := &Trie{newNode()}
	for _, word := range words {
		trie.Insert(word)
	}
	return trie.getLongestWord()
}

func (this *Trie) getLongestWord() string {
	result := ""
	if this.root == nil {
		return result
	}
	queue := list.New()
	queue.PushBack(this.root)

	for queue.Len() > 0 {
		front := queue.Front()
		queue.Remove(front)
		node := front.Value.(*Node)
		if !node.isEnd && node != this.root {
			continue
		}
		word := node.word
		// 是完整单词
		if len(word) > len(result) {
			result = word
		} else if len(word) == len(result) && word < result {
			result = word
		}

		for _, child := range node.children {
			if child != nil {
				queue.PushBack(child)
			}
		}

	}

	return result
}
