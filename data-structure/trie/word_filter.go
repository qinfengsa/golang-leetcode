package trie

// WordFilter 745. 前缀和后缀搜索
// 设计一个包含一些单词的特殊词典，并能够通过前缀和后缀来检索单词。
//
// 实现 WordFilter 类：
// WordFilter(string[] words) 使用词典中的单词 words 初始化对象。
// f(string pref, string suff) 返回词典中具有前缀 prefix 和后缀 suff 的单词的下标。如果存在不止一个满足要求的下标，返回其中 最大的下标 。如果不存在这样的单词，返回 -1 。
//
// 示例：
// 输入
// ["WordFilter", "f"]
// [[["apple"]], ["a", "e"]]
// 输出
// [null, 0]
// 解释
// WordFilter wordFilter = new WordFilter(["apple"]);
// wordFilter.f("a", "e"); // 返回 0 ，因为下标为 0 的单词：前缀 prefix = "a" 且 后缀 suff = "e" 。
//
// 提示：
// 1 <= words.length <= 104
// 1 <= words[i].length <= 7
// 1 <= pref.length, suff.length <= 7
// words[i]、pref 和 suff 仅由小写英文字母组成
// 最多对函数 f 执行 104 次调用
type WordFilter struct {
	prefixNode, suffixNode *WordNode
	prefixMap, suffixMap   map[string][]int
}

func ConstructorW(words []string) WordFilter {
	prefixNode, suffixNode := newWordFilterNode(), newWordFilterNode()
	prefixMap, suffixMap := make(map[string][]int), make(map[string][]int)
	wordSet := make(map[string]bool)
	n := len(words)
	for i := n - 1; i >= 0; i-- {
		word := words[i]
		if wordSet[word] {
			continue
		}
		node := prefixNode

		for _, c := range word {
			if node.children[c-'a'] == nil {
				node.children[c-'a'] = newWordFilterNode()
			}
			node = node.children[c-'a']
			node.weight = append(node.weight, i)
		}

		node = suffixNode
		for j := len(word) - 1; j >= 0; j-- {
			c := word[i]
			if node.children[c-'a'] == nil {
				node.children[c-'a'] = newWordFilterNode()
			}
			node = node.children[c-'a']
			node.weight = append(node.weight, j)
		}

	}
	return WordFilter{
		prefixNode, suffixNode, prefixMap, suffixMap,
	}
}

func (this *WordFilter) F(pref string, suff string) int {
	node1, node2 := this.prefixNode, this.suffixNode
	var prefixList, suffixList []int
	if v, ok := this.prefixMap[pref]; ok {
		prefixList = v
	} else {
		for _, c := range pref {
			if node1.children[c-'a'] == nil {
				return -1
			}
			node1 = node1.children[c-'a']
		}
		prefixList = node1.weight
		this.prefixMap[pref] = prefixList
	}
	if v, ok := this.suffixMap[suff]; ok {
		suffixList = v
	} else {
		for i := len(suff) - 1; i >= 0; i-- {
			c := suff[i]
			if node2.children[c-'a'] == nil {
				return -1
			}
			node2 = node2.children[c-'a']
		}
		suffixList = node2.weight
		this.suffixMap[suff] = suffixList
	}

	m, n := len(prefixList), len(suffixList)
	i, j := 0, 0
	for i < m && j < n {
		if prefixList[i] == suffixList[j] {
			return prefixList[i]
		} else if prefixList[i] > suffixList[j] {
			i++
		} else {
			j++
		}
	}
	return -1
}

type WordNode struct {
	children []*WordNode
	weight   []int
}

func newWordFilterNode() *WordNode {
	return &WordNode{
		children: make([]*WordNode, SIZE),
		weight:   make([]int, 0),
	}
}
