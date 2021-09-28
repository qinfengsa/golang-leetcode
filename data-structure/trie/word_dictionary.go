package trie

type WordDictionary struct {
	children []*WordDictionary
	isEnd    bool
}

func newWordNode() WordDictionary {
	return WordDictionary{
		children: make([]*WordDictionary, 26),
		isEnd:    false,
	}
}

func (this *WordDictionary) AddWord(word string) {
	if len(word) == 0 {
		return
	}
	node := this
	for _, c := range word {
		if node.children[c-'a'] == nil {
			node.children[c-'a'] = &WordDictionary{
				children: make([]*WordDictionary, 26),
				isEnd:    false,
			}
		}
		node = node.children[c-'a']
	}
	node.isEnd = true
}

func (this *WordDictionary) Search(word string) bool {
	if len(word) == 0 {
		return false
	}
	return getNode(word, 0, this)
}

func getNode(word string, start int, node *WordDictionary) bool {
	if start == len(word) {
		return node.isEnd
	}
	c := word[start]
	if c == '.' {
		for _, child := range node.children {
			if child == nil {
				continue
			}
			if getNode(word, start+1, child) {
				return true
			}
		}

	} else if node.children[c-'a'] != nil {
		return getNode(word, start+1, node.children[c-'a'])
	}
	return false
}
