package mapsum

// MapSum 677. 键值映射
// 实现一个 MapSum 类，支持两个方法，insert 和 sum：
//
// MapSum() 初始化 MapSum 对象
// void insert(String key, int val) 插入 key-val 键值对，字符串表示键 key ，整数表示值 val 。如果键 key 已经存在，那么原来的键值对将被替代成新的键值对。
// int sum(string prefix) 返回所有以该前缀 prefix 开头的键 key 的值的总和。
//
// 示例：
// 输入：
// ["MapSum", "insert", "sum", "insert", "sum"]
// [[], ["apple", 3], ["ap"], ["app", 2], ["ap"]]
// 输出：
// [null, null, 3, null, 5]
// 解释：
// MapSum mapSum = new MapSum();
// mapSum.insert("apple", 3);
// mapSum.sum("ap");           // return 3 (apple = 3)
// mapSum.insert("app", 2);
// mapSum.sum("ap");           // return 5 (apple + app = 3 + 2 = 5)
//
// 提示：
// 1 <= key.length, prefix.length <= 50
// key 和 prefix 仅由小写英文字母组成
// 1 <= val <= 1000
// 最多调用 50 次 insert 和 sum
type MapSum struct {
	valMap map[string]int
	root   *Node
}

type Node struct {
	children [26]*Node
	val      int
}

func Constructor() MapSum {
	return MapSum{valMap: make(map[string]int), root: newNode()}
}

func newNode() *Node {
	return &Node{val: 0}
}

func (this *MapSum) Insert(key string, val int) {
	node := this.root
	oldVal, ok := this.valMap[key]
	for i := 0; i < len(key); i++ {
		c := key[i]
		if node.children[c-'a'] == nil {
			node.children[c-'a'] = newNode()
		}
		node = node.children[c-'a']
		if ok {
			node.val += val - oldVal
		} else {
			node.val += val
		}

	}
	this.valMap[key] = val
}

func (this *MapSum) Sum(prefix string) int {
	node := this.root
	for i := 0; i < len(prefix); i++ {
		c := prefix[i]
		if node.children[c-'a'] == nil {
			return 0
		}
		node = node.children[c-'a']
	}
	return node.val
}
