package allone

import "math"

// AllOne 432. 全 O(1) 的数据结构
type AllOne struct {
	nodeMap    map[string]*OneNode
	head, tail *OneNode
}

type OneNode struct {
	key   string
	value int
	prev  *OneNode
	next  *OneNode
}

func Constructor() AllOne {
	head, tail := &OneNode{key: "", value: 0}, &OneNode{key: "", value: math.MaxInt32}
	head.next = tail
	tail.prev = head
	return AllOne{nodeMap: map[string]*OneNode{}, head: head, tail: tail}
}

func addNode(prev *OneNode, key string) *OneNode {
	next := prev.next
	node := &OneNode{key: key, value: 1}
	prev.next = node
	next.prev = node
	node.prev = prev
	node.next = next
	return node
}

func removeNode(node *OneNode) {
	prev, next := node.prev, node.next
	prev.next = next
	next.prev = prev
	node.next, node.prev = nil, nil
}

// value-- ,向前移
func moveForward(node *OneNode) {
	prev, next := node.prev, node.next
	if prev.value <= node.value {
		return
	}
	// 向前找
	for prev.value > node.value {
		// 向前移动
		prev.next, next.prev = next, prev
		prev, next = prev.prev, prev
	}
	prev.next = node
	next.prev = node
	node.prev = prev
	node.next = next
}

// value++, 向后移
func moveBackward(node *OneNode) {
	prev, next := node.prev, node.next
	if next.value >= node.value {
		return
	}

	// 向后找
	for next.value < node.value {
		// 向后移动
		prev.next, next.prev = next, prev
		prev, next = next, next.next
	}

	prev.next = node
	next.prev = node
	node.prev = prev
	node.next = next
}

// Inc  - 插入一个新的值为 1 的 key。或者使一个存在的 key 增加一，保证 key 不为空字符串。
func (this *AllOne) Inc(key string) {
	node, ok := this.nodeMap[key]
	if ok {
		node.value++
		// 后移
		moveBackward(node)
	} else {
		node = addNode(this.head, key)
		this.nodeMap[key] = node
	}
}

// Dec - 如果这个 key 的值是 1，那么把他从数据结构中移除掉。否则使一个存在的 key 值减一。
// 如果这个 key 不存在，这个函数不做任何事情。key 保证不为空字符串。
func (this *AllOne) Dec(key string) {
	if node, ok := this.nodeMap[key]; ok {
		if node.value == 1 {
			delete(this.nodeMap, key)
			removeNode(node)
		} else {
			node.value--
			// 前移
			moveForward(node)
		}
	}
}

// GetMaxKey  - 返回 key 中值最大的任意一个。如果没有元素存在，返回一个空字符串"" 。
func (this *AllOne) GetMaxKey() string {
	if len(this.nodeMap) == 0 {
		return ""
	}
	return this.tail.prev.key
}

// GetMinKey  - 返回 key 中值最小的任意一个。如果没有元素存在，返回一个空字符串""。
func (this *AllOne) GetMinKey() string {
	if len(this.nodeMap) == 0 {
		return ""
	}
	return this.head.next.key
}
