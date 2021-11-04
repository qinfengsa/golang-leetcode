package lfu

// LFUCache 460. LFU 缓存
// 请你为 最不经常使用（LFU）缓存算法设计并实现数据结构。
//
// 实现 LFUCache 类：
// LFUCache(int capacity) - 用数据结构的容量 capacity 初始化对象
// int get(int key) - 如果键存在于缓存中，则获取键的值，否则返回 -1。
// void put(int key, int value) - 如果键已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量时，则应该在插入新项之前，使最不经常使用的项无效。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 最近最久未使用 的键。
// 注意「项的使用次数」就是自插入该项以来对其调用 get 和 put 函数的次数之和。使用次数会在对应项被移除后置为 0 。
//
// 为了确定最不常使用的键，可以为缓存中的每个键维护一个 使用计数器 。使用计数最小的键是最久未使用的键。
//
// 当一个键首次插入到缓存中时，它的使用计数器被设置为 1 (由于 put 操作)。对缓存中的键执行 get 或 put 操作，使用计数器的值将会递增。
//
// 示例：
// 输入：
// ["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
// [[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
// 输出：
// [null, null, null, 1, null, -1, 3, null, -1, 3, 4]
//
// 解释：
// // cnt(x) = 键 x 的使用计数
// // cache=[] 将显示最后一次使用的顺序（最左边的元素是最近的）
// LFUCache lFUCache = new LFUCache(2);
// lFUCache.put(1, 1);   // cache=[1,_], cnt(1)=1
// lFUCache.put(2, 2);   // cache=[2,1], cnt(2)=1, cnt(1)=1
// lFUCache.get(1);      // 返回 1
//                      // cache=[1,2], cnt(2)=1, cnt(1)=2
// lFUCache.put(3, 3);   // 去除键 2 ，因为 cnt(2)=1 ，使用计数最小
//                      // cache=[3,1], cnt(3)=1, cnt(1)=2
// lFUCache.get(2);      // 返回 -1（未找到）
// lFUCache.get(3);      // 返回 3
//                      // cache=[3,1], cnt(3)=2, cnt(1)=2
// lFUCache.put(4, 4);   // 去除键 1 ，1 和 3 的 cnt 相同，但 1 最久未使用
//                      // cache=[4,3], cnt(4)=1, cnt(3)=2
// lFUCache.get(1);      // 返回 -1（未找到）
// lFUCache.get(3);      // 返回 3
//                      // cache=[3,4], cnt(4)=1, cnt(3)=3
// lFUCache.get(4);      // 返回 4
//                      // cache=[3,4], cnt(4)=2, cnt(3)=3
//
// 提示：
// 0 <= capacity, key, value <= 104
// 最多调用 105 次 get 和 put 方法
//
// 进阶：你可以为这两种操作设计时间复杂度为 O(1) 的实现吗？
type LFUCache struct {
	minFreq  int // 最小频率
	capacity int
	size     int
	cacheMap map[int]*Node
	freqMap  map[int]*DoublyLinkedList // 频率 -> Node 双向链表
}

type Node struct {
	key   int
	value int
	freq  int
	prev  *Node
	next  *Node
}

type DoublyLinkedList struct {
	head, tail *Node
}

func newList() *DoublyLinkedList {
	head, tail := &Node{key: -1, value: -1, prev: nil, next: nil}, &Node{key: -1, value: -1, prev: nil, next: nil}
	head.next = tail
	tail.prev = head
	return &DoublyLinkedList{head: head, tail: tail}
}

// 移除 node
func removeNode(node *Node) {
	prev, next := node.prev, node.next
	prev.next = next
	next.prev = prev
}

// 添加到最后
func addNode(prev, node *Node) {
	next := prev.next
	prev.next, next.prev = node, node
	node.next, node.prev = next, prev
}

func Constructor(capacity int) LFUCache {
	return LFUCache{
		capacity: capacity,
		size:     0,
		minFreq:  0,
		cacheMap: make(map[int]*Node),
		freqMap:  map[int]*DoublyLinkedList{},
	}
}
func (this *LFUCache) Get(key int) int {
	if this.capacity == 0 {
		return -1
	}
	node, ok := this.cacheMap[key]
	if !ok {
		return -1
	}
	this.freqInc(node)
	return node.value
}

// 频率增加
func (this *LFUCache) freqInc(node *Node) {
	freq := node.freq

	linkedList := this.freqMap[freq]
	removeNode(node)

	if freq == this.minFreq && linkedList.head.next == linkedList.tail {
		this.minFreq = freq + 1
	}
	// 加入新freq对应的链表
	node.freq++
	var ok bool
	linkedList, ok = this.freqMap[freq+1]
	if !ok {
		linkedList = newList()
		this.freqMap[freq+1] = linkedList
	}
	addNode(linkedList.head, node)

}

func (this *LFUCache) Put(key int, value int) {
	if this.capacity == 0 {
		return
	}
	node, ok := this.cacheMap[key]
	if ok {
		node.value = value
		this.freqInc(node)
	} else {
		// 新增 node
		if this.size == this.capacity {
			linkedList := this.freqMap[this.minFreq]

			minNode := linkedList.tail.prev
			delete(this.cacheMap, minNode.key)

			removeNode(minNode)

			this.size--
		}
		node = &Node{key: key, value: value, freq: 1}
		this.cacheMap[key] = node
		linkedList, freqExists := this.freqMap[1]
		if !freqExists {
			linkedList = newList()
			this.freqMap[1] = linkedList
		}
		this.minFreq = 1
		addNode(linkedList.head, node)
		this.size++
	}
}
