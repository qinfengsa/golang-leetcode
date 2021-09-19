package lru

import "fmt"

// LRUCache
//
// 146. LRU 缓存机制
// 运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
// 实现 LRUCache 类：
//
// LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
// int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
// void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
//
// 进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？
//
// 示例：
// 输入
// ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
// [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
// 输出
// [null, null, null, 1, null, -1, null, -1, 3, 4]
//
// 解释
// LRUCache lRUCache = new LRUCache(2);
// lRUCache.put(1, 1); // 缓存是 {1=1}
// lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
// lRUCache.get(1);    // 返回 1
// lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
// lRUCache.get(2);    // 返回 -1 (未找到)
// lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
// lRUCache.get(1);    // 返回 -1 (未找到)
// lRUCache.get(3);    // 返回 3
// lRUCache.get(4);    // 返回 4
//
// 提示：
// 1 <= capacity <= 3000
// 0 <= key <= 10000
// 0 <= value <= 105
// 最多调用 2 * 105 次 get 和 put
type LRUCache struct {
	capacity int
	size     int
	cacheMap map[int]*CacheNode
	head     *CacheNode
	tail     *CacheNode
}

type CacheNode struct {
	key   int
	value int
	prev  *CacheNode
	next  *CacheNode
}

func Constructor(capacity int) LRUCache {
	head, tail := &CacheNode{key: -1, value: -1, prev: nil, next: nil}, &CacheNode{key: -1, value: -1, prev: nil, next: nil}
	head.next = tail
	tail.prev = head
	return LRUCache{
		capacity: capacity,
		size:     0,
		cacheMap: make(map[int]*CacheNode),
		head:     head,
		tail:     tail,
	}
}

func (this *LRUCache) Get(key int) int {
	node := this.getNode(key)
	if node != nil {
		return node.value
	}
	return -1
}

func (this *LRUCache) getNode(key int) *CacheNode {
	fmt.Println(this.cacheMap)
	if v, ok := this.cacheMap[key]; ok {
		// 把node移动到最后
		this.removeNode(v)
		this.addTail(v)
		return v
	}
	return nil
}

// 移除 node
func (this *LRUCache) removeNode(node *CacheNode) {
	prev, next := node.prev, node.next
	prev.next = next
	next.prev = prev
}

// 添加到最后
func (this *LRUCache) addTail(node *CacheNode) {
	prev := this.tail.prev
	prev.next, node.prev = node, prev
	node.next, this.tail.prev = this.tail, node
}

func (this *LRUCache) Put(key int, value int) {

	node := this.getNode(key)
	if node != nil {
		node.value = value
		return
	}
	// 容量已满
	if this.size == this.capacity {
		first := this.head.next
		this.removeNode(first)
		delete(this.cacheMap, first.key)
	} else {
		this.size++
	}

	node = &CacheNode{
		key:   key,
		value: value,
		prev:  nil,
		next:  nil,
	}
	this.cacheMap[key] = node
	this.addTail(node)
}
