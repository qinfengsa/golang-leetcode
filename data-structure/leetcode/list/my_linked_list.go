package list

// MyLinkedList 707. 设计链表
// 设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。
//
// 在链表类中实现这些功能：
// get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
// addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
// addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
// addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
// deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。
//
// 示例：
// MyLinkedList linkedList = new MyLinkedList();
// linkedList.addAtHead(1);
// linkedList.addAtTail(3);
// linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
// linkedList.get(1);            //返回2
// linkedList.deleteAtIndex(1);  //现在链表是1-> 3
// linkedList.get(1);            //返回3
//
// 提示：
// 所有val值都在 [1, 1000] 之内。
// 操作次数将在  [1, 1000] 之内。
// 请不要使用内置的 LinkedList 库。
type MyLinkedList struct {
	head, tail *Node
	size       int
}

type Node struct {
	val        int
	prev, next *Node
}

func Constructor() MyLinkedList {
	head, tail := &Node{val: -1}, &Node{val: -1}
	head.next, tail.prev = tail, head
	return MyLinkedList{head: head, tail: tail, size: 0}
}

func (this *MyLinkedList) Get(index int) int {
	if this.size == 0 {
		return -1
	}
	node := this.head.next
	for i := 0; node != nil; i++ {
		if index == i {
			return node.val
		}
		node = node.next
	}
	return -1
}

func (this *MyLinkedList) AddAtHead(val int) {
	addAfter(this.head, val)
	this.size++
}

func (this *MyLinkedList) AddAtTail(val int) {
	addBefore(this.tail, val)
	this.size++
}

func addBefore(next *Node, val int) {
	node := &Node{val: val}
	prev := next.prev
	prev.next, node.prev = node, prev
	node.next, next.prev = next, node
}
func addAfter(prev *Node, val int) {
	node := &Node{val: val}
	next := prev.next
	prev.next, node.prev = node, prev
	node.next, next.prev = next, node
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
	// 如果 index 大于链表长度，则不会插入节点。
	if index > this.size {
		return
	}

	if index == this.size {
		this.size++
		addBefore(this.tail, val)
		return
	}
	node := this.head
	for i := 0; node != nil; i++ {
		if index == i {
			addAfter(node, val)
			break
		}
		node = node.next
	}
	this.size++
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
	if index < 0 || index >= this.size {
		return
	}
	this.size--
	prev, node := this.head, this.head.next
	for i := 0; node != nil; i++ {
		if i == index {
			next := node.next
			prev.next, next.prev = next, prev
			break
		}
		prev = node
		node = node.next
	}
}
