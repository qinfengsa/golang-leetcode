package circularqueue

// MyCircularQueue
// 622. 设计循环队列
// 设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。
//
// 循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。
//
// 你的实现应该支持如下操作：
//
// MyCircularQueue(k): 构造器，设置队列长度为 k 。
// Front: 从队首获取元素。如果队列为空，返回 -1 。
// Rear: 获取队尾元素。如果队列为空，返回 -1 。
// enQueue(value): 向循环队列插入一个元素。如果成功插入则返回真。
// deQueue(): 从循环队列中删除一个元素。如果成功删除则返回真。
// isEmpty(): 检查循环队列是否为空。
// isFull(): 检查循环队列是否已满。
//
// 示例：
// MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
// circularQueue.enQueue(1);  // 返回 true
// circularQueue.enQueue(2);  // 返回 true
// circularQueue.enQueue(3);  // 返回 true
// circularQueue.enQueue(4);  // 返回 false，队列已满
// circularQueue.Rear();  // 返回 3
// circularQueue.isFull();  // 返回 true
// circularQueue.deQueue();  // 返回 true
// circularQueue.enQueue(4);  // 返回 true
// circularQueue.Rear();  // 返回 4
//
// 提示：
// 所有的值都在 0 至 1000 的范围内；
// 操作数将在 1 至 1000 的范围内；
// 请不要使用内置的队列库。
type MyCircularQueue struct {
	capacity    int
	front, rear int
	elements    []int
}

func Constructor(k int) MyCircularQueue {

	return MyCircularQueue{capacity: k, front: 0, rear: -1, elements: make([]int, k)}
}

// EnQueue 向循环队列插入一个元素。如果成功插入则返回真。
func (this *MyCircularQueue) EnQueue(value int) bool {
	if this.IsFull() {
		return false
	}
	this.rear = (this.rear + 1) % this.capacity
	this.elements[this.rear] = value
	return true
}

// DeQueue 从循环队列中删除一个元素。如果成功删除则返回真。
func (this *MyCircularQueue) DeQueue() bool {
	if this.IsEmpty() {
		return false
	}
	this.elements[this.front] = -1
	if this.front == this.rear {
		this.front = 0
		this.rear = -1
	} else {
		this.front = (this.front + 1) % this.capacity
	}

	return true
}

// Front 从队首获取元素。如果队列为空，返回 -1
func (this *MyCircularQueue) Front() int {
	if this.IsEmpty() {
		return -1
	}
	return this.elements[this.front]
}

// Rear 获取队尾元素。如果队列为空，返回 -1
func (this *MyCircularQueue) Rear() int {
	if this.IsEmpty() {
		return -1
	}
	return this.elements[this.rear]
}

// IsEmpty 检查循环队列是否为空
func (this *MyCircularQueue) IsEmpty() bool {
	return this.rear == -1
}

// IsFull 检查循环队列是否已满
func (this *MyCircularQueue) IsFull() bool {
	if this.IsEmpty() {
		return false
	}
	num := (this.rear - this.front + 1 + this.capacity) % this.capacity
	return num == 0
}

// MyCircularDeque 641. 设计循环双端队列
// 设计实现双端队列。
// 你的实现需要支持以下操作：
//
// MyCircularDeque(k)：构造函数,双端队列的大小为k。
// insertFront()：将一个元素添加到双端队列头部。 如果操作成功返回 true。
// insertLast()：将一个元素添加到双端队列尾部。如果操作成功返回 true。
// deleteFront()：从双端队列头部删除一个元素。 如果操作成功返回 true。
// deleteLast()：从双端队列尾部删除一个元素。如果操作成功返回 true。
// getFront()：从双端队列头部获得一个元素。如果双端队列为空，返回 -1。
// getRear()：获得双端队列的最后一个元素。 如果双端队列为空，返回 -1。
// isEmpty()：检查双端队列是否为空。
// isFull()：检查双端队列是否满了。
//
// 示例：
// MyCircularDeque circularDeque = new MycircularDeque(3); // 设置容量大小为3
// circularDeque.insertLast(1);			        // 返回 true
// circularDeque.insertLast(2);			        // 返回 true
// circularDeque.insertFront(3);			        // 返回 true
// circularDeque.insertFront(4);			        // 已经满了，返回 false
// circularDeque.getRear();  				// 返回 2
// circularDeque.isFull();				        // 返回 true
// circularDeque.deleteLast();			        // 返回 true
// circularDeque.insertFront(4);			        // 返回 true
// circularDeque.getFront();				// 返回 4
//
// 提示：
// 所有值的范围为 [1, 1000]
// 操作次数的范围为 [1, 1000]
// 请不要使用内置的双端队列库。
type MyCircularDeque struct {
	capacity    int
	front, rear int
	elements    []int
}

func Constructor2(k int) MyCircularDeque {
	return MyCircularDeque{capacity: k + 1, front: 0, rear: 0, elements: make([]int, k+1)}
}

func (this *MyCircularDeque) InsertFront(value int) bool {
	if this.IsFull() {
		return false
	}
	this.front = (this.front - 1 + this.capacity) % this.capacity
	this.elements[this.front] = value
	return true
}

func (this *MyCircularDeque) InsertLast(value int) bool {
	if this.IsFull() {
		return false
	}
	this.elements[this.rear] = value
	this.rear = (this.rear + 1) % this.capacity
	return true
}

func (this *MyCircularDeque) DeleteFront() bool {
	if this.IsEmpty() {
		return false
	}
	this.elements[this.front] = 0
	this.front = (this.front + 1) % this.capacity
	if this.front == this.rear {
		this.front = 0
		this.rear = 0
	}
	return true
}

func (this *MyCircularDeque) DeleteLast() bool {
	if this.IsEmpty() {
		return false
	}
	this.rear = (this.rear - 1 + this.capacity) % this.capacity
	this.elements[this.rear] = 0
	if this.front == this.rear {
		this.front = 0
		this.rear = 0
	}
	return true
}

func (this *MyCircularDeque) GetFront() int {
	if this.IsEmpty() {
		return -1
	}
	return this.elements[this.front]
}

func (this *MyCircularDeque) GetRear() int {
	if this.IsEmpty() {
		return -1
	}
	index := (this.rear - 1 + this.capacity) % this.capacity
	return this.elements[index]
}

func (this *MyCircularDeque) IsEmpty() bool {
	return this.front == this.rear
}

func (this *MyCircularDeque) IsFull() bool {
	num := (this.rear - this.front + 1 + this.capacity) % this.capacity
	return num == 0
}
