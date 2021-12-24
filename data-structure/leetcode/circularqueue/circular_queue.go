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
