package myqueue

import "container/list"

type MyQueue struct {
	// 入队栈
	pushStack *list.List
	// 出队栈
	popStack *list.List
}

// Constructor   Initialize your data structure here.
func Constructor() MyQueue {
	return MyQueue{popStack: list.New(), pushStack: list.New()}
}

// Push   element x to the back of queue.
func (this *MyQueue) Push(x int) {
	this.pushStack.PushBack(x)
}

// Pop Removes the element from in front of queue and returns that element.
func (this *MyQueue) Pop() int {
	size := this.popStack.Len()
	if size == 0 {
		this.setPop()
	}
	val := this.popStack.Front()
	this.popStack.Remove(val)
	return val.Value.(int)
}

// Peek  Get the front element.
func (this *MyQueue) Peek() int {
	size := this.popStack.Len()
	if size == 0 {
		this.setPop()
	}
	val := this.popStack.Front()
	return val.Value.(int)
}

func (this *MyQueue) setPop() {
	size := this.pushStack.Len()
	for i := 0; i < size; i++ {
		val := this.pushStack.Front()
		this.pushStack.Remove(val)
		this.popStack.PushBack(val.Value)
	}
}

/** Returns whether the queue is empty. */
func (this *MyQueue) Empty() bool {
	size1, size2 := this.popStack.Len(), this.pushStack.Len()
	return size1 == 0 && size2 == 0
}
