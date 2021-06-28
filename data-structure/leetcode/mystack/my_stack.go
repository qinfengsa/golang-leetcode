package mystack

import "container/list"

type MyStack struct {
	queue *list.List
}

// Constructor Initialize your data structure here.
func Constructor() MyStack {
	return MyStack{queue: list.New()}
}

// Push Push element x onto stack.
func (this *MyStack) Push(x int) {
	l := this.queue.Len()
	this.queue.PushBack(x)
	for i := 0; i < l; i++ {
		top := this.queue.Front()
		this.queue.Remove(top)
		this.queue.PushBack(top.Value)
	}

}

// Pop  Removes the element on top of the stack and returns that element.
func (this *MyStack) Pop() int {
	top := this.queue.Front()
	this.queue.Remove(top)
	return top.Value.(int)
}

// Top Get the top element.
func (this *MyStack) Top() int {
	top := this.queue.Front()
	return top.Value.(int)
}

// Empty Returns whether the stack is empty.
func (this *MyStack) Empty() bool {

	return this.queue.Len() == 0
}
