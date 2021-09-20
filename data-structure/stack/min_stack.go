package stack

// MinStack
//
// 155. 最小栈
// 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
//
// push(x) —— 将元素 x 推入栈中。
// pop() —— 删除栈顶的元素。
// top() —— 获取栈顶元素。
// getMin() —— 检索栈中的最小元素。
//
// 示例:
// 输入：
// ["MinStack","push","push","push","getMin","pop","top","getMin"]
// [[],[-2],[0],[-3],[],[],[],[]]
// 输出：
// [null,null,null,null,-3,null,0,-2]
//
// 解释：
// MinStack minStack = new MinStack();
// minStack.push(-2);
// minStack.push(0);
// minStack.push(-3);
// minStack.getMin();   --> 返回 -3.
// minStack.pop();
// minStack.top();      --> 返回 0.
// minStack.getMin();   --> 返回 -2.
//
// 提示：
// pop、top 和 getMin 操作总是在 非空栈 上调用。
type MinStack struct {
	size int
	top  *MinStackNode
}

type MinStackNode struct {
	val  int
	min  int
	next *MinStackNode
}

/** initialize your data structure here. */
func Constructor() MinStack {
	node := &MinStackNode{val: 0}
	return MinStack{size: 0, top: node}
}

func (this *MinStack) Push(val int) {
	this.size++
	node := &MinStackNode{val: val}
	if this.size == 1 {
		node.min = val
	} else {
		node.min = min(val, this.top.min)
	}
	node.next = this.top
	this.top = node
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func (this *MinStack) Pop() {
	if this.size == 0 {
		return
	}
	this.size--
	next := this.top.next
	this.top = next
}

func (this *MinStack) Top() int {
	if this.size == 0 {
		return -1
	}
	return this.top.val
}

func (this *MinStack) GetMin() int {
	if this.size == 0 {
		return -1
	}
	return this.top.min
}
