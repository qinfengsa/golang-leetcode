package tree

import "container/list"

// CBTInserter 919. 完全二叉树插入器
// 完全二叉树 是每一层（除最后一层外）都是完全填充（即，节点数达到最大）的，并且所有的节点都尽可能地集中在左侧。
//
// 设计一种算法，将一个新节点插入到一个完整的二叉树中，并在插入后保持其完整。
//
// 实现 CBTInserter 类:
//
// CBTInserter(TreeNode root) 使用头节点为 root 的给定树初始化该数据结构；
// CBTInserter.insert(int v)  向树中插入一个值为 Node.val == val的新节点 TreeNode。使树保持完全二叉树的状态，并返回插入节点 TreeNode 的父节点的值；
// CBTInserter.get_root() 将返回树的头节点。
//
// 示例 1：
// 输入
// ["CBTInserter", "insert", "insert", "get_root"]
// [[[1, 2]], [3], [4], []]
// 输出
// [null, 1, 2, [1, 2, 3, 4]]
// 解释
// CBTInserter cBTInserter = new CBTInserter([1, 2]);
// cBTInserter.insert(3);  // 返回 1
// cBTInserter.insert(4);  // 返回 2
// cBTInserter.get_root(); // 返回 [1, 2, 3, 4]
//
// 提示：
// 树中节点数量范围为 [1, 1000]
// 0 <= Node.val <= 5000
// root 是完全二叉树
// 0 <= val <= 5000
// 每个测试用例最多调用 insert 和 get_root 操作 104 次
type CBTInserter struct {
	root       *TreeNode
	queue      *list.List
	lastParent *TreeNode
}

func ConstructorCBT(root *TreeNode) CBTInserter {
	queue := list.New()
	var lastNode *TreeNode
	queue.PushBack(root)
	for queue.Len() > 0 {
		front := queue.Front()
		queue.Remove(front)
		node := front.Value.(*TreeNode)
		if node.Left == nil {
			lastNode = node
			break
		} else {
			queue.PushBack(node.Left)
		}
		if node.Right == nil {
			lastNode = node
			break
		} else {
			queue.PushBack(node.Right)
		}
	}
	return CBTInserter{root: root, queue: queue, lastParent: lastNode}
}

func (this *CBTInserter) Insert(val int) int {
	result := this.lastParent.Val
	node := &TreeNode{Val: val}
	if this.lastParent.Left == nil {
		this.lastParent.Left = node
		this.queue.PushBack(node)
	} else if this.lastParent.Right == nil {
		this.lastParent.Right = node
		this.queue.PushBack(node)
		front := this.queue.Front()
		this.queue.Remove(front)
		this.lastParent = front.Value.(*TreeNode)
	}
	return result
}

func (this *CBTInserter) Get_root() *TreeNode {
	return this.root
}
