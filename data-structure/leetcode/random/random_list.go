package random

import "math/rand"

type ListNode struct {
	Val  int
	Next *ListNode
}

// Solution
// 382. 链表随机节点
// 给定一个单链表，随机选择链表的一个节点，并返回相应的节点值。保证每个节点被选的概率一样。
//
// 进阶:
// 如果链表十分大且长度未知，如何解决这个问题？你能否使用常数级空间复杂度实现？
//
// 示例:
// // 初始化一个单链表 [1,2,3].
// ListNode head = new ListNode(1);
// head.next = new ListNode(2);
// head.next.next = new ListNode(3);
// Solution solution = new Solution(head);
// // getRandom()方法应随机返回1,2,3中的一个，保证每个元素被返回的概率相等。
// solution.getRandom();
type Solution struct {
	head *ListNode
}

func Constructor3(head *ListNode) Solution {

	return Solution{
		head: head,
	}
}

func (this *Solution) GetRandom() int {
	result, node := this.head.Val, this.head.Next
	size := 2
	// 蓄水池抽样算法
	for node != nil {
		if rand.Intn(size) == 0 {
			result = node.Val
		}
		size++
		node = node.Next
	}

	return result
}
