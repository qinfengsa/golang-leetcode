package list

import "log"

type ListNode struct {
	Val  int
	Next *ListNode
}

// 21. 合并两个有序链表
// 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
// 示例： 输入：1->2->4, 1->3->4 输出：1->1->2->3->4->4
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	head := ListNode{Val: -1}
	node := &head
	for l1 != nil || l2 != nil {
		if l1 == nil {
			node.Next = l2
			break
		}
		if l2 == nil {
			node.Next = l1
			break
		}
		if l1.Val <= l2.Val {
			node.Next = l1
			l1 = l1.Next
		} else {
			node.Next = l2
			l2 = l2.Next
		}
		node = node.Next
	}

	return head.Next
}

// 83. 删除排序链表中的重复元素
// 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
//
// 示例 1:
//
// 输入: 1->1->2
// 输出: 1->2
// 示例 2:
//
// 输入: 1->1->2->3->3
// 输出: 1->2->3
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	root, node := head, head
	result := root
	for node != nil {
		next := node.Next
		if node.Val != root.Val {
			node.Next = nil
			root.Next = node
			root = root.Next
		}

		node = next
	}
	root.Next = nil
	return result
}

// 141. 环形链表
// 给定一个链表，判断链表中是否有环。
//
// 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
//
// 如果链表中存在环，则返回 true 。 否则，返回 false 。
//
//
//
// 进阶： 你能用 O(1)（即，常量）内存解决此问题吗？
// 示例 1： 输入：head = [3,2,0,-4], pos = 1 输出：true
// 解释：链表中有一个环，其尾部连接到第二个节点。
// 示例 2： 输入：head = [1,2], pos = 0 输出：true
// 解释：链表中有一个环，其尾部连接到第一个节点。
// 示例 3： 输入：head = [1], pos = -1 输出：false
// 解释：链表中没有环。
//
// 提示：链表中节点的数目范围是 [0, 104]
// -105 <= Node.val <= 105
// pos 为 -1 或者链表中的一个 有效索引 。
func hasCycle(head *ListNode) bool {
	if head == nil {
		return false
	}
	// 快慢指针
	fast, slow := head, head
	for slow.Next != nil && fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}

	return false
}

// 142. 环形链表 II
// 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
//
// 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
//
// 说明：不允许修改给定的链表。
// 示例 1： 输入：head = [3,2,0,-4], pos = 1 输出：tail connects to node index 1
// 解释：链表中有一个环，其尾部连接到第二个节点。
//
// 示例 2： 输入：head = [1,2], pos = 0 输出：tail connects to node index 0
// 解释：链表中有一个环，其尾部连接到第一个节点。
//
// 示例 3： 输入：head = [1], pos = -1 输出：no cycle
// 解释：链表中没有环。
// 进阶： 你是否可以不用额外空间解决此题？
func detectCycle(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	// 快慢指针
	fast, slow := head, head
	var node *ListNode
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			node = slow
			break
		}
	}
	if node == nil {
		return nil
	}
	fast = node
	slow = head
	for fast != slow {
		fast = fast.Next
		slow = slow.Next
	}

	return slow
}

// 编写一个程序，找到两个单链表相交的起始节点。
//
// 如下面的两个链表：
// 在节点 c1 开始相交。
//
// 示例 1： 输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
// 输出：Reference of the node with value = 8
// 输入解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
//
// 示例2：
// 输入：intersectVal= 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
// 输出：Reference of the node with value = 2
// 输入解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
//
// 示例3：
// 输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
// 输出：null
// 输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
// 解释：这两个链表不相交，因此返回 null。
//
// 注意：
// 如果两个链表没有交点，返回 null.
// 在返回结果后，两个链表仍须保持原有的结构。
// 可假定整个链表结构中没有循环。
// 程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	node1, node2 := headA, headB
	for node1 != nil && node2 != nil {
		node1 = node1.Next
		node2 = node2.Next
	}
	node3, node4 := headA, headB
	for node1 != nil {
		node1 = node1.Next
		node3 = node3.Next
	}
	for node2 != nil {
		node2 = node2.Next
		node4 = node4.Next
	}

	for node3 != nil && node3 != node4 {
		node3 = node3.Next
		node4 = node4.Next
	}
	return node3
}

// 203. 移除链表元素
// 删除链表中等于给定值 val 的所有节点。
//
// 示例:
//
// 输入: 1->2->6->3->4->5->6, val = 6
// 输出: 1->2->3->4->5
func removeElements(head *ListNode, val int) *ListNode {
	root := &ListNode{Val: -1}
	pre, node := root, head
	for node != nil {
		next := node.Next
		if node.Val != val {
			node.Next = nil
			pre.Next = node
			pre = pre.Next
		}
		node = next
	}
	return root.Next
}

func reverseListTest() {
	head := ListNode{Val: 1}
	node1, node2, node3, node4 := ListNode{Val: 2}, ListNode{Val: 3}, ListNode{Val: 4}, ListNode{Val: 5}
	head.Next = &node1
	node1.Next = &node2
	node2.Next = &node3
	node3.Next = &node4
	root := reverseList(&head)
	log.Print(root)
}

// 206. 反转链表
// 反转一个单链表。
//
// 示例:
//
// 输入: 1->2->3->4->5->NULL
// 输出: 5->4->3->2->1->NULL
// 进阶:
// 你可以迭代或递归地反转链表。你能否用两种方法解决这道题？
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	var pre *ListNode
	node := head
	for node != nil {
		next := node.Next
		node.Next = pre
		pre = node
		node = next
	}
	return pre
}

// 234. 回文链表
// 请判断一个链表是否为回文链表。
//
// 示例 1:
//
// 输入: 1->2 输出: false
// 示例 2:
//
// 输入: 1->2->2->1 输出: true
// 进阶：
// 你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？
func isPalindrome2(head *ListNode) bool {
	if head == nil {
		return false
	}
	// 快慢指针 分成两个链表
	slow, fast := head, head
	var pre *ListNode
	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	if pre == nil {
		return true
	}
	//  1 2 3 4 5 6
	pre.Next = nil
	// 翻转 slow
	slow = reverseList(slow)
	for head != nil && slow != nil {
		if head.Val != slow.Val {
			return false
		}
		head, slow = head.Next, slow.Next
	}

	return true
}

// 237. 删除链表中的节点
// 请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点。传入函数的唯一参数为 要被删除的节点 。
// 现有一个链表 -- head = [4,5,1,9]，它可以表示为:
// 示例 1：
//
// 输入：head = [4,5,1,9], node = 5 输出：[4,1,9]
// 解释：给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
// 示例 2：
//
// 输入：head = [4,5,1,9], node = 1 输出：[4,5,9]
// 解释：给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
//
// 提示：
//
// 链表至少包含两个节点。
// 链表中所有节点的值都是唯一的。
// 给定的节点为非末尾节点并且一定是链表中的一个有效节点。
// 不要从你的函数中返回任何结果。
func deleteNode(node *ListNode) {
	// 将node节点的值替换
	next := node.Next
	node.Val = next.Val
	node.Next = next.Next
}

// 328. 奇偶链表
// 给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。
//
// 请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。
//
// 示例 1:
//
// 输入: 1->2->3->4->5->NULL
// 输出: 1->3->5->2->4->NULL
// 示例 2:
//
// 输入: 2->1->3->5->6->4->7->NULL
// 输出: 2->3->6->7->1->5->4->NULL
// 说明:
//
// 应当保持奇数节点和偶数节点的相对顺序。
// 链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推。
func oddEvenList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	oddNode, evenNode := &ListNode{Val: -1}, &ListNode{Val: -1}
	evenHead := evenNode

	node := head
	index := 1
	for node != nil {
		next := node.Next
		node.Next = nil
		if index&1 == 1 {

			oddNode.Next = node
			oddNode = oddNode.Next
		} else {
			evenNode.Next = node
			evenNode = evenNode.Next
		}
		index++
		node = next
	}
	oddNode.Next = evenHead.Next

	return head
}

// 2. 两数相加
//
// 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
// 请你将两个数相加，并以相同形式返回一个表示和的链表。
// 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
//
// 示例 1：
// 输入：l1 = [2,4,3], l2 = [5,6,4] 输出：[7,0,8]
// 解释：342 + 465 = 807.
//
// 示例 2：
// 输入：l1 = [0], l2 = [0] 输出：[0]
//
// 示例 3：
// 输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9] 输出：[8,9,9,9,0,0,0,1]
//
// 提示：
// 每个链表中的节点数在范围 [1, 100] 内
// 0 <= Node.val <= 9
// 题目数据保证列表表示的数字不含前导零
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {

	head := &ListNode{Val: -1}
	node := head
	last := 0
	for l1 != nil || l2 != nil {
		v1, v2 := 0, 0
		if l1 != nil {
			v1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			v2 = l2.Val
			l2 = l2.Next
		}
		num := v1 + v2 + last

		if num >= 10 {
			num -= 10
			last = 1
		} else {
			last = 0
		}
		node.Next = &ListNode{Val: num}
		node = node.Next
	}
	if last == 1 {
		node.Next = &ListNode{Val: 1}
	}

	return head.Next
}

// 19. 删除链表的倒数第 N 个结点
// 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
//
// 进阶：你能尝试使用一趟扫描实现吗？
//
// 示例 1：
// 输入：head = [1,2,3,4,5], n = 2 输出：[1,2,3,5]
//
// 示例 2：
// 输入：head = [1], n = 1 输出：[]
//
// 示例 3：
// 输入：head = [1,2], n = 1 输出：[1]
//
// 提示：
// 链表中结点的数目为 sz
// 1 <= sz <= 30
// 0 <= Node.val <= 100
// 1 <= n <= sz
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	root := &ListNode{Val: -1}
	root.Next = head

	fastNode, slowNode := root, root

	for i := 0; i < n; i++ {
		fastNode = fastNode.Next
	}
	preNode := root
	for fastNode != nil {
		fastNode = fastNode.Next
		slowNode = slowNode.Next
	}

	preNode.Next = slowNode.Next

	return root.Next
}
