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

// 23. 合并K个升序链表
// 给你一个链表数组，每个链表都已经按升序排列。
// 请你将所有链表合并到一个升序链表中，返回合并后的链表。
//
// 示例 1：
// 输入：lists = [[1,4,5],[1,3,4],[2,6]] 输出：[1,1,2,3,4,4,5,6]
// 解释：链表数组如下：
// [
//   1->4->5,
//   1->3->4,
//   2->6
// ]
// 将它们合并到一个有序链表中得到。
// 1->1->2->3->4->4->5->6
//
// 示例 2：
// 输入：lists = [] 输出：[]
//
// 示例 3：
// 输入：lists = [[]] 输出：[]
//
// 提示：
// k == lists.length
// 0 <= k <= 10^4
// 0 <= lists[i].length <= 500
// -10^4 <= lists[i][j] <= 10^4
// lists[i] 按 升序 排列
// lists[i].length 的总和不超过 10^4
func mergeKLists(lists []*ListNode) *ListNode {
	size := len(lists)
	if size == 0 {
		return nil
	}
	if size == 1 {
		return lists[0]
	}

	// 两两合并 直到最后一个
	var merge func(start, end int) *ListNode

	merge = func(start, end int) *ListNode {
		if start == end {
			return lists[start]
		}
		mid := (start + end) >> 1
		left, right := merge(start, mid), merge(mid+1, end)
		return mergeTwoLists(left, right)
	}

	return merge(0, size-1)
}

// 24. 两两交换链表中的节点
// 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
//
// 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
//
// 示例 1：
// 输入：head = [1,2,3,4] 输出：[2,1,4,3]
//
// 示例 2：
// 输入：head = [] 输出：[]
//
// 示例 3：
// 输入：head = [1] 输出：[1]
//
// 提示：
// 链表中节点的数目在范围 [0, 100] 内
// 0 <= Node.val <= 100
//
// 进阶：你能在不修改链表节点值的情况下解决这个问题吗?（也就是说，仅修改节点本身。）
func swapPairs(head *ListNode) *ListNode {
	root := &ListNode{Val: -1}
	root.Next = head
	pre := root

	for pre.Next != nil && pre.Next.Next != nil {
		first, second := pre.Next, pre.Next.Next
		next := second.Next
		pre.Next = second
		second.Next = first
		first.Next = next
		pre = first
	}

	return root.Next
}

// 25. K 个一组翻转链表
//
// 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
// k 是一个正整数，它的值小于或等于链表的长度。
// 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
//
// 进阶：
// 你可以设计一个只使用常数额外空间的算法来解决此问题吗？
// 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
//
// 示例 1：
// 输入：head = [1,2,3,4,5], k = 2 输出：[2,1,4,3,5]
//
// 示例 2：
// 输入：head = [1,2,3,4,5], k = 3 输出：[3,2,1,4,5]
//
// 示例 3：
// 输入：head = [1,2,3,4,5], k = 1 输出：[1,2,3,4,5]
//
// 示例 4：
// 输入：head = [1], k = 1 输出：[1]
//
// 提示：
// 列表中节点的数量在范围 sz 内
// 1 <= sz <= 5000
// 0 <= Node.val <= 1000
// 1 <= k <= sz
func reverseKGroup(head *ListNode, k int) *ListNode {
	if k == 1 {
		return head
	}
	root := &ListNode{Val: -1}
	root.Next = head
	pre := root

out:
	for pre != nil {
		start, last := pre.Next, pre
		for i := 0; i < k; i++ {
			last = last.Next
			if last == nil {
				break out
			}
		}
		next := last.Next
		last.Next = nil
		pre.Next = reverseList(start)
		pre = start
		pre.Next = next
	}

	return root.Next
}

// 61. 旋转链表
// 给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
//
// 示例 1：
// 输入：head = [1,2,3,4,5], k = 2 输出：[4,5,1,2,3]
//
// 示例 2：
// 输入：head = [0,1,2], k = 4 输出：[2,0,1]
//
// 提示：
// 链表中节点的数目在范围 [0, 500] 内
// -100 <= Node.val <= 100
// 0 <= k <= 2 * 109
func rotateRight(head *ListNode, k int) *ListNode {

	getSize := func(node *ListNode) (int, *ListNode) {
		size := 0
		lastNode := node
		for node != nil {
			lastNode = node
			node = node.Next
			size++
		}
		return size, lastNode
	}

	size, lastNode := getSize(head)
	if size <= 1 {
		return head
	}
	k %= size
	if k == 0 {
		return head
	}
	node := head
	for i := 0; i < size-k; i++ {
		node = node.Next
	}
	newHead := node.Next
	node.Next = nil
	lastNode.Next = head
	return newHead
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

// 138. 复制带随机指针的链表
// 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
//
// 构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
//
// 例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
//
// 返回复制链表的头节点。
//
// 用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
//
// val：一个表示 Node.val 的整数。
// random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
// 你的代码 只 接受原链表的头节点 head 作为传入参数。
//
// 示例 1：
// 输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
// 输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
//
// 示例 2：
// 输入：head = [[1,1],[2,1]] 输出：[[1,1],[2,1]]
//
// 示例 3：
// 输入：head = [[3,null],[3,0],[3,null]]
// 输出：[[3,null],[3,0],[3,null]]
//
// 示例 4：
// 输入：head = [] 输出：[]
// 解释：给定的链表为空（空指针），因此返回 null。
//
// 提示：
// 0 <= n <= 1000
// -10000 <= Node.val <= 10000
// Node.random 为空（null）或指向链表中的节点。
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	root := &Node{Val: -1}
	node, newNode := head, root

	nodeMap := make(map[*Node]*Node)
	for node != nil {
		curNode := &Node{Val: node.Val}
		newNode.Next = curNode

		nodeMap[node] = curNode
		node = node.Next
		newNode = newNode.Next
	}
	node, newNode = head, root.Next
	for node != nil {
		if node.Random != nil {
			newNode.Random = nodeMap[node.Random]
		}
		node = node.Next
		newNode = newNode.Next
	}

	return root.Next
}
