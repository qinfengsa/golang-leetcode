package tree

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 100. 相同的树
// 给定两个二叉树，编写一个函数来检验它们是否相同。
//
// 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
//
// 示例 1:
// 输入:       1         1
//           / \       / \
//          2   3     2   3
//
//        [1,2,3],   [1,2,3]
// 输出: true
// 示例 2:
// 输入:      1          1
//          /           \
//         2             2
//
//        [1,2],     [1,null,2]
// 输出: false
// 示例 3:
//
// 输入:       1         1
//           / \       / \
//          2   1     1   2
//
//        [1,2,1],   [1,1,2]
// 输出: false
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q != nil {
		if p.Val != q.Val {
			return false
		}
		return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
	}
	return false
}

// 101. 对称二叉树
// 给定一个二叉树，检查它是否是镜像对称的。
// 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
//
//     1
//    / \
//   2   2
//  / \ / \
// 3  4 4  3
//
// 但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
//
//    1
//   / \
//  2   2
//   \   \
//   3    3
//
//
// 进阶：
//
// 你可以运用递归和迭代两种方法解决这个问题吗？
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isSymmetric2(root.Left, root.Right)
}

func isSymmetric2(left *TreeNode, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	}
	if left != nil && right != nil {
		if left.Val != right.Val {
			return false
		}
		return isSymmetric2(left.Left, right.Right) && isSymmetric2(left.Right, right.Left)
	}
	return false
}

// 104. 二叉树的最大深度
// 给定一个二叉树，找出其最大深度。
//
// 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
//
// 说明: 叶子节点是指没有子节点的节点。
//
// 示例：
// 给定二叉树 [3,9,20,null,null,15,7]，
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 返回它的最大深度 3 。
func maxDepth(root *TreeNode) int {
	maxVal = 0
	getDepth(root, 1)
	return maxVal
}

var maxVal int = 0

func getDepth(root *TreeNode, depth int) {
	if root == nil {
		depth = max(maxVal, depth)
		return
	}
	getDepth(root.Left, depth+1)
	getDepth(root.Right, depth+1)
}

// 107. 二叉树的层次遍历 II
// 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
//
// 例如：
// 给定二叉树 [3,9,20,null,null,15,7],
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 返回其自底向上的层次遍历为：
//
// [
//  [15,7],
//  [9,20],
//  [3]
// ]
func levelOrderBottom(root *TreeNode) [][]int {
	queue := list.New()
	result := [][]int{}
	if root == nil {
		return result
	}
	queue.PushBack(root)
	for queue.Len() != 0 {
		size := queue.Len()
		nodeList := []int{}
		for i := 0; i < size; i++ {
			obj := queue.Front()
			queue.Remove(obj)
			value := obj.Value.(*TreeNode)
			nodeList = append(nodeList, value.Val)
			if value.Left != nil {
				queue.PushBack(value.Left)
			}
			if value.Right != nil {
				queue.PushBack(value.Right)
			}

		}
		result = append(result, nodeList)
	}
	left, right := 0, len(result)-1
	for left < right {
		result[left], result[right] = result[right], result[left]
		left++
		right--
	}
	return result
}

// 108. 将有序数组转换为二叉搜索树
// 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
//
// 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
//
// 示例:
//
// 给定有序数组: [-10,-3,0,5,9],
//
// 一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
//
//      0
//     / \
//   -3   9
//   /   /
// -10  5
func sortedArrayToBST(nums []int) *TreeNode {
	// 二分
	return getTreeNodeSortedArray(nums, 0, len(nums)-1)
}

func getTreeNodeSortedArray(nums []int, left int, right int) *TreeNode {
	if left > right {
		return nil
	}
	mid := (left + right) >> 1
	root := &TreeNode{Val: nums[mid]}
	root.Left = getTreeNodeSortedArray(nums, left, mid-1)
	root.Right = getTreeNodeSortedArray(nums, mid+1, right)
	return root
}

func isBalancedTest() {
	root := TreeNode{Val: 1}
	node1 := TreeNode{Val: 2}
	root.Right = &node1
	fmt.Println(isBalanced(&root))
}

// 110. 平衡二叉树
// 给定一个二叉树，判断它是否是高度平衡的二叉树。
//
// 本题中，一棵高度平衡二叉树定义为：
//
// 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
//
// 示例 1: 给定二叉树 [3,9,20,null,null,15,7]
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 返回 true 。
//
// 示例 2: 给定二叉树 [1,2,2,3,3,null,null,4,4]
//
//       1
//      / \
//     2   2
//    / \
//   3   3
//  / \
// 4   4
// 返回 false 。
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return abs(height(root.Left)-height(root.Right)) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)
}

func height(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(height(root.Left), height(root.Right)) + 1
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

// 111. 二叉树的最小深度
// 给定一个二叉树，找出其最小深度。
//
// 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
//
// 说明: 叶子节点是指没有子节点的节点。
//
// 示例:
//
// 给定二叉树 [3,9,20,null,null,15,7],
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 返回它的最小深度  2.
func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == root.Right && root.Left == nil {
		return 1
	}
	ll := minDepth(root.Left)
	lr := minDepth(root.Right)
	if root.Left == nil || root.Right == nil {
		return ll + lr + 1
	}
	if ll > lr {
		return lr + 1
	} else {
		return ll + 1
	}
}

// 112. 路径总和
// 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
//
// 说明: 叶子节点是指没有子节点的节点。
//
// 示例:
// 给定如下二叉树，以及目标和 sum = 22，
//
//              5
//             / \
//            4   8
//           /   / \
//          11  13  4
//         /  \      \
//        7    2      1
// 返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
func hasPathSum(root *TreeNode, sum int) bool {
	// 深度优先遍历
	if root == nil {
		return false
	}
	if root.Left == nil && root.Right == nil && root.Val == sum {
		return true
	}
	return hasPathSum(root.Left, sum-root.Val) || hasPathSum(root.Right, sum-root.Val)

}

// 226. 翻转二叉树
// 翻转一棵二叉树。
//
// 示例：
//
// 输入：
//
//      4
//    /   \
//   2     7
//  / \   / \
// 1   3 6   9
// 输出：
//
//      4
//    /   \
//   7     2
//  / \   / \
// 9   6 3   1
// 备注:
// 这个问题是受到 Max Howell 的 原问题 启发的 ：
//
// 谷歌：我们90％的工程师使用您编写的软件(Homebrew)，但是您却无法在面试时在白板上写出翻转二叉树这道题，这太糟糕了。
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	left, right := root.Left, root.Right
	root.Left = invertTree(right)
	root.Right = invertTree(left)
	return root
}

// 235. 二叉搜索树的最近公共祖先
// 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
//
// 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
//
// 例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
// 示例 1:
//
// 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
// 输出: 6
// 解释: 节点 2 和节点 8 的最近公共祖先是 6。
// 示例 2:
//
// 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
// 输出: 2
// 解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
//
// 说明:
//
// 所有节点的值都是唯一的。
// p、q 为不同节点且均存在于给定的二叉搜索树中。
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil || p == root || q == root {
		return root
	}
	if p.Val > q.Val {
		p, q = q, p
	}
	if q.Val < root.Val {
		return lowestCommonAncestor(root.Left, p, q)
	}
	if root.Val < p.Val {
		return lowestCommonAncestor(root.Right, p, q)
	}
	return root
}

// 257. 二叉树的所有路径
// 给定一个二叉树，返回所有从根节点到叶子节点的路径。
//
// 说明: 叶子节点是指没有子节点的节点。
//
// 示例:
//
// 输入:
//
//    1
//  /   \
// 2     3
//  \
//   5
//
// 输出: ["1->2->5", "1->3"]
//
// 解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
func binaryTreePaths(root *TreeNode) []string {
	treePathReseult = []string{}
	if root == nil {
		return treePathReseult
	}
	treePathDfs(root, "")
	return treePathReseult
}

var treePathReseult []string

func treePathDfs(root *TreeNode, str string) {
	if root == nil {
		return
	}

	if len(str) != 0 {
		str += "->"
	}
	str += fmt.Sprintf("%d", root.Val)
	if root.Left == nil && root.Right == nil {

		treePathReseult = append(treePathReseult, str)
	}
	treePathDfs(root.Left, str)
	treePathDfs(root.Right, str)
}

// 404. 左叶子之和
// 计算给定二叉树的所有左叶子之和。
//
// 示例：
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
//
// 在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
func sumOfLeftLeaves(root *TreeNode) int {
	if root == nil {
		return 0
	}

	return sumOfLeftLeavesLeft(root, false)
}
func sumOfLeftLeavesLeft(root *TreeNode, left bool) int {
	if root == nil {
		return 0
	}
	if left && root.Left == nil && root.Right == nil {
		return root.Val
	}
	result := 0
	result += sumOfLeftLeavesLeft(root.Left, true)
	result += sumOfLeftLeavesLeft(root.Right, false)
	return result
}

// 501. 二叉搜索树中的众数
// 给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。
//
// 假定 BST 有如下定义：
//
// 结点左子树中所含结点的值小于等于当前结点的值
// 结点右子树中所含结点的值大于等于当前结点的值
// 左子树和右子树都是二叉搜索树
// 例如：
// 给定 BST [1,null,2,2],
//
//   1
//    \
//     2
//    /
//   2
// 返回[2].
//
// 提示：如果众数超过1个，不需考虑输出顺序
//
// 进阶：你可以不使用额外的空间吗？（假设由递归产生的隐式调用栈的开销不被计算在内）
func findMode(root *TreeNode) []int {
	prev := -1
	result := []int{}
	count, maxCount := 0, 0
	var inOrder func(*TreeNode)
	inOrder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inOrder(node.Left)
		if prev == node.Val {
			count++
		} else {
			prev, count = node.Val, 1
		}
		if count == maxCount {
			result = append(result, prev)
		} else if count > maxCount {
			maxCount = count
			result = []int{prev}
		}
		inOrder(node.Right)
	}
	// 中序遍历
	inOrder(root)
	return result
}

// 530. 二叉搜索树的最小绝对差
// 给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。
//
// 示例：
// 输入：
//   1
//    \
//     3
//    /
//   2
//
// 输出： 1
//
// 解释： 最小绝对差为 1，其中 2 和 1 的差的绝对值为 1（或者 2 和 3）。
//
// 提示： 树中至少有 2 个节点。
// 本题与 783 https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/ 相同
func getMinimumDifference(root *TreeNode) int {
	min := 1 << 31
	prev := -1
	// 中序遍历
	var inOrder func(*TreeNode)
	inOrder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inOrder(node.Left)
		if prev != -1 {
			num := node.Val - prev
			if num < min {
				min = num
			}
		}
		prev = node.Val
		inOrder(node.Right)
	}
	inOrder(root)
	return min
}

// 543. 二叉树的直径
// 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
//
// 示例 :
// 给定二叉树
//          1
//         / \
//        2   3
//       / \
//      4   5
// 返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。
//
// 注意：两结点之间的路径长度是以它们之间边的数目表示。
func diameterOfBinaryTree(root *TreeNode) int {
	maxTreeLen := 0
	var getDepth2 func(*TreeNode) int
	getDepth2 = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := getDepth2(node.Left)
		right := getDepth2(node.Right)
		maxTreeLen = max(maxTreeLen, left+right)
		return max(left, right) + 1
	}
	getDepth2(root)
	return maxTreeLen
}

// 563. 二叉树的坡度
// 给定一个二叉树，计算 整个树 的坡度 。
// 一个树的 节点的坡度 定义即为，该节点左子树的节点之和和右子树节点之和的 差的绝对值 。如果没有左子树的话，左子树的节点之和为 0 ；没有右子树的话也是一样。空结点的坡度是 0 。
//
// 整个树 的坡度就是其所有节点的坡度之和。
//
// 示例 1：
// 输入：root = [1,2,3] 输出：1
// 解释：
// 节点 2 的坡度：|0-0| = 0（没有子节点）
// 节点 3 的坡度：|0-0| = 0（没有子节点）
// 节点 1 的坡度：|2-3| = 1（左子树就是左子节点，所以和是 2 ；右子树就是右子节点，所以和是 3 ）
// 坡度总和：0 + 0 + 1 = 1
//
// 示例 2：
// 输入：root = [4,2,9,3,5,null,7]  输出：15
// 解释：
// 节点 3 的坡度：|0-0| = 0（没有子节点）
// 节点 5 的坡度：|0-0| = 0（没有子节点）
// 节点 7 的坡度：|0-0| = 0（没有子节点）
// 节点 2 的坡度：|3-5| = 2（左子树就是左子节点，所以和是 3 ；右子树就是右子节点，所以和是 5 ）
// 节点 9 的坡度：|0-7| = 7（没有左子树，所以和是 0 ；右子树正好是右子节点，所以和是 7 ）
// 节点 4 的坡度：|(3+5+2)-(9+7)| = |10-16| = 6（左子树值为 3、5 和 2 ，和是 10 ；右子树值为 9 和 7 ，和是 16 ）
// 坡度总和：0 + 0 + 0 + 2 + 7 + 6 = 15
//
// 示例 3：
// 输入：root = [21,7,14,1,1,2,2,3,3] 输出：9
//
// 提示：
// 树中节点数目的范围在 [0, 104] 内
// -1000 <= Node.val <= 1000
func findTilt(root *TreeNode) int {
	tilt := 0
	var getSumAndTilt func(*TreeNode) int
	getSumAndTilt = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		leftSum := getSumAndTilt(node.Left)
		rightSum := getSumAndTilt(node.Right)

		if leftSum > rightSum {
			tilt += leftSum - rightSum
		} else {
			tilt += rightSum - leftSum
		}
		return node.Val + leftSum + rightSum
	}
	getSumAndTilt(root)
	return tilt
}

// 572. 另一个树的子树
// 给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。
//
// 示例 1:
// 给定的树 s:
//
//     3
//    / \
//   4   5
//  / \
// 1   2
// 给定的树 t：
//
//   4
//  / \
// 1   2
// 返回 true，因为 t 与 s 的一个子树拥有相同的结构和节点值。
//
// 示例 2:
// 给定的树 s：
//
//     3
//    / \
//   4   5
//  / \
// 1   2
//    /
//   0
// 给定的树 t：
//
//   4
//  / \
// 1   2
// 返回 false。
func isSubtree(s *TreeNode, t *TreeNode) bool {
	if s == nil {
		return false
	}
	b1 := false
	if s.Val == t.Val {
		b1 = isSameTree(s, t)
	}
	return b1 || isSubtree(s.Left, t) || isSubtree(s.Right, t)
}

// 606. 根据二叉树创建字符串
// 你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。
//
// 空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树之间的一对一映射关系的空括号对。
//
// 示例 1:
//
// 输入: 二叉树: [1,2,3,4]
//       1
//     /   \
//    2     3
//   /
//  4
//
// 输出: "1(2(4))(3)"
// 解释: 原本将是“1(2(4)())(3())”，
// 在你省略所有不必要的空括号对之后，
// 它将是“1(2(4))(3)”。
//
// 示例 2:
//
// 输入: 二叉树: [1,2,3,null,4]
//       1
//     /   \
//    2     3
//     \
//      4
//
// 输出: "1(2()(4))(3)"
//
// 解释: 和第一个示例相似，
// 除了我们不能省略第一个对括号来中断输入和输出之间的一对一映射关系。
func tree2str(t *TreeNode) string {
	result := ""
	if t == nil {
		return result
	}
	bLeft, bRight := t.Left != nil, t.Right != nil
	if !bLeft && !bRight {
		return fmt.Sprintf("%d", t.Val)
	}
	result += fmt.Sprintf("%d", t.Val)
	result += "("
	result += tree2str(t.Left)
	result += ")"
	if bRight {
		result += "("
		result += tree2str(t.Right)
		result += ")"
	}
	/*var preorder func(node *TreeNode)
	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		result += fmt.Sprintf("%d", node.Val)
		bLeft, bRight := node.Left != nil, node.Right != nil
		if !bLeft && !bRight {
			return
		}

		result += "("
		preorder(node.Left)
		result += ")"
		if bRight {
			result += "("
			preorder(node.Right)
			result += ")"
		}

	}
	preorder(t)*/
	return result
}

// 617. 合并二叉树
// 给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
//
// 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
//
// 示例 1:
//
// 输入:
//	Tree 1                     Tree 2
//          1                         2
//         / \                       / \
//        3   2                     1   3
//       /                           \   \
//      5                             4   7
// 输出:
// 合并后的树:
//	     3
//	    / \
//	   4   5
//	  / \   \
//	 5   4   7
// 注意: 合并必须从两个树的根节点开始。
func mergeTrees(t1 *TreeNode, t2 *TreeNode) *TreeNode {
	if t1 == nil {
		return t2
	}
	if t2 == nil {
		return t1
	}
	t1.Val += t2.Val
	t1.Left = mergeTrees(t1.Left, t2.Left)
	t1.Right = mergeTrees(t1.Right, t2.Right)
	return t1
}

// 637. 二叉树的层平均值
// 给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
// 示例 1：
//
// 输入：
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 输出：[3, 14.5, 11]
// 解释：
// 第 0 层的平均值是 3 ,  第1层是 14.5 , 第2层是 11 。因此返回 [3, 14.5, 11] 。
//
// 提示： 节点值的范围在32位有符号整数范围内。
func averageOfLevels(root *TreeNode) []float64 {
	var result = []float64{}
	if root == nil {
		return result
	}
	queue := list.New()
	queue.PushBack(root)

	var sum float64
	for queue.Len() != 0 {
		size := queue.Len()
		sum = 0.0
		for i := 0; i < size; i++ {
			obj := queue.Front()
			queue.Remove(obj)
			node := obj.Value.(*TreeNode)
			sum += float64(node.Val)
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
		}
		result = append(result, sum/float64(size))
	}

	return result
}

// 653. 两数之和 IV - 输入 BST
// 给定一个二叉搜索树和一个目标结果，如果 BST 中存在两个元素且它们的和等于给定的目标结果，则返回 true。
//
// 案例 1:
//
// 输入:
//     5
//    / \
//   3   6
//  / \   \
// 2   4   7
//
// Target = 9
//
// 输出: True
//
//
// 案例 2:
//
// 输入:
//     5
//    / \
//   3   6
//  / \   \
// 2   4   7
//
// Target = 28
//
// 输出: False
func findTarget(root *TreeNode, k int) bool {
	if root == nil {
		return false
	}
	numMap := map[int]bool{}
	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node == nil {
			return false
		}
		if numMap[k-node.Val] {
			return true
		}
		numMap[node.Val] = true
		return dfs(node.Left) || dfs(node.Right)
	}
	return dfs(root)
}

// 669. 修剪二叉搜索树
// 给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，使得所有节点的值在[low, high]中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。
//
// 所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。
//
// 示例 1：
// 输入：root = [1,0,2], low = 1, high = 2 输出：[1,null,2]
//
// 示例 2：
// 输入：root = [3,0,4,null,2,null,null,1], low = 1, high = 3 输出：[3,2,null,1]
//
// 示例 3：
// 输入：root = [1], low = 1, high = 2 输出：[1]
//
// 示例 4：
// 输入：root = [1,null,2], low = 1, high = 3 输出：[1,null,2]
//
// 示例 5：
// 输入：root = [1,null,2], low = 2, high = 4 输出：[2]
//
// 提示：
// 树中节点数在范围 [1, 104] 内
// 0 <= Node.val <= 104
// 树中每个节点的值都是唯一的
// 题目数据保证输入是一棵有效的二叉搜索树
// 0 <= low <= high <= 104
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val < low {
		return trimBST(root.Right, low, high)
	}
	if root.Val > high {
		return trimBST(root.Left, low, high)
	}
	root.Left = trimBST(root.Left, low, high)
	root.Right = trimBST(root.Right, low, high)
	return root
}

// 671. 二叉树中第二小的节点
// 给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。如果一个节点有两个子节点的话，那么该节点的值等于两个子节点中较小的一个。
//
// 更正式地说，root.val = min(root.left.val, root.right.val) 总成立。
//
// 给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。
//
//
//
// 示例 1：
//
//
// 输入：root = [2,2,5,null,null,5,7]
// 输出：5
// 解释：最小的值是 2 ，第二小的值是 5 。
// 示例 2：
//
//
// 输入：root = [2,2,2]
// 输出：-1
// 解释：最小的值是 2, 但是不存在第二小的值。
func findSecondMinimumValue(root *TreeNode) int {
	if root == nil {
		return -1
	}
	max := (1 << 63) - 1
	min, second := root.Val, max
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Val < min {
			min, second = node.Val, min
		} else if node.Val != min && node.Val < second {
			second = node.Val
		}

		dfs(node.Left)
		dfs(node.Right)

	}
	dfs(root)
	if second == max {
		return -1
	}
	return second
}

// 700. 二叉搜索树中的搜索
// 给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
//
// 例如，
// 给定二叉搜索树:
//
//        4
//       / \
//      2   7
//     / \
//    1   3
//
// 和值: 2
// 你应该返回如下子树:
//
//      2
//     / \
//    1   3
// 在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
func searchBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val == val {
		return root
	}
	if root.Val < val {
		return searchBST(root.Right, val)
	}
	return searchBST(root.Left, val)
}

// 222. 完全二叉树的节点个数
// 给出一个完全二叉树，求出该树的节点个数。
//
// 说明：
// 完全二叉树的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
//
// 示例:
//
// 输入:
//     1
//    / \
//   2   3
//  / \  /
// 4  5 6
//
// 输出: 6
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return 1 + countNodes(root.Left) + countNodes(root.Right)
}

// 897. 递增顺序搜索树
// 给你一棵二叉搜索树，请你 按中序遍历 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。
//
// 示例 1：
// 输入：root = [5,3,6,2,4,null,8,1,null,null,null,7,9]
// 输出：[1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]
//
// 示例 2：
// 输入：root = [5,1,7] 输出：[1,null,5,null,7]
//
// 提示：
// 树中节点数的取值范围是 [1, 100]
// 0 <= Node.val <= 1000
func increasingBST(root *TreeNode) *TreeNode {
	prev := TreeNode{Val: -1, Left: nil, Right: nil}
	curNode := &prev
	// 中序遍历
	var inOrder func(node *TreeNode)
	inOrder = func(node *TreeNode) {
		if node == nil {
			return
		}
		left, right := node.Left, node.Right
		inOrder(left)
		node.Left = nil
		node.Right = nil
		curNode.Right = node
		curNode = curNode.Right
		inOrder(right)
	}
	inOrder(root)
	return prev.Right
}

func rangeSumBST(root *TreeNode, low int, high int) int {
	if root == nil {
		return 0
	}
	if root.Val < low {
		return rangeSumBST(root.Right, low, high)
	}
	if root.Val > high {
		return rangeSumBST(root.Left, low, high)
	}
	result := root.Val
	result += rangeSumBST(root.Left, low, high)
	result += rangeSumBST(root.Right, low, high)
	return result
}

// 94. 二叉树的中序遍历
// 给定一个二叉树的根节点 root ，返回它的 中序 遍历。
//
// 示例 1：
// 输入：root = [1,null,2,3] 输出：[1,3,2]
//
// 示例 2：
// 输入：root = [] 输出：[]
//
// 示例 3：
// 输入：root = [1] 输出：[1]
//
// 示例 4：
// 输入：root = [1,2] 输出：[2,1]
//
// 示例 5：
// 输入：root = [1,null,2] 输出：[1,2]
//
// 提示：
// 树中节点数目在范围 [0, 100] 内
// -100 <= Node.val <= 100
//
// 进阶: 递归算法很简单，你可以通过迭代算法完成吗？
func inorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)

	var inorder func(node *TreeNode)

	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		result = append(result, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return result
}

// 95. 不同的二叉搜索树 II
// 给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。
//
// 示例 1：
// 输入：n = 3
// 输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
//
// 示例 2：
// 输入：n = 1 输出：[[1]]
//
// 提示：
// 1 <= n <= 8
func generateTrees(n int) []*TreeNode {

	var createTree func(start, end int) []*TreeNode

	createTree = func(start, end int) []*TreeNode {
		trees := make([]*TreeNode, 0)
		if start > end {
			trees = append(trees, nil)
			return trees
		}
		for i := start; i <= end; i++ {
			leftTrees := createTree(start, i-1)
			rightTrees := createTree(i+1, end)
			for _, left := range leftTrees {
				for _, right := range rightTrees {
					root := &TreeNode{
						Val:   i,
						Left:  left,
						Right: right,
					}
					trees = append(trees, root)
				}
			}
		}

		return trees
	}

	return createTree(1, n)
}

// 96. 不同的二叉搜索树
// 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。
//
// 示例 1：
// 输入：n = 3 输出：5
//
// 示例 2：
// 输入：n = 1 输出：1
//
// 提示：
// 1 <= n <= 19
func numTrees(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1
	for i := 2; i <= n; i++ {
		count := 0
		for j := 1; j <= i; j++ {
			// 左子树个数 j -1  右子树个数 i - j
			count += dp[j-1] * dp[i-j]
		}
		dp[i] = count
	}

	return dp[n]
}

// 98. 验证二叉搜索树
// 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
//
// 假设一个二叉搜索树具有如下特征：
//
// 节点的左子树只包含小于当前节点的数。
// 节点的右子树只包含大于当前节点的数。
// 所有左子树和右子树自身必须也是二叉搜索树。
//
// 示例 1:
// 输入:
//    2
//   / \
//  1   3
// 输出: true
//
// 示例 2:
// 输入:
//    5
//   / \
//  1   4
//     / \
//    3   6
// 输出: false
// 解释: 输入为: [5,1,4,null,null,3,6]。
//     根节点的值为 5 ，但是其右子节点值为 4 。
func isValidBST(root *TreeNode) bool {

	result := true
	var prevNode *TreeNode
	var inOrder func(*TreeNode)
	inOrder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inOrder(node.Left)
		if prevNode != nil && prevNode.Val >= node.Val {
			result = false
			return
		} else {
			prevNode = node
		}
		inOrder(node.Right)
	}
	// 中序遍历
	inOrder(root)
	return result
}

// 99. 恢复二叉搜索树
// 给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。
//
// 进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？
//
// 示例 1：
// 输入：root = [1,3,null,null,2] 输出：[3,1,null,null,2]
// 解释：3 不能是 1 左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。
//
// 示例 2：
// 输入：root = [3,1,4,null,null,2] 输出：[2,1,4,null,null,3]
// 解释：2 不能在 3 的右子树中，因为 2 < 3 。交换 2 和 3 使二叉搜索树有效。
//
// 提示：
// 树上节点的数目在范围 [2, 1000] 内
// -231 <= Node.val <= 231 - 1
func recoverTree(root *TreeNode) {

	var prevNode, node1, node2 *TreeNode
	var inOrder func(*TreeNode)
	inOrder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inOrder(node.Left)
		if prevNode != nil && prevNode.Val > node.Val {

			node2 = node
			if node1 == nil {
				node1 = prevNode
			} else {
				return
			}

		} else {
			prevNode = node
		}
		inOrder(node.Right)
	}
	// 中序遍历
	inOrder(root)
	node1.Val, node2.Val = node2.Val, node1.Val

}
