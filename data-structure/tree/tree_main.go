package tree

import (
	"container/list"
	"fmt"
	"math"
	"strings"
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
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	queue.PushBack(root)
	for queue.Len() != 0 {
		size := queue.Len()
		nodeList := make([]int, 0)
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
	var getTreeNodeSortedArray func(nums []int, left int, right int) *TreeNode
	getTreeNodeSortedArray = func(nums []int, left int, right int) *TreeNode {
		if left > right {
			return nil
		}
		mid := (left + right) >> 1
		root := &TreeNode{Val: nums[mid]}
		root.Left = getTreeNodeSortedArray(nums, left, mid-1)
		root.Right = getTreeNodeSortedArray(nums, mid+1, right)
		return root
	}
	// 二分
	return getTreeNodeSortedArray(nums, 0, len(nums)-1)
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
	var height func(root *TreeNode) int
	height = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		return max(height(root.Left), height(root.Right)) + 1
	}
	if root == nil {
		return true
	}
	return abs(height(root.Left)-height(root.Right)) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)
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
	treePathReseult := []string{}
	if root == nil {
		return treePathReseult
	}
	var treePathDfs func(root *TreeNode, str string)
	treePathDfs = func(root *TreeNode, str string) {
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
	treePathDfs(root, "")

	return treePathReseult
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
	var sumOfLeft func(node *TreeNode, left bool) int

	sumOfLeft = func(node *TreeNode, left bool) int {
		if node == nil {
			return 0
		}
		if left && node.Left == nil && node.Right == nil {
			return node.Val
		}
		result := 0
		result += sumOfLeft(node.Left, true)
		result += sumOfLeft(node.Right, false)
		return result
	}

	return sumOfLeft(root, false)
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
	var symmetric func(left *TreeNode, right *TreeNode) bool

	symmetric = func(left *TreeNode, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}
		if left != nil && right != nil {
			if left.Val != right.Val {
				return false
			}
			return symmetric(left.Left, right.Right) && symmetric(left.Right, right.Left)
		}
		return false
	}

	return symmetric(root.Left, root.Right)
}

// 102. 二叉树的层序遍历
// 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
//
// 示例：
// 二叉树：[3,9,20,null,null,15,7],
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 返回其层序遍历结果：
// [
//  [3],
//  [9,20],
//  [15,7]
// ]
func levelOrder(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	queue := list.New()
	queue.PushBack(root)

	for queue.Len() > 0 {
		size := queue.Len()
		level := make([]int, 0)
		for i := 0; i < size; i++ {
			front := queue.Front()
			queue.Remove(front)
			node := front.Value.(*TreeNode)
			level = append(level, node.Val)
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
		}
		result = append(result, level)
	}

	return result
}

// 103. 二叉树的锯齿形层序遍历
// 给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
//
// 例如：
// 给定二叉树 [3,9,20,null,null,15,7],
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
// 返回锯齿形层序遍历如下：
//
// [
//  [3],
//  [20,9],
//  [15,7]
// ]
func zigzagLevelOrder(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}

	getNextList := func(treeList []*TreeNode, isLeft bool) []*TreeNode {
		nextList := make([]*TreeNode, 0)
		size := len(treeList)
		for i := size - 1; i >= 0; i-- {
			node := treeList[i]
			if isLeft {
				if node.Left != nil {
					nextList = append(nextList, node.Left)
				}
				if node.Right != nil {
					nextList = append(nextList, node.Right)
				}
			} else {
				if node.Right != nil {
					nextList = append(nextList, node.Right)
				}
				if node.Left != nil {
					nextList = append(nextList, node.Left)
				}
			}
		}

		return nextList
	}

	treeList := []*TreeNode{root}
	isLeft := true
	for len(treeList) > 0 {
		isLeft = !isLeft
		row := make([]int, 0)
		for i := range treeList {
			row = append(row, treeList[i].Val)
		}
		treeList = getNextList(treeList, isLeft)
		result = append(result, row)
	}

	return result
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
	maxVal := 0

	var getDepth func(root *TreeNode, depth int)

	getDepth = func(root *TreeNode, depth int) {
		if root == nil {
			depth = max(maxVal, depth)
			return
		}
		getDepth(root.Left, depth+1)
		getDepth(root.Right, depth+1)
	}

	getDepth(root, 0)
	return maxVal
}

// 105. 从前序与中序遍历序列构造二叉树
//
// 给定一棵树的前序遍历 preorder 与中序遍历  inorder。请构造二叉树并返回其根节点。
//
// 示例 1:
// Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
// Output: [3,9,20,null,null,15,7]
//
// 示例 2:
// Input: preorder = [-1], inorder = [-1] Output: [-1]
//
// 提示:
// 1 <= preorder.length <= 3000
// inorder.length == preorder.length
// -3000 <= preorder[i], inorder[i] <= 3000
// preorder 和 inorder 均无重复元素
// inorder 均出现在 preorder
// preorder 保证为二叉树的前序遍历序列
// inorder 保证为二叉树的中序遍历序列
func buildTree(preorder []int, inorder []int) *TreeNode {

	idxMap := make(map[int]int)
	for i := range inorder {
		idxMap[inorder[i]] = i
	}
	var build func(preStart, preEnd, inStart, inEnd int) *TreeNode

	build = func(preStart, preEnd, inStart, inEnd int) *TreeNode {
		if preStart > preEnd || inStart > inEnd {
			return nil
		}
		rootVal := preorder[preStart]
		root := &TreeNode{
			Val: rootVal,
		}
		inRootIdx := idxMap[rootVal]
		leftLen, rightLen := inRootIdx-inStart, inEnd-inRootIdx
		if leftLen > 0 {
			root.Left = build(preStart+1, preStart+leftLen, inStart, inRootIdx-1)
		}
		if rightLen > 0 {
			root.Right = build(preEnd-rightLen+1, preEnd, inRootIdx+1, inEnd)
		}

		return root
	}
	size := len(preorder)
	return build(0, size-1, 0, size-1)
}

// 106. 从中序与后序遍历序列构造二叉树
//
// 根据一棵树的中序遍历与后序遍历构造二叉树。
//
// 注意:
// 你可以假设树中没有重复的元素。
//
// 例如，给出
//
// 中序遍历 inorder = [9,3,15,20,7]
// 后序遍历 postorder = [9,15,7,20,3]
// 返回如下的二叉树：
//
//    3
//   / \
//  9  20
//    /  \
//   15   7
func buildTreeII(inorder []int, postorder []int) *TreeNode {
	idxMap := make(map[int]int)
	for i := range inorder {
		idxMap[inorder[i]] = i
	}
	var build func(postStart, postEnd, inStart, inEnd int) *TreeNode

	build = func(postStart, postEnd, inStart, inEnd int) *TreeNode {
		if postStart > postEnd || inStart > inEnd {
			return nil
		}
		rootVal := postorder[postEnd]
		root := &TreeNode{
			Val: rootVal,
		}
		inRootIdx := idxMap[rootVal]
		leftLen, rightLen := inRootIdx-inStart, inEnd-inRootIdx
		if leftLen > 0 {
			root.Left = build(postStart, postStart+leftLen-1, inStart, inRootIdx-1)
		}
		if rightLen > 0 {
			root.Right = build(postEnd-rightLen, postEnd-1, inRootIdx+1, inEnd)
		}

		return root
	}
	size := len(inorder)
	return build(0, size-1, 0, size-1)
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 109. 有序链表转换二叉搜索树
// 给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
//
// 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
//
// 示例:
//
// 给定的有序链表： [-10, -3, 0, 5, 9],
//
// 一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：
//
//      0
//     / \
//   -3   9
//   /   /
// -10  5
func sortedListToBST(head *ListNode) *TreeNode {

	var getSize = func(node *ListNode) int {
		size := 0
		for node != nil {
			size++
			node = node.Next
		}
		return size
	}
	size := getSize(head)

	var build func(left, right int) *TreeNode

	build = func(left, right int) *TreeNode {
		if left > right {
			return nil
		}
		mid := (left + right + 1) >> 1
		root := &TreeNode{}
		root.Left = build(left, mid-1)
		root.Val = head.Val
		head = head.Next
		root.Right = build(mid+1, right)

		return root

	}

	return build(0, size-1)
}

// 113. 路径总和 II
// 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
//
// 叶子节点 是指没有子节点的节点。
//
// 示例 1：
// 输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
// 输出：[[5,4,11,2],[5,8,4,5]]
//
// 示例 2：
// 输入：root = [1,2,3], targetSum = 5 输出：[]
//
// 示例 3：
// 输入：root = [1,2], targetSum = 0 输出：[]
//
// 提示：
// 树中节点总数在范围 [0, 5000] 内
// -1000 <= Node.val <= 1000
// -1000 <= targetSum <= 1000
func pathSum(root *TreeNode, targetSum int) [][]int {
	result := make([][]int, 0)
	// 深度优先遍历
	var dfs func(node *TreeNode, sum int, path []int)

	dfs = func(node *TreeNode, sum int, path []int) {
		if node == nil {
			return
		}
		path = append(path, node.Val)
		if node.Left == nil && node.Right == nil && node.Val == sum {
			tmp := make([]int, len(path))
			copy(tmp, path)
			result = append(result, tmp)
			return
		}
		sum -= node.Val

		dfs(node.Left, sum, path)
		dfs(node.Right, sum, path)
	}
	dfs(root, targetSum, make([]int, 0))
	return result
}

// 114. 二叉树展开为链表
// 给你二叉树的根结点 root ，请你将它展开为一个单链表：
//
// 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
// 展开后的单链表应该与二叉树 先序遍历 顺序相同。
//
// 示例 1：
// 输入：root = [1,2,5,3,4,null,6] 输出：[1,null,2,null,3,null,4,null,5,null,6]
//
// 示例 2：
// 输入：root = [] 输出：[]
//
// 示例 3：
// 输入：root = [0] 输出：[0]
//
// 提示：
// 树中结点数在范围 [0, 2000] 内
// -100 <= Node.val <= 100
//
// 进阶：你可以使用原地算法（O(1) 额外空间）展开这棵树吗？
func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	// 先序遍历
	var preorder func(node *TreeNode)

	head := &TreeNode{
		Val: -1,
	}
	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		left, right := node.Left, node.Right
		node.Left, node.Right = nil, nil
		head.Right = node
		head = head.Right
		preorder(left)
		preorder(right)
	}
	preorder(root)
}

type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

// 116. 填充每个节点的下一个右侧节点指针
// 117. 填充每个节点的下一个右侧节点指针 II
// 给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
//
// struct Node {
//  int val;
//  Node *left;
//  Node *right;
//  Node *next;
// }
// 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
//
// 初始状态下，所有 next 指针都被设置为 NULL。
//
// 进阶：
// 你只能使用常量级额外空间。
// 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
//
// 示例：
//
// 输入：root = [1,2,3,4,5,6,7]
// 输出：[1,#,2,3,#,4,5,6,7,#]
// 解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
//
// 提示：
// 树中节点的数量少于 4096
// -1000 <= node.val <= 1000
func connect(root *Node) *Node {

	if root == nil {
		return root
	}
	queue := list.New()
	queue.PushBack(root)

	for queue.Len() > 0 {
		size := queue.Len()
		level := make([]int, 0)
		var pre *Node
		for i := 0; i < size; i++ {
			front := queue.Front()
			queue.Remove(front)
			node := front.Value.(*Node)
			level = append(level, node.Val)
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			if pre != nil {
				pre.Next = node
			}
			pre = node
		}
	}
	return root
}

// 124. 二叉树中的最大路径和
//
// 路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
// 路径和 是路径中各节点值的总和。
// 给你一个二叉树的根节点 root ，返回其 最大路径和 。
//
// 示例 1：
// 输入：root = [1,2,3] 输出：6
// 解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
//
// 示例 2：
// 输入：root = [-10,9,20,null,null,15,7] 输出：42
// 解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
//
// 提示：
// 树中节点数目范围是 [1, 3 * 104]
// -1000 <= Node.val <= 1000
func maxPathSum(root *TreeNode) int {
	maxResult := math.MinInt32

	var getPathSum func(node *TreeNode) int

	getPathSum = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		leftVal, rightVal := getPathSum(node.Left), getPathSum(node.Right)

		maxResult = max(maxResult, leftVal+node.Val+rightVal)

		tmpResult := max(max(leftVal, rightVal), 0) + node.Val

		maxResult = max(maxResult, tmpResult)
		return tmpResult
	}

	getPathSum(root)

	return maxResult
}

// 129. 求根节点到叶节点数字之和
// 给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
// 每条从根节点到叶节点的路径都代表一个数字：
//
// 例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
// 计算从根节点到叶节点生成的 所有数字之和 。
//
// 叶节点 是指没有子节点的节点。
//
// 示例 1：
//
// 输入：root = [1,2,3] 输出：25
// 解释：
// 从根到叶子节点路径 1->2 代表数字 12
// 从根到叶子节点路径 1->3 代表数字 13
// 因此，数字总和 = 12 + 13 = 25
//
// 示例 2：
// 输入：root = [4,9,0,5,1] 输出：1026
// 解释：
// 从根到叶子节点路径 4->9->5 代表数字 495
// 从根到叶子节点路径 4->9->1 代表数字 491
// 从根到叶子节点路径 4->0 代表数字 40
// 因此，数字总和 = 495 + 491 + 40 = 1026
//
// 提示：
// 树中节点的数目在范围 [1, 1000] 内
// 0 <= Node.val <= 9
// 树的深度不超过 10
func sumNumbers(root *TreeNode) int {
	result := 0
	if root == nil {
		return 0
	}

	var dfs func(node *TreeNode, num int)

	dfs = func(node *TreeNode, num int) {
		val := num*10 + node.Val
		if node.Left == nil && node.Right == nil {
			result += val
		}
		if node.Left != nil {
			dfs(node.Left, val)
		}
		if node.Right != nil {
			dfs(node.Right, val)
		}
	}
	dfs(root, 0)
	return result
}

// 144. 二叉树的前序遍历
// 给你二叉树的根节点 root ，返回它节点值的 前序 遍历。
//
// 示例 1：
// 输入：root = [1,null,2,3] 输出：[1,2,3]
//
// 示例 2：
// 输入：root = [] 输出：[]
//
// 示例 3：
// 输入：root = [1] 输出：[1]
//
// 示例 4：
// 输入：root = [1,2] 输出：[1,2]
//
// 示例 5：
// 输入：root = [1,null,2] 输出：[1,2]
//
// 提示：
// 树中节点数目在范围 [0, 100] 内
// -100 <= Node.val <= 100
// 进阶：递归算法很简单，你可以通过迭代算法完成吗？
func preorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	var preorder func(node *TreeNode)

	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		result = append(result, node.Val)
		preorder(node.Left)
		preorder(node.Right)
	}

	preorder(root)

	return result
}

// 145. 二叉树的后序遍历
// 给定一个二叉树，返回它的 后序 遍历。
//
// 示例:
// 输入: [1,null,2,3]
//   1
//    \
//     2
//    /
//   3
//
// 输出: [3,2,1]
// 进阶: 递归算法很简单，你可以通过迭代算法完成吗？
func postorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	var postorder func(node *TreeNode)

	postorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		postorder(node.Left)
		postorder(node.Right)
		result = append(result, node.Val)

	}

	postorder(root)

	return result
}

// 199. 二叉树的右视图
// 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
//
// 示例 1:
// 输入: [1,2,3,null,5,null,4]  输出: [1,3,4]
//
// 示例 2:
// 输入: [1,null,3] 输出: [1,3]
//
// 示例 3:
// 输入: [] 输出: []
//
// 提示:
// 二叉树的节点个数的范围是 [0,100]
// -100 <= Node.val <= 100
func rightSideView(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	// 广度优先遍历
	queue := list.New()
	queue.PushBack(root)

	for queue.Len() > 0 {
		n := queue.Len()
		val := 0
		for i := 0; i < n; i++ {
			front := queue.Front()
			queue.Remove(front)
			node := front.Value.(*TreeNode)
			val = node.Val
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
		}
		result = append(result, val)
	}

	return result
}

// 230. 二叉搜索树中第K小的元素
// 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
//
// 示例 1：
// 输入：root = [3,1,4,null,2], k = 1
// 输出：1
//
// 示例 2：
// 输入：root = [5,3,6,2,4,null,null,1], k = 3
// 输出：3
//
// 提示：
// 树中的节点数为 n 。
// 1 <= k <= n <= 104
// 0 <= Node.val <= 104
// 进阶：如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化算法？
func kthSmallest(root *TreeNode, k int) int {
	// 中序遍历
	result := 0
	var inorder func(node *TreeNode)

	inorder = func(node *TreeNode) {
		if node == nil || k <= 0 {
			return
		}
		inorder(node.Left)
		k--
		if k == 0 {
			result = node.Val
		}
		inorder(node.Right)
	}
	inorder(root)
	return result
}

// 236. 二叉树的最近公共祖先
// 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
//
// 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
//
// 示例 1：
// 输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
// 输出：3
// 解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
//
// 示例 2：
// 输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
// 输出：5
// 解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
//
// 示例 3：
// 输入：root = [1,2], p = 1, q = 2
// 输出：1
//
// 提示：
// 树中节点数目在范围 [2, 105] 内。
// -109 <= Node.val <= 109
// 所有 Node.val 互不相同 。
// p != q
// p 和 q 均存在于给定的二叉树中。
func lowestCommonAncestorII(root, p, q *TreeNode) *TreeNode {
	if root == nil || p == root || q == root {
		return root
	}
	left, right := lowestCommonAncestorII(root.Left, p, q), lowestCommonAncestorII(
		root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left != nil {
		return left
	}
	return right
}

// 331. 验证二叉树的前序序列化
// 序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 #。
//
//      _9_
//     /   \
//    3     2
//   / \   / \
//  4   1  #  6
// / \ / \   / \
// # # # #   # #
// 例如，上面的二叉树可以被序列化为字符串 "9,3,4,#,#,1,#,#,2,#,6,#,#"，其中 # 代表一个空节点。
//
// 给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。
//
// 每个以逗号分隔的字符或为一个整数或为一个表示 null 指针的 '#' 。
//
// 你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 "1,,3" 。
//
// 示例 1:
// 输入: "9,3,4,#,#,1,#,#,2,#,6,#,#"
// 输出: true
//
// 示例 2:
// 输入: "1,#"
// 输出: false
//
// 示例 3:
// 输入: "9,#,#,1"
// 输出: false
func isValidSerialization(preorder string) bool {
	strs := strings.Split(preorder, ",")

	count := 1
	// 一个结点 都会有个两个子结点
	for _, str := range strs {
		count--
		if count < 0 {
			return false
		}
		if str != "#" {
			count += 2
		}
	}
	return count == 0

}

// 337. 打家劫舍 III
// 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。
// 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
// 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
//
// 计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
//
// 示例 1:
// 输入: [3,2,3,null,3,null,1]
//
//     3
//    / \
//   2   3
//    \   \
//     3   1
// 输出: 7
// 解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
//
// 示例 2:
// 输入: [3,4,5,1,3,null,1]
//
//     3
//    / \
//   4   5
//  / \   \
// 1   3   1
// 输出: 9
// 解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
func rob(root *TreeNode) int {

	// 偷当前node 获得的最大金额  不偷当前node 活动的最大金额
	var getRob func(node *TreeNode) (int, int)

	getRob = func(node *TreeNode) (int, int) {
		if node == nil {
			return 0, 0
		}
		leftRobAmount, leftNoRobAmount := getRob(node.Left)
		rightRobAmount, rightNoRobAmount := getRob(node.Right)

		robAmount := node.Val + leftNoRobAmount + rightNoRobAmount
		noRobAmount := max(leftRobAmount, leftNoRobAmount) + max(rightRobAmount, rightNoRobAmount)
		return robAmount, noRobAmount
	}

	robAmount, noRobAmount := getRob(root)
	return max(robAmount, noRobAmount)
}

// 437. 路径总和 III
// 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
//
// 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
//
// 示例 1：
// 输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
// 输出：3
// 解释：和等于 8 的路径有 3 条，如图所示。
//
// 示例 2：
// 输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
// 输出：3
//
// 提示:
// 二叉树的节点个数的范围是 [0,1000]
// -109 <= Node.val <= 109
// -1000 <= targetSum <= 1000
func pathSumIII(root *TreeNode, targetSum int) int {
	count := 0
	// 深度优先遍历
	if root == nil {
		return 0
	}
	count += pathRootSum(root, targetSum)
	count += pathSumIII(root.Left, targetSum)
	count += pathSumIII(root.Right, targetSum)
	return count
}
func pathRootSum(root *TreeNode, targetSum int) int {
	count := 0
	// 深度优先遍历
	if root == nil {
		return 0
	}
	if root.Val == targetSum {
		count++
	}
	count += pathRootSum(root.Left, targetSum-root.Val)
	count += pathRootSum(root.Right, targetSum-root.Val)
	return count
}

// 450. 删除二叉搜索树中的节点
// 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
//
// 一般来说，删除节点可分为两个步骤：
//
// 首先找到需要删除的节点；
// 如果找到了，删除它。
//
// 示例 1:
// 输入：root = [5,3,6,2,4,null,7], key = 3
// 输出：[5,4,6,2,null,null,7]
// 解释：给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
// 一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
// 另一个正确答案是 [5,2,6,null,4,null,7]。
//
// 示例 2:
// 输入: root = [5,3,6,2,4,null,7], key = 0
// 输出: [5,3,6,2,4,null,7]
// 解释: 二叉树不包含值为 0 的节点
//
// 示例 3:
// 输入: root = [], key = 0
// 输出: []
//
// 提示:
// 节点数的范围 [0, 104].
// -105 <= Node.val <= 105
// 节点值唯一
// root 是合法的二叉搜索树
// -105 <= key <= 105
//
// 进阶： 要求算法时间复杂度为 O(h)，h 为树的高度。
func deleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val == key {
		if root.Left == nil {
			return root.Right
		}
		if root.Right == nil {
			return root.Left
		}
		node := root.Left
		if node.Right == nil {
			node.Right = root.Right
		} else {
			rightNode := node.Right
			for rightNode.Right != nil {
				rightNode = rightNode.Right
			}
			rightNode.Right = root.Right
		}
		return node
	}
	if root.Val > key {
		root.Left = deleteNode(root.Left, key)
	} else {
		root.Right = deleteNode(root.Right, key)
	}

	return root
}
