package narytree

type Node struct {
	Val      int
	Children []*Node
}

func maxDepth2(root *Node) int {
	if root == nil {
		return 0
	}
	maxDep := 0
	for _, child := range root.Children {
		dep := maxDepth2(child)
		if dep > maxDep {
			maxDep = dep
		}
	}
	return maxDep + 1
}

// 589. N叉树的前序遍历
// 给定一个 N 叉树，返回其节点值的前序遍历。
//
// 例如，给定一个 3叉树 :
// 返回其前序遍历: [1,3,5,6,2,4]。
// 说明: 递归法很简单，你可以使用迭代法完成此题吗?
func preorder(root *Node) []int {
	var result []int
	var preorderList func(*Node)
	preorderList = func(node *Node) {
		if node == nil {
			return
		}
		result = append(result, node.Val)
		for _, child := range node.Children {
			preorderList(child)
		}
	}
	preorderList(root)
	return result
}

// 590. N叉树的后序遍历
// 给定一个 N 叉树，返回其节点值的后序遍历。
//
// 例如，给定一个 3叉树 :
// 返回其后序遍历: [5,6,3,2,4,1].
// 说明: 递归法很简单，你可以使用迭代法完成此题吗?
func postorder(root *Node) []int {
	var result []int
	var postorderList func(*Node)
	postorderList = func(node *Node) {
		if node == nil {
			return
		}
		for _, child := range node.Children {
			postorderList(child)
		}
		result = append(result, node.Val)

	}
	postorderList(root)
	return result
}
