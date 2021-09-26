package graph

import "container/list"

type Node struct {
	Val       int
	Neighbors []*Node
}

// 133. 克隆图
// 给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。
//
// 图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。
// class Node {
//    public int val;
//    public List<Node> neighbors;
// }
//
// 测试用例格式：
// 简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。
// 邻接列表 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。
// 给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对克隆图的引用返回。
//
// 示例 1：
// 输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
// 输出：[[2,4],[1,3],[2,4],[1,3]]
// 解释：
// 图中有 4 个节点。
// 节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
// 节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
// 节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
// 节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
//
// 示例 2：
// 输入：adjList = [[]]
// 输出：[[]]
// 解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。
//
// 示例 3：
// 输入：adjList = [] 输出：[]
// 解释：这个图是空的，它不含任何节点。
//
// 示例 4：
// 输入：adjList = [[2],[1]]  输出：[[2],[1]]
//
// 提示：
// 节点数不超过 100 。
// 每个节点值 Node.val 都是唯一的，1 <= Node.val <= 100。
// 无向图是一个简单图，这意味着图中没有重复的边，也没有自环。
// 由于图是无向的，如果节点 p 是节点 q 的邻居，那么节点 q 也必须是节点 p 的邻居。
// 图是连通图，你可以从给定节点访问到所有节点。
func cloneGraph(node *Node) *Node {
	if node == nil {
		return nil
	}
	graphMap := make(map[int]*Node)

	var getGraphNode func(root *Node) *Node

	getGraphNode = func(root *Node) *Node {
		if root == nil {
			return nil
		}
		if v, ok := graphMap[root.Val]; ok {
			return v
		}
		newNode := &Node{
			Val: root.Val,
		}
		graphMap[newNode.Val] = newNode
		tmpNeighbors := make([]*Node, 0)
		for _, child := range root.Neighbors {
			tmpNeighbors = append(tmpNeighbors, getGraphNode(child))
		}
		newNode.Neighbors = tmpNeighbors

		return newNode
	}

	return getGraphNode(node)
}

// 207. 课程表
// 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
//
// 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
//
// 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
// 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：numCourses = 2, prerequisites = [[1,0]]
// 输出：true
// 解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
//
// 示例 2：
// 输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
// 输出：false
// 解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
//
// 提示：
// 1 <= numCourses <= 105
// 0 <= prerequisites.length <= 5000
// prerequisites[i].length == 2
// 0 <= ai, bi < numCourses
// prerequisites[i] 中的所有课程对 互不相同
func canFinish(numCourses int, prerequisites [][]int) bool {
	// 入度
	inDegrees := make([]int, numCourses)
	for i := range prerequisites {
		inDegrees[prerequisites[i][0]]++
	}
	// 寻找入度为0的点 需要先学（不需要学习前置课程）
	queue := list.New()
	for i, num := range inDegrees {
		if num == 0 {
			queue.PushBack(i)
		}
	}

	for queue.Len() > 0 {
		front := queue.Front()
		queue.Remove(front)
		numCourses--
		preNum := front.Value.(int)

		for i := range prerequisites {
			// 前置课程
			if prerequisites[i][1] != preNum {
				continue
			}
			inDegrees[prerequisites[i][0]]--
			if inDegrees[prerequisites[i][0]] == 0 {
				queue.PushBack(prerequisites[i][0])
			}

		}

	}

	return numCourses == 0
}
