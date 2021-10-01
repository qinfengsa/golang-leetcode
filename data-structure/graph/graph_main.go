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

// 210. 课程表 II
// 现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。
//
// 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
// 返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。
//
// 示例 1：
// 输入：numCourses = 2, prerequisites = [[1,0]]
// 输出：[0,1]
// 解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
//
// 示例 2：
// 输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
// 输出：[0,2,1,3]
// 解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
// 因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
//
// 示例 3：
// 输入：numCourses = 1, prerequisites = []
// 输出：[0]
//
// 提示：
// 1 <= numCourses <= 2000
// 0 <= prerequisites.length <= numCourses * (numCourses - 1)
// prerequisites[i].length == 2
// 0 <= ai, bi < numCourses
// ai != bi
// 所有[ai, bi] 匹配 互不相同
//
// 拓展：
// 这个问题相当于查找一个循环是否存在于有向图中。如果存在循环，则不存在拓扑排序，因此不可能选取所有课程进行学习。
// 通过 DFS 进行拓扑排序 - 一个关于Coursera的精彩视频教程（21分钟），介绍拓扑排序的基本概念。
// 拓扑排序也可以通过 BFS 完成。
func findOrder(numCourses int, prerequisites [][]int) []int {

	// 拓扑排序流程
	// 1.将所有入度为0的点加入队列中
	// 2.每次出队一个入度为0的点，然后将该点删除（意思是将所有与该点相连的边都删掉，即将边另一端对应的点的入度减1），
	//   若删除该点后与该点相连的点入度变为了0，则将该点加入队列。
	// 3.重复2过程直到队列中的元素被删完
	//
	// 排除有环的情况
	// 因为只有入度为 0 的点才能入队，故若存在环，环上的点一定无法入队。
	// 所以只需统计入过队的点数之和是否等于点的总数 n 即可。
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
	result := make([]int, numCourses)
	index := 0
	for queue.Len() > 0 {
		front := queue.Front()
		queue.Remove(front)
		preNum := front.Value.(int)
		result[index] = preNum
		index++
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
	if index < numCourses {
		return make([]int, 0)
	}

	return result
}

// 1436. 旅行终点站
// 给你一份旅游线路图，该线路图中的旅行线路用数组 paths 表示，其中 paths[i] = [cityAi, cityBi] 表示该线路将会从 cityAi 直接前往 cityBi 。请你找出这次旅行的终点站，即没有任何可以通往其他城市的线路的城市。
//
// 题目数据保证线路图会形成一条不存在循环的线路，因此恰有一个旅行终点站。
//
// 示例 1：
// 输入：paths = [["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
// 输出："Sao Paulo"
// 解释：从 "London" 出发，最后抵达终点站 "Sao Paulo" 。本次旅行的路线是 "London" -> "New York" -> "Lima" -> "Sao Paulo" 。
//
// 示例 2：
// 输入：paths = [["B","C"],["D","B"],["C","A"]]
// 输出："A"
// 解释：所有可能的线路是：
// "D" -> "B" -> "C" -> "A".
// "B" -> "C" -> "A".
// "C" -> "A".
// "A".
// 显然，旅行终点站是 "A" 。
//
// 示例 3：
// 输入：paths = [["A","Z"]]
// 输出："Z"
//
// 提示：
// 1 <= paths.length <= 100
// paths[i].length == 2
// 1 <= cityAi.length, cityBi.length <= 10
// cityAi != cityBi
// 所有字符串均由大小写英文字母和空格字符组成。
func destCity(paths [][]string) string {
	// 计算出度为0的城市
	inMap, outMap := make(map[string]int), make(map[string]int)

	for _, path := range paths {
		citya, cityb := path[0], path[1]
		inMap[cityb]++
		outMap[citya]++
	}
	for k := range inMap {
		if _, ok := outMap[k]; !ok {
			return k
		}
	}

	return ""
}
