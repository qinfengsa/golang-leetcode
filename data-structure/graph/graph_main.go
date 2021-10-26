package graph

import (
	"container/list"
	"sort"
)

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

// 310. 最小高度树
// 树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。
//
// 给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。
// 可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。
// 请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。
//
// 树的 高度 是指根节点和叶子节点之间最长向下路径上边的数量。
//
// 示例 1：
// 输入：n = 4, edges = [[1,0],[1,2],[1,3]]
// 输出：[1]
// 解释：如图所示，当根是标签为 1 的节点时，树的高度是 1 ，这是唯一的最小高度树。
//
// 示例 2：
// 输入：n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
// 输出：[3,4]
//
// 示例 3：
// 输入：n = 1, edges = []
// 输出：[0]
//
// 示例 4：
// 输入：n = 2, edges = [[0,1]]
// 输出：[0,1]
//
// 提示：
// 1 <= n <= 2 * 104
// edges.length == n - 1
// 0 <= ai, bi < n
// ai != bi
// 所有 (ai, bi) 互不相同
// 给定的输入 保证 是一棵树，并且 不会有重复的边
func findMinHeightTrees(n int, edges [][]int) []int {
	if n == 1 {
		return []int{0}
	}
	degree := make([]int, n)
	graph := make([][]int, n)
	for i := 0; i < n; i++ {
		graph[i] = make([]int, 0)
	}
	for _, edge := range edges {
		a, b := edge[0], edge[1]
		degree[a]++
		degree[b]++
		graph[a] = append(graph[a], b)
		graph[b] = append(graph[b], a)
	}
	queue := list.New()
	for i := 0; i < n; i++ {
		if degree[i] == 1 {
			queue.PushBack(i)
		}
	}
	var result []int
	for queue.Len() > 0 {
		result = make([]int, 0)
		size := queue.Len()
		for i := 0; i < size; i++ {
			front := queue.Front()
			queue.Remove(front)
			num := front.Value.(int)
			result = append(result, num)
			for _, next := range graph[num] {
				degree[next]--
				if degree[next] == 1 {
					queue.PushBack(next)
				}
			}
		}
	}

	return result
}

// 332. 重新安排行程
// 给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。
//
// 所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典排序返回最小的行程组合。
//
// 例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。
// 假定所有机票至少存在一种合理的行程。且所有的机票 必须都用一次 且 只能用一次。
//
// 示例 1：
// 输入：tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
// 输出：["JFK","MUC","LHR","SFO","SJC"]
//
// 示例 2：
// 输入：tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
// 输出：["JFK","ATL","JFK","SFO","ATL","SFO"]
// 解释：另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"] ，但是它字典排序更大更靠后。
//
// 提示：
// 1 <= tickets.length <= 300
// tickets[i].length == 2
// fromi.length == 3
// toi.length == 3
// fromi 和 toi 由大写英文字母组成
// fromi != toi
func findItinerary(tickets [][]string) []string {
	graph := make(map[string][]string)
	result := make([]string, 0)

	for _, ticket := range tickets {
		from, to := ticket[0], ticket[1]
		graph[from] = append(graph[from], to)
	}
	for _, v := range graph {
		sort.Strings(v)
	}

	// 深度优先遍历
	var dfs func(from string)

	dfs = func(from string) {
		for {
			if v, ok := graph[from]; !ok || len(v) == 0 {
				break
			}
			tolist := graph[from]
			next := tolist[0]
			graph[from] = tolist[1:]
			dfs(next)
		}

		result = append(result, from)
	}

	dfs("JFK")
	left, right := 0, len(result)-1
	for left < right {
		result[left], result[right] = result[right], result[left]
		left++
		right--
	}

	return result
}

// 399. 除法求值
// 给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。
// 另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。
// 返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。
// 注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。
//
// 示例 1：
// 输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
// 输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
// 解释：
// 条件：a / b = 2.0, b / c = 3.0
// 问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
// 结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]
//
// 示例 2：
// 输入：equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
// 输出：[3.75000,0.40000,5.00000,0.20000]
//
// 示例 3：
// 输入：equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
// 输出：[0.50000,2.00000,-1.00000,-1.00000]
//
// 提示：
// 1 <= equations.length <= 20
// equations[i].length == 2
// 1 <= Ai.length, Bi.length <= 5
// values.length == equations.length
// 0.0 < values[i] <= 20.0
// 1 <= queries.length <= 20
// queries[i].length == 2
// 1 <= Cj.length, Dj.length <= 5
// Ai, Bi, Cj, Dj 由小写英文字母与数字组成
func calcEquation(equations [][]string, values []float64, queries [][]string) []float64 {
	n := len(queries)
	result := make([]float64, n)
	size := len(equations)
	parent, vals := make([]int, size<<1), make([]float64, size<<1)
	for i := 0; i < len(parent); i++ {
		parent[i] = i
		vals[i] = 1.0
	}
	id := 0
	idMap := make(map[string]int)
	// 并查集  通过 parent 找到对应的 变量
	// 通过valMap 找到变量对应的值
	// getParent 找到对应的 变量
	var find func(x int) int
	find = func(x int) int {
		if x != parent[x] {
			origin := parent[x]
			parent[x] = find(origin)
			vals[x] *= vals[origin]
		}
		return parent[x]
	}
	union := func(x, y int, val float64) {
		rootX, rootY := find(x), find(y)
		if rootX == rootY {
			return
		}
		parent[rootX] = rootY
		vals[rootX] = vals[y] * val / vals[x]
	}
	calc := func(x, y int) float64 {
		rootX, rootY := find(x), find(y)
		if rootX == rootY {
			return vals[x] / vals[y]
		}
		return -1.0
	}

	for i, equation := range equations {
		a, b := equation[0], equation[1]
		if _, ok := idMap[a]; !ok {
			idMap[a] = id
			id++
		}
		if _, ok := idMap[b]; !ok {
			idMap[b] = id
			id++
		}
		union(idMap[a], idMap[b], values[i])
	}
	for i, query := range queries {
		a, b := query[0], query[1]
		if _, ok := idMap[a]; !ok {
			result[i] = -1.0
			continue
		}
		if _, ok := idMap[b]; !ok {
			result[i] = -1.0
			continue
		}
		result[i] = calc(idMap[a], idMap[b])
	}

	return result
}
