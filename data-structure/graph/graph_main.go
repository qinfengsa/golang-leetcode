package graph

import (
	"container/list"
	"math"
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
//
//	class Node {
//	   public int val;
//	   public List<Node> neighbors;
//	}
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

// 684. 冗余连接
// 树可以看成是一个连通且 无环 的 无向 图。
//
// 给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。
//
// 请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。
//
// 示例 1：
// 输入: edges = [[1,2], [1,3], [2,3]]
// 输出: [2,3]
//
// 示例 2：
// 输入: edges = [[1,2], [2,3], [3,4], [1,4], [1,5]]
// 输出: [1,4]
//
// 提示:
// n == edges.length
// 3 <= n <= 1000
// edges[i].length == 2
// 1 <= ai < bi <= edges.length
// ai != bi
// edges 中无重复元素
// 给定的图是连通的
func findRedundantConnection(edges [][]int) []int {
	n := len(edges)
	nums := make([]int, n+1)
	for i := 0; i <= n; i++ {
		nums[i] = i
	}

	var findParent func(num int) int

	findParent = func(num int) int {
		if nums[num] == num {
			return num
		}
		return findParent(nums[num])
	}

	union := func(num1, num2 int) {
		nums[num2] = num1
	}

	for _, edge := range edges {
		num1, num2 := findParent(edge[0]), findParent(edge[1])
		if num1 == num2 {
			return edge
		}
		union(num1, num2)
	}

	return nil
}

// 685. 冗余连接 II
// 在本问题中，有根树指满足以下条件的 有向 图。该树只有一个根节点，所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点，而根节点没有父节点。
//
// 输入一个有向图，该图由一个有着 n 个节点（节点值不重复，从 1 到 n）的树及一条附加的有向边构成。附加的边包含在 1 到 n 中的两个不同顶点间，这条附加的边不属于树中已存在的边。
//
// 结果图是一个以边组成的二维数组 edges 。 每个元素是一对 [ui, vi]，用以表示 有向 图中连接顶点 ui 和顶点 vi 的边，其中 ui 是 vi 的一个父节点。
//
// 返回一条能删除的边，使得剩下的图是有 n 个节点的有根树。若有多个答案，返回最后出现在给定二维数组的答案。
//
// 示例 1：
// 输入：edges = [[1,2],[1,3],[2,3]]
// 输出：[2,3]
//
// 示例 2：
// 输入：edges = [[1,2],[2,3],[3,4],[4,1],[1,5]]
// 输出：[4,1]
//
// 提示：
// n == edges.length
// 3 <= n <= 1000
// edges[i].length == 2
// 1 <= ui, vi <= n
func findRedundantDirectedConnection(edges [][]int) []int {
	n := len(edges)
	nums := make([]int, n+1)
	for i := 0; i <= n; i++ {
		nums[i] = i
	}
	var findParent func(num int) int

	findParent = func(num int) int {
		if nums[num] == num {
			return num
		}
		return findParent(nums[num])
	}
	union := func(num1, num2 int) {
		nums[num2] = num1
	}
	// 1.入度全为1， 则找出构成环的最后一条边
	// 2.有度为2的两条边(A->B, C->B)，则删除的边一定是在其中
	// 先不将C->B加入并查集中，若不能构成环，则C->B是需要删除的点边，
	// 反之，则A->B是删除的边(去掉C->B还能构成环，则C->B一定不是要删除的边)
	inDegree := make([]int, n+1)
	repeatNode := -1
	for _, edge := range edges {
		inDegree[edge[1]]++
		if inDegree[edge[1]] == 2 {
			repeatNode = edge[1]
		}
	}
	var result []int
	if repeatNode == -1 {
		// 入度全为1
		for _, edge := range edges {
			num1, num2 := findParent(edge[0]), findParent(edge[1])
			if num1 == num2 {
				return edge
			}
			union(num1, num2)
		}
	} else {
		nodes := make([][]int, 0)
		for _, edge := range edges {
			if edge[1] == repeatNode {
				nodes = append(nodes, edge)
				continue
			}
			num1, num2 := findParent(edge[0]), findParent(edge[1])
			union(num1, num2)
		}
		for _, node := range nodes {
			num1, num2 := findParent(node[0]), findParent(node[1])
			if num1 == num2 {
				return node
			}
			union(num1, num2)
		}

	}
	return result

}

// 1791. 找出星型图的中心节点
// 有一个无向的 星型 图，由 n 个编号从 1 到 n 的节点组成。星型图有一个 中心 节点，并且恰有 n - 1 条边将中心节点与其他每个节点连接起来。
//
// 给你一个二维整数数组 edges ，其中 edges[i] = [ui, vi] 表示在节点 ui 和 vi 之间存在一条边。请你找出并返回 edges 所表示星型图的中心节点。
//
// 示例 1：
// 输入：edges = [[1,2],[2,3],[4,2]]
// 输出：2
// 解释：如上图所示，节点 2 与其他每个节点都相连，所以节点 2 是中心节点。
//
// 示例 2：
// 输入：edges = [[1,2],[5,1],[1,3],[1,4]]
// 输出：1
//
// 提示：
// 3 <= n <= 105
// edges.length == n - 1
// edges[i].length == 2
// 1 <= ui, vi <= n
// ui != vi
// 题目数据给出的 edges 表示一个有效的星型图
func findCenter(edges [][]int) int {
	n := len(edges) + 1
	degree := make([]int, n+1)
	for _, edge := range edges {
		degree[edge[0]]++
		degree[edge[1]]++
	}
	for i := 1; i <= n; i++ {
		if degree[i] == n-1 {
			return i
		}
	}
	return -1
}

// 剑指 Offer II 114. 外星文字典
// 现有一种使用英语字母的外星文语言，这门语言的字母顺序与英语顺序不同。
//
// 给定一个字符串列表 words ，作为这门语言的词典，words 中的字符串已经 按这门新语言的字母顺序进行了排序 。
//
// 请你根据该词典还原出此语言中已知的字母顺序，并 按字母递增顺序 排列。若不存在合法字母顺序，返回 "" 。若存在多种可能的合法字母顺序，返回其中 任意一种 顺序即可。
//
// 字符串 s 字典顺序小于 字符串 t 有两种情况：
//
// 在第一个不同字母处，如果 s 中的字母在这门外星语言的字母顺序中位于 t 中字母之前，那么 s 的字典顺序小于 t 。
// 如果前面 min(s.length, t.length) 字母都相同，那么 s.length < t.length 时，s 的字典顺序也小于 t 。
//
// 示例 1：
// 输入：words = ["wrt","wrf","er","ett","rftt"]
// 输出："wertf"
//
// 示例 2：
// 输入：words = ["z","x"]
// 输出："zx"
//
// 示例 3：
// 输入：words = ["z","x","z"]
// 输出：""
// 解释：不存在合法字母顺序，因此返回 "" 。
//
// 提示：
// 1 <= words.length <= 100
// 1 <= words[i].length <= 100
// words[i] 仅由小写英文字母组成
//
// 注意：本题与主站 269 题相同： https://leetcode-cn.com/problems/alien-dictionary/
func alienOrder(words []string) string {
	graph := make(map[byte][]byte)
	for _, c := range words[0] {
		graph[byte(c)] = nil
	}
next:
	for i := 1; i < len(words); i++ {
		prev, word := words[i-1], words[i]
		for _, c := range word {
			graph[byte(c)] = graph[byte(c)]
		}
		for j := 0; j < len(prev) && j < len(word); j++ {
			if prev[j] != word[j] {
				graph[prev[j]] = append(graph[prev[j]], word[j])
				continue next
			}
		}
		if len(prev) > len(word) {
			return ""
		}
	}
	order := make([]byte, len(graph))
	state := make(map[byte]int)
	const visiting, visited = 1, 2
	idx := len(graph) - 1

	var dfs func(c byte) bool

	dfs = func(c byte) bool {
		state[c] = visiting
		for _, v := range graph[c] {
			if state[v] == 0 {
				if !dfs(v) {
					return false
				}

			} else if state[v] == visiting {
				return false
			}
		}
		order[idx] = c
		idx--
		state[c] = visited
		return true
	}
	for c := range graph {
		if state[c] == 0 && !dfs(c) {
			return ""
		}
	}

	return string(order)
}

// 剑指 Offer II 115. 重建序列
// 给定一个长度为 n 的整数数组 nums ，其中 nums 是范围为 [1，n] 的整数的排列。还提供了一个 2D 整数数组 sequences ，其中 sequences[i] 是 nums 的子序列。
// 检查 nums 是否是唯一的最短 超序列 。最短 超序列 是 长度最短 的序列，并且所有序列 sequences[i] 都是它的子序列。对于给定的数组 sequences ，可能存在多个有效的 超序列 。
//
// 例如，对于 sequences = [[1,2],[1,3]] ，有两个最短的 超序列 ，[1,2,3] 和 [1,3,2] 。
// 而对于 sequences = [[1,2],[1,3],[1,2,3]] ，唯一可能的最短 超序列 是 [1,2,3] 。[1,2,3,4] 是可能的超序列，但不是最短的。
// 如果 nums 是序列的唯一最短 超序列 ，则返回 true ，否则返回 false 。
// 子序列 是一个可以通过从另一个序列中删除一些元素或不删除任何元素，而不改变其余元素的顺序的序列。
//
// 示例 1：
// 输入：nums = [1,2,3], sequences = [[1,2],[1,3]]
// 输出：false
// 解释：有两种可能的超序列：[1,2,3]和[1,3,2]。
// 序列 [1,2] 是[1,2,3]和[1,3,2]的子序列。
// 序列 [1,3] 是[1,2,3]和[1,3,2]的子序列。
// 因为 nums 不是唯一最短的超序列，所以返回false。
//
// 示例 2：
// 输入：nums = [1,2,3], sequences = [[1,2]]
// 输出：false
// 解释：最短可能的超序列为 [1,2]。
// 序列 [1,2] 是它的子序列：[1,2]。
// 因为 nums 不是最短的超序列，所以返回false。
//
// 示例 3：
// 输入：nums = [1,2,3], sequences = [[1,2],[1,3],[2,3]]
// 输出：true
// 解释：最短可能的超序列为[1,2,3]。
// 序列 [1,2] 是它的一个子序列：[1,2,3]。
// 序列 [1,3] 是它的一个子序列：[1,2,3]。
// 序列 [2,3] 是它的一个子序列：[1,2,3]。
// 因为 nums 是唯一最短的超序列，所以返回true。
//
// 提示：
// n == nums.length
// 1 <= n <= 104
// nums 是 [1, n] 范围内所有整数的排列
// 1 <= sequences.length <= 104
// 1 <= sequences[i].length <= 104
// 1 <= sum(sequences[i].length) <= 105
// 1 <= sequences[i][j] <= n
// sequences 的所有数组都是 唯一 的
// sequences[i] 是 nums 的一个子序列
//
// 注意：本题与主站 444 题相同：https://leetcode-cn.com/problems/sequence-reconstruction/
func sequenceReconstruction(nums []int, sequences [][]int) bool {
	n := len(nums)
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
	inDegrees := make([]int, n+1)
	graph := make([][]int, n+1)
	for _, sequence := range sequences {
		for i := 1; i < len(sequence); i++ {
			from, to := sequence[i-1], sequence[i]
			inDegrees[to]++
			graph[from] = append(graph[from], to)
		}
	}
	// 寻找入度为0的点 起点
	queue := list.New()
	for i := 1; i <= n; i++ {
		if inDegrees[i] == 0 {
			queue.PushBack(i)
		}
	}
	//  检查 nums 是否是唯一的最短 超序列  表示 每次入队的num只有一个
	for queue.Len() > 0 {
		if queue.Len() > 1 {
			return false
		}
		front := queue.Front()
		queue.Remove(front)
		from := front.Value.(int)
		for _, to := range graph[from] {
			if inDegrees[to]--; inDegrees[to] == 0 {
				queue.PushBack(to)
			}
		}
	}
	return true
}

// 743. 网络延迟时间
// 有 n 个网络节点，标记为 1 到 n。
//
// 给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。
//
// 现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
//
// 示例 1：
// 输入：times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
// 输出：2
//
// 示例 2：
// 输入：times = [[1,2,1]], n = 2, k = 1
// 输出：1
//
// 示例 3：
// 输入：times = [[1,2,1]], n = 2, k = 2
// 输出：-1
//
// 提示：
// 1 <= k <= n <= 100
// 1 <= times.length <= 6000
// times[i].length == 3
// 1 <= ui, vi <= n
// ui != vi
// 0 <= wi <= 100
// 所有 (ui, vi) 对都 互不相同（即，不含重复边）
func networkDelayTime(times [][]int, n int, k int) int {
	graph := make([][]int, n+1)
	for i := 0; i <= n; i++ {
		graph[i] = make([]int, n+1)
		for j := 0; j <= n; j++ {
			graph[i][j] = -1
		}
	}
	for _, time := range times {
		graph[time[0]][time[1]] = time[2]
	}
	netTimes := make([]int, n+1)
	for i := 1; i <= n; i++ {
		netTimes[i] = math.MaxInt
	}
	netTimes[k] = 0
	// 广度优先遍历
	queue := list.New()
	queue.PushBack(k)
	visited := make([]bool, n+1)
	visited[k] = true
	for queue.Len() > 0 {
		front := queue.Front()
		queue.Remove(front)
		start := front.Value.(int)
		visited[start] = false
		for i := 1; i <= n; i++ {
			if graph[start][i] == -1 {
				continue
			}
			if netTimes[i] > netTimes[start]+graph[start][i] {
				netTimes[i] = netTimes[start] + graph[start][i]
				if !visited[i] {
					queue.PushBack(i)
					visited[i] = true
				}
			}
		}
	}
	result := 0
	for i := 1; i <= n; i++ {
		t := netTimes[i]
		if t == math.MaxInt {
			return -1
		}
		result = max(t, result)
	}
	return result
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

// 785. 判断二分图
// 存在一个 无向图 ，图中有 n 个节点。其中每个节点都有一个介于 0 到 n - 1 之间的唯一编号。给你一个二维数组 graph ，其中 graph[u] 是一个节点数组，由节点 u 的邻接节点组成。形式上，对于 graph[u] 中的每个 v ，都存在一条位于节点 u 和节点 v 之间的无向边。该无向图同时具有以下属性：
// 不存在自环（graph[u] 不包含 u）。
// 不存在平行边（graph[u] 不包含重复值）。
// 如果 v 在 graph[u] 内，那么 u 也应该在 graph[v] 内（该图是无向图）
// 这个图可能不是连通图，也就是说两个节点 u 和 v 之间可能不存在一条连通彼此的路径。
// 二分图 定义：如果能将一个图的节点集合分割成两个独立的子集 A 和 B ，并使图中的每一条边的两个节点一个来自 A 集合，一个来自 B 集合，就将这个图称为 二分图 。
//
// 如果图是二分图，返回 true ；否则，返回 false 。
//
// 示例 1：
// 输入：graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
// 输出：false
// 解释：不能将节点分割成两个独立的子集，以使每条边都连通一个子集中的一个节点与另一个子集中的一个节点。
//
// 示例 2：
// 输入：graph = [[1,3],[0,2],[1,3],[0,2]]
// 输出：true
// 解释：可以将节点分成两组: {0, 2} 和 {1, 3} 。
//
// 提示：
// graph.length == n
// 1 <= n <= 100
// 0 <= graph[u].length < n
// 0 <= graph[u][i] <= n - 1
// graph[u] 不会包含 u
// graph[u] 的所有值 互不相同
// 如果 graph[u] 包含 v，那么 graph[v] 也会包含 u
func isBipartite(graph [][]int) bool {
	n := len(graph)
	// 广度优先遍历 染色
	colors := make([]int, 101)
	for i := 0; i < n; i++ {
		if colors[i] != 0 {
			continue
		}
		queue := list.New()
		queue.PushBack(i)
		colors[i] = 1
		for queue.Len() > 0 {
			front := queue.Front()
			queue.Remove(front)
			node := front.Value.(int)

			var nextColor int
			if colors[node] == 1 {
				nextColor = 2
			} else {
				nextColor = 1
			}
			for _, next := range graph[node] {
				if colors[next] == 0 {
					colors[next] = nextColor
					queue.PushBack(next)
				} else if colors[next] != nextColor {
					return false
				}
			}
		}
	}
	return true
}

// 787. K 站中转内最便宜的航班
// 有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [fromi, toi, pricei] ，表示该航班都从城市 fromi 开始，以价格 pricei 抵达 toi。
//
// 现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。
//
// 示例 1：
// 输入:
// n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
// src = 0, dst = 2, k = 1
// 输出: 200
// 解释:
// 城市航班图如下
//
// 从城市 0 到城市 2 在 1 站中转以内的最便宜价格是 200，如图中红色所示。
//
// 示例 2：
// 输入:
// n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
// src = 0, dst = 2, k = 0
// 输出: 500
// 解释:
// 城市航班图如下
//
// 从城市 0 到城市 2 在 0 站中转以内的最便宜价格是 500，如图中蓝色所示。
//
// 提示：
// 1 <= n <= 100
// 0 <= flights.length <= (n * (n - 1) / 2)
// flights[i].length == 3
// 0 <= fromi, toi < n
// fromi != toi
// 1 <= pricei <= 104
// 航班没有重复，且不存在自环
// 0 <= src, dst, k < n
// src != dst
func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
	const inf = 10000*101 + 1
	result := inf
	// cost[k][dst] k站 目的地dst 的最小花费
	cost := make([][]int, k+2)
	for i := range cost {
		cost[i] = make([]int, n)
		for j := range cost[i] {
			cost[i][j] = inf
		}
	}
	cost[0][src] = 0
	for i := 1; i <= k+1; i++ {
		for _, flight := range flights {
			start, end, price := flight[0], flight[1], flight[2]
			// 目的地
			cost[i][end] = min(cost[i][end], price+cost[i-1][start])
		}
	}
	for i := 1; i <= k+1; i++ {
		result = min(result, cost[i][dst])
	}
	if result == inf {
		return -1
	}

	return result
}

type FlightInfo struct {
	start int
	price int
}

// 1971. 寻找图中是否存在路径
// 有一个具有 n 个顶点的 双向 图，其中每个顶点标记从 0 到 n - 1（包含 0 和 n - 1）。图中的边用一个二维整数数组 edges 表示，其中 edges[i] = [ui, vi] 表示顶点 ui 和顶点 vi 之间的双向边。 每个顶点对由 最多一条 边连接，并且没有顶点存在与自身相连的边。
//
// 请你确定是否存在从顶点 source 开始，到顶点 destination 结束的 有效路径 。
//
// 给你数组 edges 和整数 n、source 和 destination，如果从 source 到 destination 存在 有效路径 ，则返回 true，否则返回 false 。
//
// 示例 1：
// 输入：n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
// 输出：true
// 解释：存在由顶点 0 到顶点 2 的路径:
// - 0 → 1 → 2
// - 0 → 2
//
// 示例 2：
// 输入：n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5
// 输出：false
// 解释：不存在由顶点 0 到顶点 5 的路径.
//
// 提示：
// 1 <= n <= 2 * 105
// 0 <= edges.length <= 2 * 105
// edges[i].length == 2
// 0 <= ui, vi <= n - 1
// ui != vi
// 0 <= source, destination <= n - 1
// 不存在重复边
// 不存在指向顶点自身的边
func validPath(n int, edges [][]int, source int, destination int) bool {
	edgeList := make([][]int, n)
	for i := 0; i < n; i++ {
		edgeList[i] = make([]int, 0)
	}
	for _, edge := range edges {
		x, y := edge[0], edge[1]
		edgeList[x] = append(edgeList[x], y)
		edgeList[y] = append(edgeList[y], x)
	}
	visited := make([]bool, n)

	var dfs func(cur int) bool
	dfs = func(cur int) bool {
		if cur == destination {
			return true
		}
		visited[cur] = true
		for _, next := range edgeList[cur] {
			if !visited[next] && dfs(next) {
				return true
			}
		}
		return false
	}

	return dfs(source)
}

// 997. 找到小镇的法官
// 小镇里有 n 个人，按从 1 到 n 的顺序编号。传言称，这些人中有一个暗地里是小镇法官。
//
// 如果小镇法官真的存在，那么：
// 小镇法官不会信任任何人。
// 每个人（除了小镇法官）都信任这位小镇法官。
// 只有一个人同时满足属性 1 和属性 2 。
// 给你一个数组 trust ，其中 trust[i] = [ai, bi] 表示编号为 ai 的人信任编号为 bi 的人。
//
// 如果小镇法官存在并且可以确定他的身份，请返回该法官的编号；否则，返回 -1 。
//
// 示例 1：
// 输入：n = 2, trust = [[1,2]]
// 输出：2
//
// 示例 2：
// 输入：n = 3, trust = [[1,3],[2,3]]
// 输出：3
//
// 示例 3：
// 输入：n = 3, trust = [[1,3],[2,3],[3,1]]
// 输出：-1
//
// 提示：
// 1 <= n <= 1000
// 0 <= trust.length <= 104
// trust[i].length == 2
// trust 中的所有trust[i] = [ai, bi] 互不相同
// ai != bi
// 1 <= ai, bi <= n
func findJudge(n int, trust [][]int) int {
	inDegrees, outDegrees := make([]int, n+1), make([]int, n+1)
	for _, t := range trust {
		a, b := t[0], t[1]
		outDegrees[a]++
		inDegrees[b]++
	}
	for i := 1; i <= n; i++ {
		if outDegrees[i] == 0 && inDegrees[i] == n-1 {
			return i
		}
	}
	return -1
}
