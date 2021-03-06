package matrix

import (
	"container/heap"
	"container/list"
	"math"
	"sort"
)

var (
	DirCol       = []int{1, -1, 0, 0}
	DirRow       = []int{0, 0, 1, -1}
	AroundDirRow = []int{-1, -1, -1, 0, 0, 1, 1, 1}
	AroundDirCol = []int{-1, 0, 1, -1, 1, -1, 0, 1}
)

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func inArea(row, col, rows, cols int) bool {
	return row >= 0 && row < rows && col >= 0 && col < cols
}

// 329. 矩阵中的最长递增路径
// 给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。
//
// 对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。
//
// 示例 1：
// 输入：matrix = [[9,9,4],[6,6,8],[2,1,1]]
// 输出：4
// 解释：最长递增路径为 [1, 2, 6, 9]。
//
// 示例 2：
// 输入：matrix = [[3,4,5],[3,2,6],[2,2,1]]
// 输出：4
// 解释：最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
//
// 示例 3：
// 输入：matrix = [[1]]
// 输出：1
//
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 200
// 0 <= matrix[i][j] <= 231 - 1
func longestIncreasingPath(matrix [][]int) int {
	m, n := len(matrix), len(matrix[0])

	visited := make([][]int, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]int, n)
	}

	var dfs func(row, col int) int

	dfs = func(row, col int) int {
		if !inArea(row, col, m, n) {
			return 0
		}
		if visited[row][col] != 0 {
			return visited[row][col]
		}
		result := 0
		for k := 0; k < 4; k++ {
			nextRow, nextCol := row+DirRow[k], col+DirCol[k]
			if !inArea(nextRow, nextCol, m, n) {
				continue
			}
			if matrix[nextRow][nextCol] > matrix[row][col] {
				result = max(result, dfs(nextRow, nextCol))
			}
		}
		visited[row][col] = 1 + result
		return visited[row][col]
	}

	result := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result = max(result, dfs(i, j))
		}
	}
	return result
}

// 363. 矩形区域不超过 K 的最大数值和
// 给你一个 m x n 的矩阵 matrix 和一个整数 k ，找出并返回矩阵内部矩形区域的不超过 k 的最大数值和。
//
// 题目数据保证总会存在一个数值和不超过 k 的矩形区域。
//
// 示例 1：
// 输入：matrix = [[1,0,1],[0,-2,3]], k = 2
// 输出：2
// 解释：蓝色边框圈出来的矩形区域 [[0, 1], [-2, 3]] 的数值和是 2，且 2 是不超过 k 的最大数字（k = 2）。
//
// 示例 2：
// 输入：matrix = [[2,2,-1]], k = 3
// 输出：3
// 提示：
// m == matrix.length
// n == matrix[i].length
// 1 <= m, n <= 100
// -100 <= matrix[i][j] <= 100
// -105 <= k <= 105
func maxSumSubmatrix(matrix [][]int, k int) int {
	m, n := len(matrix), len(matrix[0])
	// 前缀和
	for i := 0; i < m; i++ {
		for j := 1; j < n; j++ {
			matrix[i][j] += matrix[i][j-1]
		}
	}
	result := math.MinInt32
	for left := 0; left < n; left++ {
		for right := left; right < n; right++ {
			// 固定 left ~ right 的列

			for top := 0; top < m; top++ {
				sum := 0
				for bottom := top; bottom < m; bottom++ {
					sum += matrix[bottom][right]
					if left > 0 {
						sum -= matrix[bottom][left-1]
					}
					if sum <= k && sum > result {
						result = sum
					}
				}
			}
		}
	}

	return result
}

// 378. 有序矩阵中第 K 小的元素
// 给你一个 n x n 矩阵 matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
// 请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。
//
// 示例 1：
// 输入：matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
// 输出：13
// 解释：矩阵中的元素为 [1,5,9,10,11,12,13,13,15]，第 8 小元素是 13
//
// 示例 2：
// 输入：matrix = [[-5]], k = 1 输出：-5
//
// 提示：
// n == matrix.length
// n == matrix[i].length
// 1 <= n <= 300
// -109 <= matrix[i][j] <= 109
// 题目数据 保证 matrix 中的所有行和列都按 非递减顺序 排列
// 1 <= k <= n2
func kthSmallest(matrix [][]int, k int) int {
	n := len(matrix)

	low, high := matrix[0][0], matrix[n-1][n-1]

	// 二分计算 所有小于等于 target 的 元素数量
	countLess := func(target int) int {
		count := 0
		// 从 左上角开始
		i, j := n-1, 0
		for i >= 0 && j < n {
			if matrix[i][j] <= target {
				count += i + 1
				j++
			} else {
				i--
			}
		}
		return count
	}

	for low < high {
		mid := (low + high) >> 1
		count := countLess(mid)
		if count < k {
			low = mid + 1
		} else {
			high = mid
		}
	}

	return low
}

type TrapNode struct {
	row, col, height int
}

type hp []*TrapNode

func (h hp) Len() int {
	return len(h)
}

func (h hp) Less(i, j int) bool {
	return h[i].height < h[j].height
}

func (h hp) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *hp) Push(x interface{}) {
	*h = append(*h, x.(*TrapNode))
}

func (h *hp) Pop() interface{} {
	tmp := *h
	v := tmp[len(tmp)-1]
	*h = tmp[:len(tmp)-1]
	return v
}

// 407. 接雨水 II
// 给你一个 m x n 的矩阵，其中的值均为非负整数，代表二维高度图每个单元的高度，请计算图中形状最多能接多少体积的雨水。
//
// 示例 1:
// 输入: heightMap = [[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]
// 输出: 4
// 解释: 下雨后，雨水将会被上图蓝色的方块中。总的接雨水量为1+2+1=4。
//
// 示例 2:
// 输入: heightMap = [[3,3,3,3,3],[3,2,2,2,3],[3,2,1,2,3],[3,2,2,2,3],[3,3,3,3,3]]
// 输出: 10
//
// 提示:
// m == heightMap.length
// n == heightMap[i].length
// 1 <= m, n <= 200
// 0 <= heightMap[i][j] <= 2 * 104
func trapRainWater(heightMap [][]int) int {
	m, n := len(heightMap), len(heightMap[0])
	if m < 3 || n < 3 {
		return 0
	}
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	visited[0][0], visited[0][n-1], visited[m-1][0], visited[m-1][n-1] = true, true, true, true
	h := hp{}
	for i := 1; i < m-1; i++ {
		heap.Push(&h, &TrapNode{row: i, col: 0, height: heightMap[i][0]})
		heap.Push(&h, &TrapNode{row: i, col: n - 1, height: heightMap[i][n-1]})
		visited[i][0] = true
		visited[i][n-1] = true
	}
	for j := 1; j < n-1; j++ {
		heap.Push(&h, &TrapNode{row: 0, col: j, height: heightMap[0][j]})
		heap.Push(&h, &TrapNode{row: m - 1, col: j, height: heightMap[m-1][j]})
		visited[0][j] = true
		visited[m-1][j] = true
	}

	result := 0
	for h.Len() > 0 {
		trap := heap.Pop(&h).(*TrapNode)
		for k := 0; k < 4; k++ {
			nextRow, nextCol := trap.row+DirRow[k], trap.col+DirCol[k]
			if !inArea(nextRow, nextCol, m, n) || visited[nextRow][nextCol] {
				continue
			}
			if trap.height > heightMap[nextRow][nextCol] {
				result += trap.height - heightMap[nextRow][nextCol]
			}
			heap.Push(&h, &TrapNode{row: nextRow, col: nextCol, height: max(trap.height, heightMap[nextRow][nextCol])})
			visited[nextRow][nextCol] = true
		}
	}

	return result
}

// 417. 太平洋大西洋水流问题
// 给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。
// 规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。
// 请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。
//
// 提示：
// 输出坐标的顺序不重要
// m 和 n 都小于150
//
// 示例：
// 给定下面的 5x5 矩阵:
//
//  太平洋 ~   ~   ~   ~   ~
//       ~  1   2   2   3  (5) *
//       ~  3   2   3  (4) (4) *
//       ~  2   4  (5)  3   1  *
//       ~ (6) (7)  1   4   5  *
//       ~ (5)  1   1   2   4  *
//          *   *   *   *   * 大西洋
// 返回:
// [[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
func pacificAtlantic(heights [][]int) [][]int {
	result := make([][]int, 0)
	m, n := len(heights), len(heights[0])

	var find func(canReach *[][]bool, row, col int)

	find = func(canReach *[][]bool, row, col int) {
		height := heights[row][col]

		matrix := *canReach
		if matrix[row][col] {
			return
		}
		matrix[row][col] = true
		for k := 0; k < 4; k++ {
			nextRow, nextCol := row+DirRow[k], col+DirCol[k]
			if inArea(nextRow, nextCol, m, n) && heights[nextRow][nextCol] >= height {
				find(canReach, nextRow, nextCol)
			}
		}
	}
	// 第一行 第一列
	canReach1 := make([][]bool, m)
	for i := 0; i < m; i++ {
		canReach1[i] = make([]bool, n)
	}
	for j := 0; j < n; j++ {
		find(&canReach1, 0, j)
	}
	for i := 1; i < m; i++ {
		find(&canReach1, i, 0)
	}

	// 第m-1行 第m-1列
	canReach2 := make([][]bool, m)
	for i := 0; i < m; i++ {
		canReach2[i] = make([]bool, n)
	}

	for j := 0; j < n; j++ {
		find(&canReach2, m-1, j)
	}
	for i := 0; i < m-1; i++ {
		find(&canReach2, i, n-1)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if canReach1[i][j] && canReach2[i][j] {
				result = append(result, []int{i, j})
			}
		}
	}

	return result
}

// 419. 甲板上的战舰
// 给定一个二维的甲板， 请计算其中有多少艘战舰。 战舰用 'X'表示，空位用 '.'表示。 你需要遵守以下规则：
//
// 给你一个有效的甲板，仅由战舰或者空位组成。
// 战舰只能水平或者垂直放置。换句话说,战舰只能由 1xN (1 行, N 列)组成，或者 Nx1 (N 行, 1 列)组成，其中N可以是任意大小。
// 两艘战舰之间至少有一个水平或垂直的空位分隔 - 即没有相邻的战舰。
//
// 示例 :
// X..X
// ...X
// ...X
// 在上面的甲板中有2艘战舰。
//
// 无效样例 :
// ...X
// XXXX
// ...X
// 你不会收到这样的无效甲板 - 因为战舰之间至少会有一个空位将它们分开。
//
// 进阶:
// 你可以用一次扫描算法，只使用O(1)额外空间，并且不修改甲板的值来解决这个问题吗？
func countBattleships(board [][]byte) int {
	m, n := len(board), len(board[0])
	result := 0

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if board[i][j] != 'X' {
				continue
			}
			// 只看 左边 和 上边
			if (i > 0 && board[i-1][j] == 'X') || (j > 0 && board[i][j-1] == 'X') {
				continue
			}
			result++
		}
	}
	return result
}

// 498. 对角线遍历
// 给你一个大小为 m x n 的矩阵 mat ，请以对角线遍历的顺序，用一个数组返回这个矩阵中的所有元素。
//
// 示例 1：
// 输入：mat = [[1,2,3],[4,5,6],[7,8,9]]
// 输出：[1,2,4,7,5,3,6,8,9]
//
// 示例 2：
// 输入：mat = [[1,2],[3,4]]
// 输出：[1,2,3,4]
//
// 提示：
// m == mat.length
// n == mat[i].length
// 1 <= m, n <= 104
// 1 <= m * n <= 104
// -105 <= mat[i][j] <= 105
func findDiagonalOrder(mat [][]int) []int {
	m, n := len(mat), len(mat[0])
	result := make([]int, m*n)
	if m == 1 {
		return mat[0]
	}
	index := 0
	for k := 0; k < m+n-1; k++ {
		if k&1 == 0 { // 向上
			for i := min(k, m-1); i >= 0; i-- {
				j := k - i
				if j < 0 || j >= n {
					break
				}
				result[index] = mat[i][j]
				index++
			}
		} else { // 向下
			for j := min(k, n-1); j >= 0; j-- {
				i := k - j
				if i < 0 || i >= m {
					break
				}
				result[index] = mat[i][j]
				index++
			}
		}
	}
	return result
}

// 529. 扫雷游戏
// 让我们一起来玩扫雷游戏！
//
// 给你一个大小为 m x n 二维字符矩阵 board ，表示扫雷游戏的盘面，其中：
//
// 'M' 代表一个 未挖出的 地雷，
// 'E' 代表一个 未挖出的 空方块，
// 'B' 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的 已挖出的 空白方块，
// 数字（'1' 到 '8'）表示有多少地雷与这块 已挖出的 方块相邻，
// 'X' 则表示一个 已挖出的 地雷。
// 给你一个整数数组 click ，其中 click = [clickr, clickc] 表示在所有 未挖出的 方块（'M' 或者 'E'）中的下一个点击位置（clickr 是行下标，clickc 是列下标）。
//
// 根据以下规则，返回相应位置被点击后对应的盘面：
//
// 如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 'X' 。
// 如果一个 没有相邻地雷 的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的 未挖出 方块都应该被递归地揭露。
// 如果一个 至少与一个地雷相邻 的空方块（'E'）被挖出，修改它为数字（'1' 到 '8' ），表示相邻地雷的数量。
// 如果在此次点击中，若无更多方块可被揭露，则返回盘面。
//
// 示例 1：
// 输入：board = [["E","E","E","E","E"],["E","E","M","E","E"],["E","E","E","E","E"],["E","E","E","E","E"]], click = [3,0]
// 输出：[["B","1","E","1","B"],["B","1","M","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]]
//
// 示例 2：
// 输入：board = [["B","1","E","1","B"],["B","1","M","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]], click = [1,2]
// 输出：[["B","1","E","1","B"],["B","1","X","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]]
//
// 提示：
// m == board.length
// n == board[i].length
// 1 <= m, n <= 50
// board[i][j] 为 'M'、'E'、'B' 或数字 '1' 到 '8' 中的一个
// click.length == 2
// 0 <= clickr < m
// 0 <= clickc < n
// board[clickr][clickc] 为 'M' 或 'E'
func updateBoard(board [][]byte, click []int) [][]byte {
	m, n := len(board), len(board[0])

	countLandmine := func(row, col int) int {
		count := 0

		for i := 0; i < 8; i++ {
			nextRow, nextCol := row+AroundDirRow[i], col+AroundDirCol[i]
			if inArea(nextRow, nextCol, m, n) && board[nextRow][nextCol] == 'M' {
				count++
			}
		}

		return count
	}

	var dfs func(row, col int)

	dfs = func(row, col int) {
		if !inArea(row, col, m, n) {
			return
		}
		if board[row][col] == 'M' {
			board[row][col] = 'X'
		} else if board[row][col] == 'E' {
			count := countLandmine(row, col)
			if count == 0 {
				board[row][col] = 'B'
				for i := 0; i < 8; i++ {
					nextRow, nextCol := row+AroundDirRow[i], col+AroundDirCol[i]
					dfs(nextRow, nextCol)
				}
			} else {
				board[row][col] = byte('0' + count)
			}
		}
	}

	dfs(click[0], click[1])
	return board
}

// 542. 01 矩阵
// 给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
// 两个相邻元素间的距离为 1 。
//
// 示例 1：
// 输入：mat = [[0,0,0],[0,1,0],[0,0,0]]
// 输出：[[0,0,0],[0,1,0],[0,0,0]]
//
// 示例 2：
// 输入：mat = [[0,0,0],[0,1,0],[1,1,1]]
// 输出：[[0,0,0],[0,1,0],[1,2,1]]
//
// 提示：
// m == mat.length
// n == mat[i].length
// 1 <= m, n <= 104
// 1 <= m * n <= 104
// mat[i][j] is either 0 or 1.
// mat 中至少有一个 0
func updateMatrix(mat [][]int) [][]int {
	m, n := len(mat), len(mat[0])
	result := make([][]int, m)
	for i := 0; i < m; i++ {
		result[i] = make([]int, n)
		for j := 0; j < n; j++ {
			if mat[i][j] != 0 {
				result[i][j] = math.MaxInt32 >> 1
			}
		}
	}
	// 从左上到右下
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i > 0 {
				result[i][j] = min(result[i][j], result[i-1][j]+1)
			}
			if j > 0 {
				result[i][j] = min(result[i][j], result[i][j-1]+1)
			}
		}
	}

	// 从右下到左上
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			if i < m-1 {
				result[i][j] = min(result[i][j], result[i+1][j]+1)
			}
			if j < n-1 {
				result[i][j] = min(result[i][j], result[i][j+1]+1)
			}
		}
	}
	return result
}

// 2022. 将一维数组转变成二维数组
// 给你一个下标从 0 开始的一维整数数组 original 和两个整数 m 和  n 。你需要使用 original 中 所有 元素创建一个 m 行 n 列的二维数组。
//
// original 中下标从 0 到 n - 1 （都 包含 ）的元素构成二维数组的第一行，下标从 n 到 2 * n - 1 （都 包含 ）的元素构成二维数组的第二行，依此类推。
//
// 请你根据上述过程返回一个 m x n 的二维数组。如果无法构成这样的二维数组，请你返回一个空的二维数组。
//
// 示例 1：
// 输入：original = [1,2,3,4], m = 2, n = 2
// 输出：[[1,2],[3,4]]
// 解释：
// 构造出的二维数组应该包含 2 行 2 列。
// original 中第一个 n=2 的部分为 [1,2] ，构成二维数组的第一行。
// original 中第二个 n=2 的部分为 [3,4] ，构成二维数组的第二行。
//
// 示例 2：
// 输入：original = [1,2,3], m = 1, n = 3
// 输出：[[1,2,3]]
// 解释：
// 构造出的二维数组应该包含 1 行 3 列。
// 将 original 中所有三个元素放入第一行中，构成要求的二维数组。
//
// 示例 3：
// 输入：original = [1,2], m = 1, n = 1
// 输出：[]
// 解释：
// original 中有 2 个元素。
// 无法将 2 个元素放入到一个 1x1 的二维数组中，所以返回一个空的二维数组。
//
// 示例 4：
// 输入：original = [3], m = 1, n = 2
// 输出：[]
// 解释：
// original 中只有 1 个元素。
// 无法将 1 个元素放满一个 1x2 的二维数组，所以返回一个空的二维数组。
//
// 提示：
// 1 <= original.length <= 5 * 104
// 1 <= original[i] <= 105
// 1 <= m, n <= 4 * 104
func construct2DArray(original []int, m int, n int) [][]int {
	size := len(original)

	if size != m*n {
		return make([][]int, 0)
	}
	result := make([][]int, m)
	for i := 0; i < m; i++ {
		result[i] = make([]int, n)
	}

	for i := 0; i < size; i++ {
		num := original[i]
		row, col := i/n, i%n
		result[row][col] = num
	}
	return result
}

// 675. 为高尔夫比赛砍树
// 你被请来给一个要举办高尔夫比赛的树林砍树。树林由一个 m x n 的矩阵表示， 在这个矩阵中：
//
// 0 表示障碍，无法触碰
// 1 表示地面，可以行走
// 比 1 大的数 表示有树的单元格，可以行走，数值表示树的高度
// 每一步，你都可以向上、下、左、右四个方向之一移动一个单位，如果你站的地方有一棵树，那么你可以决定是否要砍倒它。
//
// 你需要按照树的高度从低向高砍掉所有的树，每砍过一颗树，该单元格的值变为 1（即变为地面）。
//
// 你将从 (0, 0) 点开始工作，返回你砍完所有树需要走的最小步数。 如果你无法砍完所有的树，返回 -1 。
//
// 可以保证的是，没有两棵树的高度是相同的，并且你至少需要砍倒一棵树。
//
// 示例 1：
// 输入：forest = [[1,2,3],[0,0,4],[7,6,5]]
// 输出：6
// 解释：沿着上面的路径，你可以用 6 步，按从最矮到最高的顺序砍掉这些树。
//
// 示例 2：
// 输入：forest = [[1,2,3],[0,0,0],[7,6,5]]
// 输出：-1
// 解释：由于中间一行被障碍阻塞，无法访问最下面一行中的树。
//
// 示例 3：
// 输入：forest = [[2,3,4],[0,0,5],[8,7,6]]
// 输出：6
// 解释：可以按与示例 1 相同的路径来砍掉所有的树。
// (0,0) 位置的树，可以直接砍去，不用算步数。
//
// 提示：
// m == forest.length
// n == forest[i].length
// 1 <= m, n <= 50
// 0 <= forest[i][j] <= 109
func cutOffTree(forest [][]int) int {
	m, n := len(forest), len(forest[0])

	nodeList := make([]*CutOffNode, 0)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if forest[i][j] > 1 {
				nodeList = append(nodeList, &CutOffNode{i, j, forest[i][j]})
			}
		}
	}
	sort.Slice(nodeList, func(i, j int) bool {
		return nodeList[i].height < nodeList[j].height
	})
	bfs := func(startRow, startCol, endRow, endCol int) int {
		step := 0
		queue := list.New()
		queue.PushBack([]int{startRow, startCol})
		visited := make([][]bool, m)
		for i := 0; i < m; i++ {
			visited[i] = make([]bool, n)
		}
		for queue.Len() > 0 {
			l := queue.Len()
			for i := 0; i < l; i++ {
				front := queue.Front()
				node := front.Value.([]int)
				queue.Remove(front)
				if node[0] == endRow && node[1] == endCol {
					return step
				}
				for k := 0; k < 4; k++ {
					row, col := node[0]+DirRow[k], node[1]+DirCol[k]
					if inArea(row, col, m, n) && !visited[row][col] && forest[row][col] > 0 {
						queue.PushBack([]int{row, col})
						visited[row][col] = true
					}
				}

			}
			step++
		}

		return -1
	}

	result := 0
	startRow, startCol := 0, 0
	for _, node := range nodeList {
		d := bfs(startRow, startCol, node.row, node.col)
		if d == -1 {
			return -1
		}
		result += d
		startRow = node.row
		startCol = node.col
	}

	return result
}

type CutOffNode struct {
	row, col, height int
}

// 1380. 矩阵中的幸运数
// 给你一个 m * n 的矩阵，矩阵中的数字 各不相同 。请你按 任意 顺序返回矩阵中的所有幸运数。
//
// 幸运数是指矩阵中满足同时下列两个条件的元素：
// 在同一行的所有元素中最小
// 在同一列的所有元素中最大
//
// 示例 1：
// 输入：matrix = [[3,7,8],[9,11,13],[15,16,17]]
// 输出：[15]
// 解释：15 是唯一的幸运数，因为它是其所在行中的最小值，也是所在列中的最大值。
//
// 示例 2：
// 输入：matrix = [[1,10,4,2],[9,3,8,7],[15,16,17,12]]
// 输出：[12]
// 解释：12 是唯一的幸运数，因为它是其所在行中的最小值，也是所在列中的最大值。
//
// 示例 3：
// 输入：matrix = [[7,8],[1,2]]
// 输出：[7]
//
// 提示：
// m == mat.length
// n == mat[i].length
// 1 <= n, m <= 50
// 1 <= matrix[i][j] <= 10^5
// 矩阵中的所有元素都是不同的
func luckyNumbers(matrix [][]int) []int {
	result := make([]int, 0)
	m, n := len(matrix), len(matrix[0])

	for i := 0; i < m; i++ {
		// 当前行最小
		minVal := matrix[i][0]
		minIdx := 0
		for j := 1; j < n; j++ {
			if matrix[i][j] < minVal {
				minVal = matrix[i][j]
				minIdx = j
			}
		}
		// 判断在同一列的所有元素中是否最大
		flag := true
		for k := 0; k < m; k++ {
			if matrix[k][minIdx] > minVal {
				flag = false
				break
			}
		}
		if flag {
			result = append(result, minVal)
		}

	}

	return result
}

// 695. 岛屿的最大面积
// 给你一个大小为 m x n 的二进制矩阵 grid 。
//
// 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。
//
// 岛屿的面积是岛上值为 1 的单元格的数目。
//
// 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
//
// 示例 1：
// 输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
// 输出：6
// 解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
//
// 示例 2：
// 输入：grid = [[0,0,0,0,0,0,0,0]]
// 输出：0
//
// 提示：
// m == grid.length
// n == grid[i].length
// 1 <= m, n <= 50
// grid[i][j] 为 0 或 1
func maxAreaOfIsland(grid [][]int) int {
	m, n := len(grid), len(grid[0])

	var getArea func(i, j int) int

	getArea = func(i, j int) int {
		if grid[i][j] != 1 {
			return 0
		}
		grid[i][j] = 2
		area := 1
		for k := 0; k < 4; k++ {
			row, col := i+DirRow[k], j+DirCol[k]
			if inArea(row, col, m, n) {
				area += getArea(row, col)
			}
		}
		return area
	}

	maxArea := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] != 1 {
				continue
			}
			maxArea = max(maxArea, getArea(i, j))
		}
	}
	return maxArea
}

// 1672. 最富有客户的资产总量
// 给你一个 m x n 的整数网格 accounts ，其中 accounts[i][j] 是第 i​​​​​​​​​​​​ 位客户在第 j 家银行托管的资产数量。返回最富有客户所拥有的 资产总量 。
//
// 客户的 资产总量 就是他们在各家银行托管的资产数量之和。最富有客户就是 资产总量 最大的客户。
//
// 示例 1：
// 输入：accounts = [[1,2,3],[3,2,1]]
// 输出：6
// 解释：
// 第 1 位客户的资产总量 = 1 + 2 + 3 = 6
// 第 2 位客户的资产总量 = 3 + 2 + 1 = 6
// 两位客户都是最富有的，资产总量都是 6 ，所以返回 6 。
//
// 示例 2：
// 输入：accounts = [[1,5],[7,3],[3,5]]
// 输出：10
// 解释：
// 第 1 位客户的资产总量 = 6
// 第 2 位客户的资产总量 = 10
// 第 3 位客户的资产总量 = 8
// 第 2 位客户是最富有的，资产总量是 10
//
// 示例 3：
// 输入：accounts = [[2,8,7],[7,1,3],[1,9,5]]
// 输出：17
//
// 提示：
// m == accounts.length
// n == accounts[i].length
// 1 <= m, n <= 50
// 1 <= accounts[i][j] <= 100
func maximumWealth(accounts [][]int) int {
	m, n := len(accounts), len(accounts[0])
	result := 0
	for i := 0; i < m; i++ {
		sum := 0
		for j := 0; j < n; j++ {
			sum += accounts[i][j]
		}
		result = max(result, sum)
	}
	return result
}

// 733. 图像渲染
// 有一幅以 m x n 的二维整数数组表示的图画 image ，其中 image[i][j] 表示该图画的像素值大小。
//
// 你也被给予三个整数 sr ,  sc 和 newColor 。你应该从像素 image[sr][sc] 开始对图像进行 上色填充 。
//
// 为了完成 上色工作 ，从初始像素开始，记录初始坐标的 上下左右四个方向上 像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应 四个方向上 像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为 newColor 。
//
// 最后返回 经过上色渲染后的图像 。
//
// 示例 1:
// 输入: image = [[1,1,1],[1,1,0],[1,0,1]]，sr = 1, sc = 1, newColor = 2
// 输出: [[2,2,2],[2,2,0],[2,0,1]]
// 解析: 在图像的正中间，(坐标(sr,sc)=(1,1)),在路径上所有符合条件的像素点的颜色都被更改成2。
// 注意，右下角的像素没有更改为2，因为它不是在上下左右四个方向上与初始点相连的像素点。
//
// 示例 2:
// 输入: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, newColor = 2
// 输出: [[2,2,2],[2,2,2]]
//
// 提示:
// m == image.length
// n == image[i].length
// 1 <= m, n <= 50
// 0 <= image[i][j], newColor < 216
// 0 <= sr < m
// 0 <= sc < n
func floodFill(image [][]int, sr int, sc int, newColor int) [][]int {
	m, n := len(image), len(image[0])
	oldColor := image[sr][sc]
	var update func(i, j int)
	update = func(i, j int) {
		if !inArea(i, j, m, n) {
			return
		}
		if image[i][j] != oldColor {
			return
		}
		image[i][j] = newColor
		for k := 0; k < 4; k++ {
			update(i+DirRow[k], j+DirCol[k])
		}
	}
	if oldColor != newColor {
		update(sr, sc)
	}
	return image
}

// 1252. 奇数值单元格的数目
// 给你一个 m x n 的矩阵，最开始的时候，每个单元格中的值都是 0。
//
// 另有一个二维索引数组 indices，indices[i] = [ri, ci] 指向矩阵中的某个位置，其中 ri 和 ci 分别表示指定的行和列（从 0 开始编号）。
//
// 对 indices[i] 所指向的每个位置，应同时执行下述增量操作：
//
// ri 行上的所有单元格，加 1 。
// ci 列上的所有单元格，加 1 。
// 给你 m、n 和 indices 。请你在执行完所有 indices 指定的增量操作后，返回矩阵中 奇数值单元格 的数目。
//
// 示例 1：
// 输入：m = 2, n = 3, indices = [[0,1],[1,1]]
// 输出：6
// 解释：最开始的矩阵是 [[0,0,0],[0,0,0]]。
// 第一次增量操作后得到 [[1,2,1],[0,1,0]]。
// 最后的矩阵是 [[1,3,1],[1,3,1]]，里面有 6 个奇数。
//
// 示例 2：
// 输入：m = 2, n = 2, indices = [[1,1],[0,0]]
// 输出：0
// 解释：最后的矩阵是 [[2,2],[2,2]]，里面没有奇数。
//
// 提示：
// 1 <= m, n <= 50
// 1 <= indices.length <= 100
// 0 <= ri < m
// 0 <= ci < n
//
// 进阶：你可以设计一个时间复杂度为 O(n + m + indices.length) 且仅用 O(n + m) 额外空间的算法来解决此问题吗？
func oddCells(m int, n int, indices [][]int) int {
	rows, cols := make([]int, m), make([]int, n)
	for _, indice := range indices {
		rows[indice[0]]++
		cols[indice[1]]++
	}
	//
	oddRow, evenRow, oddCol, evenCol := 0, 0, 0, 0
	for _, num := range rows {
		if num&1 == 1 {
			oddRow++
		} else {
			evenRow++
		}
	}
	for _, num := range cols {
		if num&1 == 1 {
			oddCol++
		} else {
			evenCol++
		}
	}
	return oddRow*evenCol + evenRow*oddCol
}

// 1260. 二维网格迁移
// 给你一个 m 行 n 列的二维网格 grid 和一个整数 k。你需要将 grid 迁移 k 次。
//
// 每次「迁移」操作将会引发下述活动：
//
// 位于 grid[i][j] 的元素将会移动到 grid[i][j + 1]。
// 位于 grid[i][n - 1] 的元素将会移动到 grid[i + 1][0]。
// 位于 grid[m - 1][n - 1] 的元素将会移动到 grid[0][0]。
// 请你返回 k 次迁移操作后最终得到的 二维网格。
//
// 示例 1：
// 输入：grid = [[1,2,3],[4,5,6],[7,8,9]], k = 1
// 输出：[[9,1,2],[3,4,5],[6,7,8]]
//
// 示例 2：
// 输入：grid = [[3,8,1,9],[19,7,2,5],[4,6,11,10],[12,0,21,13]], k = 4
// 输出：[[12,0,21,13],[3,8,1,9],[19,7,2,5],[4,6,11,10]]
//
// 示例 3：
// 输入：grid = [[1,2,3],[4,5,6],[7,8,9]], k = 9
// 输出：[[1,2,3],[4,5,6],[7,8,9]]
//
// 提示：
// m == grid.length
// n == grid[i].length
// 1 <= m <= 50
// 1 <= n <= 50
// -1000 <= grid[i][j] <= 1000
// 0 <= k <= 100
func shiftGrid(grid [][]int, k int) [][]int {
	m, n := len(grid), len(grid[0])
	l := m * n
	k %= l
	result := make([][]int, m)
	for i := 0; i < m; i++ {
		result[i] = make([]int, n)
	}
	row, col := k/n, k%n
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result[row][col] = grid[i][j]
			col++
			if col == n {
				col = 0
				row++
			}
			if row == m {
				row = 0
			}
		}
	}

	return result
}
