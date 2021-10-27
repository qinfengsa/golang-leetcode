package matrix

import (
	"container/heap"
	"math"
)

var (
	DirCol = []int{1, -1, 0, 0}
	DirRow = []int{0, 0, 1, -1}
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
