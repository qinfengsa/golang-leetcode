package matrix

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
