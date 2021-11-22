package random

import "math/rand"

// Matrix 519. 随机翻转矩阵
// 给你一个 m x n 的二元矩阵 matrix ，且所有值被初始化为 0 。请你设计一个算法，随机选取一个满足 matrix[i][j] == 0 的下标 (i, j) ，并将它的值变为 1 。所有满足 matrix[i][j] == 0 的下标 (i, j) 被选取的概率应当均等。
// 尽量最少调用内置的随机函数，并且优化时间和空间复杂度。
// 实现 Solution 类：
// Solution(int m, int n) 使用二元矩阵的大小 m 和 n 初始化该对象
// int[] flip() 返回一个满足 matrix[i][j] == 0 的随机下标 [i, j] ，并将其对应格子中的值变为 1
// void reset() 将矩阵中所有的值重置为 0
//
// 示例：
// 输入
// ["Solution", "flip", "flip", "flip", "reset", "flip"]
// [[3, 1], [], [], [], [], []]
// 输出
// [null, [1, 0], [2, 0], [0, 0], null, [2, 0]]
//
// 解释
// Solution solution = new Solution(3, 1);
// solution.flip();  // 返回 [1, 0]，此时返回 [0,0]、[1,0] 和 [2,0] 的概率应当相同
// solution.flip();  // 返回 [2, 0]，因为 [1,0] 已经返回过了，此时返回 [2,0] 和 [0,0] 的概率应当相同
// solution.flip();  // 返回 [0, 0]，根据前面已经返回过的下标，此时只能返回 [0,0]
// solution.reset(); // 所有值都重置为 0 ，并可以再次选择下标返回
// solution.flip();  // 返回 [2, 0]，此时返回 [0,0]、[1,0] 和 [2,0] 的概率应当相同
//
// 提示：
// 1 <= m, n <= 104
// 每次调用flip 时，矩阵中至少存在一个值为 0 的格子。
// 最多调用 1000 次 flip 和 reset 方法。
type Matrix struct {
	m, n   int
	size   int
	matMap map[int]int
}

func Constructor4(m int, n int) Matrix {
	return Matrix{m: m, n: n, size: m * n, matMap: make(map[int]int)}
}

func (this *Matrix) Flip() []int {
	if this.size <= 0 {
		return nil
	}
	// 随机下标
	index := rand.Intn(this.size)
	this.size--
	num := index
	if v, ok := this.matMap[index]; ok {
		num = v
	}
	// 原下标与尾部下标交换
	last := this.size
	if v, ok := this.matMap[last]; ok {
		last = v
	}
	this.matMap[index] = last
	return []int{num / this.n, num % this.n}
}

func (this *Matrix) Reset() {
	this.matMap = make(map[int]int)
	this.size = this.m * this.n
}
