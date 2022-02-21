package random

import "math/rand"

// BlackRandom 710. 黑名单中的随机数
// 给定一个整数 n 和一个 无重复 黑名单整数数组 blacklist 。设计一种算法，从 [0, n - 1] 范围内的任意整数中选取一个 未加入 黑名单 blacklist 的整数。任何在上述范围内且不在黑名单 blacklist 中的整数都应该有 同等的可能性 被返回。
//
// 优化你的算法，使它最小化调用语言 内置 随机函数的次数。
// 实现 Solution 类:
// Solution(int n, int[] blacklist) 初始化整数 n 和被加入黑名单 blacklist 的整数
// int pick() 返回一个范围为 [0, n - 1] 且不在黑名单 blacklist 中的随机整数
//
// 示例 1：
// 输入
// ["Solution", "pick", "pick", "pick", "pick", "pick", "pick", "pick"]
// [[7, [2, 3, 5]], [], [], [], [], [], [], []]
// 输出
// [null, 0, 4, 1, 6, 1, 0, 4]
//
// 解释
// Solution solution = new Solution(7, [2, 3, 5]);
// solution.pick(); // 返回0，任何[0,1,4,6]的整数都可以。注意，对于每一个pick的调用，
//                  // 0、1、4和6的返回概率必须相等(即概率为1/4)。
// solution.pick(); // 返回 4
// solution.pick(); // 返回 1
// solution.pick(); // 返回 6
// solution.pick(); // 返回 1
// solution.pick(); // 返回 0
// solution.pick(); // 返回 4
//
// 提示:
// 1 <= n <= 109
// 0 <= blacklist.length <- min(105, n - 1)
// 0 <= blacklist[i] < n
// blacklist 中所有值都 不同
// pick 最多被调用 2 * 104 次
type BlackRandom struct {
	bound    int
	blackMap map[int]int
}

// ConstructorBlack 分析
//
// 白名单中数的个数为 N - len(B)，那么可以直接在 [0, N - len(B)) 中随机生成整数。我们把所有小于 N - len(B)
// 且在黑名单中数一一映射到大于等于 N -len(B) 且出现在白名单中的数
// 。这样一来，如果随机生成的整数出现在黑名单中，我们就返回它唯一对应的那个出现在白名单中的数即可。
//
// 例如当 N = 6，B = [0, 2, 3] 时，我们在 [0, 3) 中随机生成整数，并将 2 映射到 4，3 映射到 5，这样随机生成的整数就是 [0, 1, 4, 5]
// 中的一个。
//
// 算法
//
// 我们将黑名单分成两部分，第一部分 X 的数都小于 N - len(B)，需要进行映射；
//
// 第二部分 Y 的数都大于等于 N - len(B)，这些数不需要进行映射，因为并不会随机到它们。
//
func ConstructorBlack(n int, blacklist []int) BlackRandom {
	bound := n - len(blacklist)
	blackMap := make(map[int]int)
	// 黑名单中 < bound 的 映射 成  > bound 的白名单
	set := make(map[int]bool)
	for i := bound; i < n; i++ {
		set[i] = true
	}
	for _, num := range blacklist {
		if set[num] == true {
			set[num] = false
		}
	}
	blacknums := make([]int, 0)
	for num, v := range set {
		if v {
			blacknums = append(blacknums, num)
		}

	}
	idx := 0
	for _, num := range blacklist {
		if num < bound {
			blackMap[num] = blacknums[idx]
			idx++
		}
	}

	return BlackRandom{bound: bound, blackMap: blackMap}
}

func (this *BlackRandom) Pick() int {
	idx := rand.Intn(this.bound)
	if v, ok := this.blackMap[idx]; ok {
		return v
	}
	return idx
}
