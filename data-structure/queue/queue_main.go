package queue

import "container/list"

// 752. 打开转盘锁
// 你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
//
// 锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
//
// 列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
//
// 字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。
//
// 示例 1:
// 输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
// 输出：6
// 解释：
// 可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
// 注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
// 因为当拨动到 "0102" 时这个锁就会被锁定。
//
// 示例 2:
// 输入: deadends = ["8888"], target = "0009"
// 输出：1
// 解释：把最后一位反向旋转一次即可 "0000" -> "0009"。
//
// 示例 3:
// 输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
// 输出：-1
// 解释：无法旋转到目标数字且不被锁定。
//
// 提示：
// 1 <= deadends.length <= 500
// deadends[i].length == 4
// target.length == 4
// target 不在 deadends 之中
// target 和 deadends[i] 仅由若干位数字组成
func openLock(deadends []string, target string) int {
	getNextNums := func(num string) []string {
		result := make([]string, 0)
		n := len(num)
		bytes := []byte(num)
		for i := 0; i < n; i++ {
			c := bytes[i]
			newCh := c
			if c == '0' {
				newCh = '9'
			} else {
				newCh--
			}
			bytes[i] = newCh
			result = append(result, string(bytes))

			newCh = c
			if c == '9' {
				newCh = '0'
			} else {
				newCh++
			}
			bytes[i] = newCh
			result = append(result, string(bytes))

			bytes[i] = c
		}

		return result
	}

	deadMap := make(map[string]bool)
	for _, dead := range deadends {
		deadMap[dead] = true
	}
	visited := make(map[string]bool)
	init := "0000"
	ok1 := deadMap[init]
	ok2 := deadMap[target]
	if ok1 || ok2 {
		return -1
	}
	step := 0
	queue := list.New()
	queue.PushBack(init)
	visited[init] = true
	for queue.Len() > 0 {
		n := queue.Len()
		for i := 0; i < n; i++ {
			front := queue.Front()
			queue.Remove(front)
			num := front.Value.(string)
			if deadMap[num] {
				continue
			}
			if num == target {
				return step
			}
			nextNums := getNextNums(num)
			for _, next := range nextNums {
				if visited[next] {
					continue
				}
				queue.PushBack(next)
				visited[next] = true
			}
		}

		step++
	}
	return -1
}

// 1700. 无法吃午餐的学生数量
// 学校的自助午餐提供圆形和方形的三明治，分别用数字 0 和 1 表示。所有学生站在一个队列里，每个学生要么喜欢圆形的要么喜欢方形的。
// 餐厅里三明治的数量与学生的数量相同。所有三明治都放在一个 栈 里，每一轮：
//
// 如果队列最前面的学生 喜欢 栈顶的三明治，那么会 拿走它 并离开队列。
// 否则，这名学生会 放弃这个三明治 并回到队列的尾部。
// 这个过程会一直持续到队列里所有学生都不喜欢栈顶的三明治为止。
//
// 给你两个整数数组 students 和 sandwiches ，其中 sandwiches[i] 是栈里面第 i​​​​​​ 个三明治的类型（i = 0 是栈的顶部）， students[j] 是初始队列里第 j​​​​​​ 名学生对三明治的喜好（j = 0 是队列的最开始位置）。请你返回无法吃午餐的学生数量。
//
// 示例 1：
// 输入：students = [1,1,0,0], sandwiches = [0,1,0,1]
// 输出：0
// 解释：
// - 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [1,0,0,1]。
// - 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [0,0,1,1]。
// - 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = [0,1,1]，三明治栈为 sandwiches = [1,0,1]。
// - 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [1,1,0]。
// - 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = [1,0]，三明治栈为 sandwiches = [0,1]。
// - 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [0,1]。
// - 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = [1]，三明治栈为 sandwiches = [1]。
// - 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = []，三明治栈为 sandwiches = []。
// 所以所有学生都有三明治吃。
//
// 示例 2：
// 输入：students = [1,1,1,0,0,1], sandwiches = [1,0,0,0,1,1]
// 输出：3
//
// 提示：
// 1 <= students.length, sandwiches.length <= 100
// students.length == sandwiches.length
// sandwiches[i] 要么是 0 ，要么是 1 。
// students[i] 要么是 0 ，要么是 1 。
func countStudents(students []int, sandwiches []int) int {
	n := len(students)
	queue := list.New()
	for _, student := range students {
		queue.PushBack(student)
	}
	index := 0
	for index < n && queue.Len() > 0 {
		count := 0
		front := queue.Front()
		// 不喜欢
		for front.Value.(int) != sandwiches[index] {
			if count == queue.Len() {
				return count
			}
			queue.Remove(front)
			queue.PushBack(front.Value.(int))
			count++
			front = queue.Front()
		}

		index++
		queue.Remove(front)
	}
	return 0
}
