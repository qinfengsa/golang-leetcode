package calendar

import (
	"github.com/emirpasic/gods/trees/redblacktree"
)

// MyCalendar 729. 我的日程安排表 I
// 实现一个 MyCalendar 类来存放你的日程安排。如果要添加的日程安排不会造成 重复预订 ，则可以存储这个新的日程安排。
//
// 当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生 重复预订 。
//
// 日程可以用一对整数 start 和 end 表示，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end 。
//
// 实现 MyCalendar 类：
//
// MyCalendar() 初始化日历对象。
// boolean book(int start, int end) 如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true 。否则，返回 false 并且不要将该日程安排添加到日历中。
//
// 示例：
// 输入：
// ["MyCalendar", "book", "book", "book"]
// [[], [10, 20], [15, 25], [20, 30]]
// 输出：
// [null, true, false, true]
//
// 解释：
// MyCalendar myCalendar = new MyCalendar();
// myCalendar.book(10, 20); // return True
// myCalendar.book(15, 25); // return False ，这个日程安排不能添加到日历中，因为时间 15 已经被另一个日程安排预订了。
// myCalendar.book(20, 30); // return True ，这个日程安排可以添加到日历中，因为第一个日程安排预订的每个时间都小于 20 ，且不包含时间 20 。
//
// 提示：
// 0 <= start < end <= 109
// 每个测试用例，调用 book 方法的次数最多不超过 1000 次。
type MyCalendar struct {
	*redblacktree.Tree
}

func Constructor() MyCalendar {
	tree := redblacktree.NewWithIntComparator()
	tree.Put(-1, -1)
	return MyCalendar{tree}
}

func (this *MyCalendar) Book(start int, end int) bool {
	// 找左边的节点 肯定存在
	left, _ := this.Floor(start)
	leftVal := left.Value.(int)
	if leftVal > start {
		return false
	}

	right, rightFound := this.Ceiling(start)

	if rightFound {
		rightKey := right.Key.(int)
		if end > rightKey {
			return false
		}
	}
	this.Put(start, end)
	return true
}

// MyCalendarTwo 731. 我的日程安排表 II
// 实现一个 MyCalendar 类来存放你的日程安排。如果要添加的时间内不会导致三重预订时，则可以存储这个新的日程安排。
//
// MyCalendar 有一个 book(int start, int end)方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end。
//
// 当三个日程安排有一些时间上的交叉时（例如三个日程安排都在同一时间内），就会产生三重预订。
//
// 每次调用 MyCalendar.book方法时，如果可以将日程安排成功添加到日历中而不会导致三重预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。
//
// 请按照以下步骤调用MyCalendar 类: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)
//
// 示例：
// MyCalendar();
// MyCalendar.book(10, 20); // returns true
// MyCalendar.book(50, 60); // returns true
// MyCalendar.book(10, 40); // returns true
// MyCalendar.book(5, 15); // returns false
// MyCalendar.book(5, 10); // returns true
// MyCalendar.book(25, 55); // returns true
// 解释：
// 前两个日程安排可以添加至日历中。 第三个日程安排会导致双重预订，但可以添加至日历中。
// 第四个日程安排活动（5,15）不能添加至日历中，因为它会导致三重预订。
// 第五个日程安排（5,10）可以添加至日历中，因为它未使用已经双重预订的时间10。
// 第六个日程安排（25,55）可以添加至日历中，因为时间 [25,40] 将和第三个日程安排双重预订；
// 时间 [40,50] 将单独预订，时间 [50,55）将和第二个日程安排双重预订。
//
// 提示：
// 每个测试用例，调用 MyCalendar.book 函数最多不超过 1000次。
// 调用函数 MyCalendar.book(start, end)时， start 和 end 的取值范围为 [0, 10^9]。
type MyCalendarTwo struct {
	*redblacktree.Tree
}

func ConstructorII() MyCalendarTwo {
	return MyCalendarTwo{redblacktree.NewWithIntComparator()}
}

func (this *MyCalendarTwo) Book(start int, end int) bool {
	left, leftFound := this.Get(start)
	right, rightFound := this.Get(end)
	if leftFound {
		this.Put(start, left.(int)+1)
	} else {
		this.Put(start, 1)
	}
	if rightFound {
		this.Put(end, right.(int)-1)
	} else {
		this.Put(end, -1)
	}
	active := 0
	for _, val := range this.Values() {
		active += val.(int)
		if active >= 3 {
			left, leftFound = this.Get(start)
			right, rightFound = this.Get(end)
			this.Put(start, left.(int)-1)
			this.Put(end, right.(int)+1)
			return false
		}
	}
	return true
}

// MyCalendarThree 732. 我的日程安排表 III
// 当 k 个日程安排有一些时间上的交叉时（例如 k 个日程安排都在同一时间内），就会产生 k 次预订。
//
// 给你一些日程安排 [start, end) ，请你在每个日程安排添加后，返回一个整数 k ，表示所有先前日程安排会产生的最大 k 次预订。
//
// 实现一个 MyCalendarThree 类来存放你的日程安排，你可以一直添加新的日程安排。
//
// MyCalendarThree() 初始化对象。
// int book(int start, int end) 返回一个整数 k ，表示日历中存在的 k 次预订的最大值。
//
// 示例：
// 输入：
// ["MyCalendarThree", "book", "book", "book", "book", "book", "book"]
// [[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
// 输出：
// [null, 1, 1, 2, 3, 3, 3]
//
// 解释：
// MyCalendarThree myCalendarThree = new MyCalendarThree();
// myCalendarThree.book(10, 20); // 返回 1 ，第一个日程安排可以预订并且不存在相交，所以最大 k 次预订是 1 次预订。
// myCalendarThree.book(50, 60); // 返回 1 ，第二个日程安排可以预订并且不存在相交，所以最大 k 次预订是 1 次预订。
// myCalendarThree.book(10, 40); // 返回 2 ，第三个日程安排 [10, 40) 与第一个日程安排相交，所以最大 k 次预订是 2 次预订。
// myCalendarThree.book(5, 15); // 返回 3 ，剩下的日程安排的最大 k 次预订是 3 次预订。
// myCalendarThree.book(5, 10); // 返回 3
// myCalendarThree.book(25, 55); // 返回 3
//
// 提示：
// 0 <= start < end <= 109
// 每个测试用例，调用 book 函数最多不超过 400次
type MyCalendarThree struct {
	*redblacktree.Tree
}

func ConstructorIII() MyCalendarThree {
	return MyCalendarThree{redblacktree.NewWithIntComparator()}
}

func (this *MyCalendarThree) Book(start int, end int) int {
	left, leftFound := this.Get(start)
	right, rightFound := this.Get(end)
	if leftFound {
		this.Put(start, left.(int)+1)
	} else {
		this.Put(start, 1)
	}
	if rightFound {
		this.Put(end, right.(int)-1)
	} else {
		this.Put(end, -1)
	}
	active, result := 0, 0
	for _, val := range this.Values() {
		active += val.(int)
		if active > result {
			result = active
		}
	}
	return result
}
