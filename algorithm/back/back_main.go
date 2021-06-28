package back

import "fmt"

// 回溯算法

// 二进制手表顶部有 4 个 LED 代表 小时（0-11），底部的 6 个 LED 代表 分钟（0-59）。
//
// 每个 LED 代表一个 0 或 1，最低位在右侧。
//
// 例如，上面的二进制手表读取 “3:25”。
//
// 给定一个非负整数 n代表当前 LED 亮着的数量，返回所有可能的时间。
//
// 示例：
//
// 输入: n = 1
// 返回: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
//
// 提示：
//
// 输出的顺序没有要求。
// 小时不会以零开头，比如 “01:00”是不允许的，应为 “1:00”。
// 分钟必须由两位数组成，可能会以零开头，比如 “10:2”是无效的，应为 “10:02”。
// 超过表示范围（小时 0-11，分钟 0-59）的数据将会被舍弃，也就是说不会出现 "13:00", "0:61" 等时间。
func readBinaryWatch(num int) []string {
	var result []string

	leds := [10]bool{}
	readBinaryWatchBack(num, 0, &leds, &result)
	return result
}

func readBinaryWatchBack(num int, start int, leds *[10]bool, list *[]string) {
	if num == 0 {
		hours, minutes := getTime(leds)
		if hours != -1 {
			*list = append(*list, fmt.Sprintf("%d:%02d", hours, minutes))
		}
		return
	}
	for i := start; i < 10; i++ {
		if leds[i] {
			continue
		}
		leds[i] = true
		readBinaryWatchBack(num-1, i+1, leds, list)
		leds[i] = false
	}
}

func getTime(leds *[10]bool) (int, int) {
	hours, minutes := 0, 0
	// 8 4 2 1
	for i := 0; i < 4; i++ {
		if leds[i] {
			hours += 1 << (3 - i)
		}
	}
	if hours >= 12 {
		return -1, -1
	}

	// 32 16 8 4 2 1
	for i := 4; i < 10; i++ {
		if leds[i] {
			minutes += 1 << (9 - i)
		}
	}
	if minutes > 59 {
		return -1, -1
	}

	return hours, minutes
}
