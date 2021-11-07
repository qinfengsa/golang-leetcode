package random

import (
	"math"
	"math/rand"
)

// Circular 478. 在圆内随机生成点
// 给定圆的半径和圆心的 x、y 坐标，写一个在圆中产生均匀随机点的函数 randPoint 。
//
// 说明:
// 输入值和输出值都将是浮点数。
// 圆的半径和圆心的 x、y 坐标将作为参数传递给类的构造函数。
// 圆周上的点也认为是在圆中。
// randPoint 返回一个包含随机点的x坐标和y坐标的大小为2的数组。
//
// 示例 1：
// 输入:  ["Solution","randPoint","randPoint","randPoint"]  [[1,0,0],[],[],[]]
// 输出: [null,[-0.72939,-0.65505],[-0.78502,-0.28626],[-0.83119,-0.19803]]
//
// 示例 2：
// 输入:  ["Solution","randPoint","randPoint","randPoint"]  [[10,5,-7.5],[],[],[]]
// 输出: [null,[11.52438,-8.33273],[2.46992,-16.21705],[11.13430,-12.42337]]
// 输入语法说明：
//
// 输入是两个列表：调用成员函数名和调用的参数。Solution 的构造函数有三个参数，圆的半径、圆心的 x 坐标、圆心的 y 坐标。randPoint 没有参数。输入参数是一个列表，即使参数为空，也会输入一个 [] 空列表。
type Circular struct {
	radius float64
	x      float64
	y      float64
}

func ConstructorCircular(radius float64, x_center float64, y_center float64) Circular {
	return Circular{radius: radius, x: x_center, y: y_center}
}

func (this *Circular) RandPoint() []float64 {
	// 随机半径
	randomRaius := math.Sqrt(rand.Float64()) * this.radius
	// 随机角度
	theta := rand.Float64() * 2.0 * math.Pi

	x, y := this.x+math.Cos(theta)*randomRaius, this.y+math.Sin(theta)*randomRaius
	return []float64{x, y}

}
