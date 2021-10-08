package codec

import (
	"strconv"
	"strings"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// Codec
// 297. 二叉树的序列化与反序列化
// 序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。
//
// 请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。
//
// 提示: 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。
//
// 示例 1：
// 输入：root = [1,2,3,null,null,4,5]
// 输出：[1,2,3,null,null,4,5]
//
// 示例 2：
// 输入：root = []
// 输出：[]
//
// 示例 3：
// 输入：root = [1]
// 输出：[1]
//
// 示例 4：
// 输入：root = [1,2]
// 输出：[1,2]
//
// 提示：
// 树中结点数在范围 [0, 104] 内
// -1000 <= Node.val <= 1000
type Codec struct {
}

func Constructor() Codec {
	return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	var builder strings.Builder
	var serial func(node *TreeNode)

	serial = func(node *TreeNode) {
		if builder.Len() > 0 {
			builder.WriteString(",")
		}
		if node == nil {
			builder.WriteString("#")
			return
		}
		builder.WriteString(strconv.Itoa(node.Val))
		serial(node.Left)
		serial(node.Right)
	}
	serial(root)
	return builder.String()
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {

	nodes := strings.Split(data, ",")
	idx, n := 0, len(nodes)

	var deserial func() *TreeNode

	deserial = func() *TreeNode {
		if idx == n {
			return nil
		}
		val := nodes[idx]
		idx++
		if val == "#" {
			return nil
		}
		intVal, _ := strconv.Atoi(val)
		node := &TreeNode{Val: intVal}
		node.Left = deserial()
		node.Right = deserial()
		return node
	}

	return deserial()
}
