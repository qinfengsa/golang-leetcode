package list

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v string_main_test.go string_main.go -test.run Test_convert

func Test_removeNthFromEnd(t *testing.T) {
	// [1,2,3,4,5], n = 2
	node1 := &ListNode{Val: 1}
	node2 := &ListNode{Val: 2}
	node3 := &ListNode{Val: 3}
	node4 := &ListNode{Val: 4}
	node5 := &ListNode{Val: 5}
	node1.Next = node2
	node2.Next = node3
	node3.Next = node4
	node4.Next = node5

	n := 2

	result := removeNthFromEnd(node1, n)
	fmt.Println(result)
}
