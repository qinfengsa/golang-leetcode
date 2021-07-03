package hash

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v hash_main_test.go hash_main.go -test.run Test_frequencySort
func Test_frequencySort(t *testing.T) {
	s := "tree"
	result := frequencySort(s)
	fmt.Println(result)
}
