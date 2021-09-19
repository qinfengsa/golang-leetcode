package lru

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v lru_cache_test.go lru_cache.go -run Test_main
func Test_main(t *testing.T) {
	cache := Constructor(2)
	cache.Put(1, 1)
	cache.Put(2, 2)
	fmt.Printf("get 1 => %d", cache.Get(1))
	fmt.Println()
	cache.Put(3, 3)
	fmt.Printf("get 2 => %d", cache.Get(2))
	fmt.Println()
	fmt.Printf("get 1 => %d", cache.Get(1))
	fmt.Println()

}
