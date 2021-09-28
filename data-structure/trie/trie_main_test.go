package trie

import (
	"fmt"
	"testing"
)

// go test -v trie_main_test.go trie_main.go -run Test_findWords
func Test_findWords(t *testing.T) {
	board := [][]byte{{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}}

	words := []string{"oath", "pea", "eat", "rain"}

	result := findWords(board, words)
	fmt.Println(result)
}
