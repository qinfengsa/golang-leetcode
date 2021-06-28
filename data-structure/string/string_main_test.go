package string

import (
	"fmt"
	"testing"
)

func Test_longestPalindrome(t *testing.T) {
	s := "babad"
	result := longestPalindrome(s)
	fmt.Println(result)
}
