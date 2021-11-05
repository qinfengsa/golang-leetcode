package back

import (
	"fmt"
	"testing"
)

// test 命令
// go test -v string_main_test.go string_main.go -test.run Test_convert

func Test_letterCombinations(t *testing.T) {
	digits := "23"
	result := letterCombinations(digits)
	fmt.Println(result)
}

func Test_generateParenthesis(t *testing.T) {
	n := 3
	result := generateParenthesis(n)
	fmt.Println(result)
}

func Test_solveSudoku(t *testing.T) {

}

func Test_combinationSum(t *testing.T) {
	candidates := []int{2, 3, 5}
	target := 8
	result := combinationSum(candidates, target)
	fmt.Println(result)
}

func Test_combinationSum2(t *testing.T) {
	candidates := []int{10, 1, 2, 7, 6, 1, 5}
	target := 8
	result := combinationSum2(candidates, target)
	fmt.Println(result)
}

func Test_permute(t *testing.T) {
	nums := []int{1, 2, 3}
	result := permute(nums)
	fmt.Println(result)
}

func Test_permuteUnique(t *testing.T) {
	nums := []int{1, 3, 2}
	result := permuteUnique(nums)
	fmt.Println(result)
}

func Test_solveNQueens(t *testing.T) {
	n := 4
	result := solveNQueens(n)
	fmt.Println(result)
}

func Test_totalNQueens(t *testing.T) {
	n := 4
	result := totalNQueens(n)
	fmt.Println(result)
}

func Test_makesquare(t *testing.T) {
	matchsticks := []int{13, 11, 1, 8, 6, 7, 8, 8, 6, 7, 8, 9, 8}
	result := makesquare(matchsticks)
	fmt.Println(result)
}
