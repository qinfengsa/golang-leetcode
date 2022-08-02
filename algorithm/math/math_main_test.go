package math

import (
	"reflect"
	"testing"
)

func Test_evaluate(t *testing.T) {
	tests := []struct {
		name string
		args string
		want int
	}{
		{"1", "(let x 2 (mult x (let x 3 y 4 (add x y))))", 14},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := evaluate(test.args); !reflect.DeepEqual(got, test.want) {
				t.Errorf("evaluate() = %v, want %v", got, test.want)
			}
		})
	}
}
