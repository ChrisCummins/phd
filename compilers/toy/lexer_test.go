package compiler

import "testing"

func TestLex(t *testing.T) {
	if Lex("Hello, world!") != 5 {
		t.Error("not 5")
	}
}
