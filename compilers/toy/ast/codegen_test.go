package ast

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestGenerateAssemblyMultiDigit(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 100}}}}

	asm := p.GenerateAssembly()
	assert.Equal(`.globl _main
_main:
 movl    $100, %eax
 ret
`, asm)
}

func TestGenerateAssemblyReturn0(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 0}}}}

	asm := p.GenerateAssembly()
	assert.Equal(`.globl _main
_main:
 movl    $0, %eax
 ret
`, asm)
}

func TestGenerateAssemblyReturn2(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 2}}}}

	asm := p.GenerateAssembly()
	assert.Equal(`.globl _main
_main:
 movl    $2, %eax
 ret
`, asm)
}
