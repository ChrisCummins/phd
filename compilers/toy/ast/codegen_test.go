package ast

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestGenerateAssemblyMultiDigit(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 100}}}}

	as, err := p.GenerateAssembly()
	assert.Nil(err)
	assert.Equal(`.globl _main
_main:
 movl    $100, %eax
 ret
`, as)
}

func TestGenerateAssemblyReturn0(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 0}}}}

	as, err := p.GenerateAssembly()
	assert.Nil(err)
	assert.Equal(`.globl _main
_main:
 movl    $0, %eax
 ret
`, as)
}

func TestGenerateAssemblyReturn2(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 2}}}}

	as, err := p.GenerateAssembly()
	assert.Nil(err)
	assert.Equal(`.globl _main
_main:
 movl    $2, %eax
 ret
`, as)
}
