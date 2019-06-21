package ast

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestStringMultiDigit(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 100}}}}

	assert.Equal(`FUN INT main:
  params: ()
  body:
    RETURN Int<100>`, p.String())
}

func TestStringReturn0(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 0}}}}

	assert.Equal(`FUN INT main:
  params: ()
  body:
    RETURN Int<0>`, p.String())
}

func TestStringReturn2(t *testing.T) {
	assert := assert.New(t)
	p := Program{Function: &Function{Identifier: "main", Statement: &ReturnStatement{&Int32Literal{Value: 2}}}}

	assert.Equal(`FUN INT main:
  params: ()
  body:
    RETURN Int<2>`, p.String())
}
