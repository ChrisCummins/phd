package ast

import (
	"fmt"

	"github.com/ChrisCummins/phd/compilers/toy/token"
)

type UnaryOp struct {
	Operator   *UnaryOpOperator
	Expression Expression
}

func (u UnaryOp) String() string {
	return fmt.Sprintf("%v %v", u.Operator, u.Expression)
}

func (u *UnaryOp) GenerateAssembly() string {
	op := u.Operator.GenerateAssembly()
	exp := u.Expression.GenerateAssembly()
	return fmt.Sprintf("%v\n\t%v", exp, op)
}

type UnaryOpOperator struct {
	Type token.TokenType
}

func (u UnaryOpOperator) String() string {
	switch u.Type {
	case token.LogicalNegationToken:
		return "!"
	case token.BitwiseComplementToken:
		return "~"
	case token.NegationToken:
		return "-"
	}
	panic("unreachable!")
}

func (u *UnaryOpOperator) GenerateAssembly() string {
	switch u.Type {
	case token.LogicalNegationToken:
		return ("cmpl    $0, %eax  # logical negation\n\t" +
			"movl    $0, %eax\n\t" +
			"sete    %al")
	case token.BitwiseComplementToken:
		return "not     %eax  # bitwise complement"
	case token.NegationToken:
		return "neg     %eax  # unary negation"
	}
	panic("unreachable!")
}
