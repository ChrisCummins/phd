package ast

import (
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/token"
)

type BinaryOp struct {
	Operator *BinaryOpOperator
	Term     Expression
	NextTerm Expression
}

func (u BinaryOp) String() string {
	return fmt.Sprintf("%v %v %v", u.Term, u.Operator, u.NextTerm)
}

func (u *BinaryOp) GenerateAssembly() string {
	// TODO:
	panic("TODO")
}

type BinaryOpOperator struct {
	Type token.TokenType
}

func (u BinaryOpOperator) String() string {
	switch u.Type {
	case token.AdditionToken:
		return "+"
	case token.NegationToken:
		return "-"
	case token.MultiplicationToken:
		return "*"
	case token.DivisionToken:
		return "/"
	}
	panic("unreachable!")
}

func (u *BinaryOpOperator) GenerateAssembly() string {
	// TODO:
	panic("unreachable!")
}
