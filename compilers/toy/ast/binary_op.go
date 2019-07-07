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
	return fmt.Sprintf("(%v %v %v)", u.Term, u.Operator, u.NextTerm)
}

func (u *BinaryOp) GenerateAssembly() string {
	return fmt.Sprintf("%v\n\tpush    %%rax\n\t%v\n\tpop     %%rcx\n\t%v",
		u.NextTerm.GenerateAssembly(), u.Term.GenerateAssembly(),
		u.Operator.GenerateAssembly())
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
	switch u.Type {
	case token.NegationToken:
		return "subl    %ecx, %eax"
	case token.AdditionToken:
		return "addl    %ecx, %eax"
	case token.MultiplicationToken:
		return "imul    %ecx, %eax"
	case token.DivisionToken:
		return ("movl    $0, %edx\n\tidivl   %ecx\n\t" +
			"movl    %ecx, %eax")
	}
	panic("unreachable!")
}
