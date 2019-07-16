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
	Token token.Token
}

func (op BinaryOpOperator) String() string {
	if !IsBinaryOp(op.Token.Type) {
		panic("unreachable!")
	}
	return op.Token.Value
}

func (u *BinaryOpOperator) GenerateAssembly() string {
	switch u.Token.Type {
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

func IsBinaryOp(t token.TokenType) bool {
	switch t {
	case token.NegationToken:
		return true
	case token.AdditionToken:
		return true
	case token.MultiplicationToken:
		return true
	case token.DivisionToken:
		return true
	case token.AndToken:
		return true
	case token.OrToken:
		return true
	case token.GreaterThanToken:
		return true
	case token.GreaterThanOrEqualToken:
		return true
	case token.LessThanToken:
		return true
	case token.LessThanOrEqualToken:
		return true
	case token.EqualToken:
		return true
	case token.NotEqualToken:
		return true
	default:
		return false
	}
}
