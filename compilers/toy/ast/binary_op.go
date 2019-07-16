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
	return fmt.Sprintf("%v\n\t"+
		"push    %%rax\n\t"+
		"%v\n\t"+
		"pop     %%rcx\n\t"+
		"%v",
		u.Term.GenerateAssembly(), u.NextTerm.GenerateAssembly(),
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
		return "subl    %eax, %ecx  # - operator"
	case token.AdditionToken:
		return "addl    %ecx, %eax  # + operator"
	case token.MultiplicationToken:
		return "imul    %ecx, %eax  # * operator"
	case token.DivisionToken:
		return ("movl    $0, %edx  # / operator\n\t" +
			"idivl   %ecx\n\t" +
			"movl    %ecx, %eax")
	case token.EqualToken:
		return ("cmpl    %eax, %ecx  # == operator\n\t" +
			"movl    $0, %eax\n\t" +
			"sete    %al")
	case token.NotEqualToken:
		return ("cmpl    %eax, %ecx  # != operator\n\t" +
			"movl    $0, %eax\n\t" +
			"setne   %al")
	case token.LessThanToken:
		return ("cmpl    %eax, %ecx  # < operator\n\t" +
			"movl    $0, %eax\n\t" +
			"setl    %al")
	case token.LessThanOrEqualToken:
		return ("cmpl    %eax, %ecx  # <= operator\n\t" +
			"movl    $0, %eax\n\t" +
			"setle   %al")
	case token.GreaterThanToken:
		return ("cmpl    %eax, %ecx  # < operator\n\t" +
			"movl    $0, %eax\n\t" +
			"setg    %al")
	case token.GreaterThanOrEqualToken:
		return ("cmpl    %eax, %ecx  # <= operator\n\t" +
			"movl    $0, %eax\n\t" +
			"setge   %al")
	case token.OrToken:
		return ("cmpl    $0, %eax  # || operator\n\t" +
			"je      _there\n\t" +
			"movl    $1, %eax\n\t" +
			"jmp     _end\n" +
			"_there:\n\t" +
			"cmpl    $0, %ecx\n\t" +
			"movl    $0, %eax\n\t" +
			"setne   %al\n" +
			"_end:")
	case token.AndToken:
		return ("cmpl    $0, %eax  # || operator\n\t" +
			"jne     _there\n\t" +
			"jmp     _end\n" +
			"_there:\n\t" +
			"cmpl    $0, %ecx\n\t" +
			"movl    $0, %eax\n\t" +
			"setne   %al\n" +
			"_end:")
	default:
		panic(fmt.Sprintf("Code-gen not support for binary op `%s`", u.Token))
	}
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
