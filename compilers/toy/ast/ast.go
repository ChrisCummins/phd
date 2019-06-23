package ast

import (
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/token"
)

type Program struct {
	// This will later be extended to multiple functions.
	Function *Function
}

func (p Program) String() string {
	return p.Function.String()
}

func (p *Program) GenerateAssembly() (string, error) {
	return p.Function.GenerateAssembly()
}

type Function struct {
	Identifier string
	// This will later be extended to multiple statements and types.
	Statement *ReturnStatement
}

func (f Function) String() string {
	return fmt.Sprintf("FUN INT %v:\n  params: ()\n  body:\n    %v",
		f.Identifier, f.Statement.String())
}

func (f *Function) GenerateAssembly() (string, error) {
	s, err := f.Statement.GenerateAssembly()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(".globl _%v\n_%v:\n%v\n", f.Identifier,
		f.Identifier, s), nil
}

type ReturnStatement struct {
	// This will later be extended to different return types.
	Expression Expression
}

func (s ReturnStatement) String() string {
	return fmt.Sprintf("RETURN %v", s.Expression.String())
}

func (s *ReturnStatement) GenerateAssembly() (string, error) {
	expression, err := s.Expression.GenerateAssembly()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(" movl    %v, %%eax\n ret", expression), nil
}

type Expression interface {
	GenerateAssembly() (string, error)
	String() string
}

type Int32Literal struct {
	Value int32
}

func (i Int32Literal) String() string {
	return fmt.Sprintf("Int<%v>", i.Value)
}

func (i *Int32Literal) GenerateAssembly() (string, error) {
	return fmt.Sprintf("$%v", i.Value), nil
}

type UnaryOp struct {
	Operator   *UnaryOpOperator
	Expression Expression
}

func (u UnaryOp) String() string {
	return fmt.Sprintf("%v %v", u.Operator, u.Expression)
}

func (u *UnaryOp) GenerateAssembly() (string, error) {
	op, err := u.Operator.GenerateAssembly()
	if err != nil {
		return "", err
	}
	exp, err := u.Expression.GenerateAssembly()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%v\n %v", exp, op), nil
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

func (u *UnaryOpOperator) GenerateAssembly() (string, error) {
	switch u.Type {
	case token.LogicalNegationToken:
		return "cmpl   $0, %eax\n movl   $0, %eax\n sete   %al", nil
	case token.BitwiseComplementToken:
		return "not     %eax", nil
	case token.NegationToken:
		return "neg     %eax", nil
	}
	panic("unreachable!")
}
