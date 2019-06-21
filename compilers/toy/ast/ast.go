package ast

import "fmt"

type Program struct {
	// This will later be extended to multiple functions.
	Function *Function
}

func (p *Program) GenerateAssembly() (string, error) {
	return p.Function.GenerateAssembly()
}

type Function struct {
	Identifier string
	// This will later be extended to multiple statements and types.
	Statement *ReturnStatement
}

func (f *Function) GenerateAssembly() (string, error) {
	s, err := f.Statement.GenerateAssembly()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(".globl _%v\n_%v:\n%v", f.Identifier,
		f.Identifier, s), nil
}

type ReturnStatement struct {
	// This will later be extended to different return types.
	Expression *Int32Literal
}

func (s *ReturnStatement) GenerateAssembly() (string, error) {
	expression, err := s.Expression.GenerateAssembly()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(" movl    %v, %%eax\n ret", expression), nil
}

type Int32Literal struct {
	Value int32
}

func (i *Int32Literal) GenerateAssembly() (string, error) {
	return fmt.Sprintf("$%v", i.Value), nil
}
