package ast

import "fmt"

type ReturnStatement struct {
	// This will later be extended to different return types.
	Expression Expression
}

func (s ReturnStatement) String() string {
	return fmt.Sprintf("RETURN %v", s.Expression.String())
}

func (s *ReturnStatement) GenerateAssembly() string {
	expression := s.Expression.GenerateAssembly()
	return fmt.Sprintf("%v\n\tret", expression)
}
