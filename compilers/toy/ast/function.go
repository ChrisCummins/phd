package ast

import "fmt"

type Function struct {
	Identifier string
	// This will later be extended to multiple statements and types.
	Statement *ReturnStatement
}

func (f Function) String() string {
	return fmt.Sprintf("FUN INT %v:\n  params: ()\n  body:\n    %v",
		f.Identifier, f.Statement.String())
}

func (f *Function) GenerateAssembly() string {
	stmt := f.Statement.GenerateAssembly()
	return fmt.Sprintf(".globl _%v\n_%v:\n\t%v\n", f.Identifier,
		f.Identifier, stmt)
}
