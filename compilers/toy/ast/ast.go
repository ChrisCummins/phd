package ast

type Program struct {
	// This will later be extended to multiple functions.
	Function *Function
}

type Function struct {
	Identifier string
	// This will later be extended to multiple statements and types.
	Statement *ReturnStatement
}

type ReturnStatement struct {
	// This will later be extended to different return types.
	Expression *Int32Literal
}

type Int32Literal struct {
	Value int32
}
