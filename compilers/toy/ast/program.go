package ast

type Program struct {
	// This will later be extended to multiple functions.
	Function *Function
}

func (p Program) String() string {
	return p.Function.String()
}

func (p *Program) GenerateAssembly() string {
	return p.Function.GenerateAssembly()
}
