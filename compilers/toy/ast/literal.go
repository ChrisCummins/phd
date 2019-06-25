package ast

import "fmt"

type Int32Literal struct {
	Value int32
}

func (i Int32Literal) String() string {
	return fmt.Sprintf("Int<%v>", i.Value)
}

func (i *Int32Literal) GenerateAssembly() string {
	return fmt.Sprintf("movl    $%v, %%eax", i.Value)
}
