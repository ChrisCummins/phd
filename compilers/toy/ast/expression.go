package ast

type Expression interface {
	GenerateAssembly() string
	String() string
}
