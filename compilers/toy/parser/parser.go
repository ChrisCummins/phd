package parser

import (
	"errors"
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/ast"
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"strconv"
)

// Parser grammar (in Backus-Naur Form):
//     <program> ::= <function>
//     <function> ::= "int" <id> "(" ")" "{" <statement> "}"
//     <statement> ::= "return" <exp> ";"
//     <exp> ::= <unary_op> <exp> | <int>
//     <unary_op> ::= "!" | "~" | "-"

func Parse(ts token.TokenStream) (*ast.Program, error) {
	function, err := ParseFunction(ts)
	if err != nil {
		return nil, err
	}
	if ts.Next() {
		return nil, errors.New("expected EOF")
	}
	return &ast.Program{Function: function}, nil
}

func ParseFunction(ts token.TokenStream) (*ast.Function, error) {
	if !ts.Next() || ts.Value().Type != token.IntKeywordToken {
		return nil, errors.New("expected `int` keyword")
	}

	if !ts.Next() || ts.Value().Type != token.IdentifierToken {
		return nil, errors.New("expected identifier")
	}
	identifier := ts.Value().Value

	if !ts.Next() || ts.Value().Type != token.OpenParenthesisToken {
		return nil, errors.New("expected `(`")
	}
	if !ts.Next() || ts.Value().Type != token.CloseParenthesisToken {
		return nil, errors.New("expected `)`")
	}
	if !ts.Next() || ts.Value().Type != token.OpenBraceToken {
		return nil, errors.New("expected `{`")
	}
	statement, err := ParseStatement(ts)
	if err != nil {
		return nil, err
	}
	if !ts.Next() || ts.Value().Type != token.CloseBraceToken {
		return nil, errors.New("expected `}`")
	}

	return &ast.Function{Identifier: identifier, Statement: statement}, nil
}

func ParseStatement(ts token.TokenStream) (*ast.ReturnStatement, error) {
	if !ts.Next() || ts.Value().Type != token.ReturnKeywordToken {
		return nil, errors.New("expected `return` keyword")
	}

	expression, err := ParseExpression(ts)
	if err != nil {
		return nil, err
	}

	statement := ast.ReturnStatement{Expression: expression}
	if !ts.Next() || ts.Value().Type != token.SemicolonToken {
		return nil, errors.New("expected semicolon")
	}

	return &statement, nil
}

func ParseExpression(ts token.TokenStream) (ast.Expression, error) {
	if !ts.Next() {
		return nil, errors.New("ran out of tokens")
	}

	if ts.Value().Type == token.NumberToken {
		return ConsumeIntegerLiteral(ts.Value())
	}

	op, err := ConsumeUnaryOp(ts.Value())
	if err != nil {
		return nil, err
	}

	exp, err := ParseExpression(ts)
	if err != nil {
		return nil, err
	}

	return &ast.UnaryOp{Operator: op, Expression: exp}, nil
}

func ConsumeIntegerLiteral(t token.Token) (*ast.Int32Literal, error) {
	i, err := strconv.Atoi(t.Value)
	if err != nil {
		return nil, err
	}
	return &ast.Int32Literal{Value: int32(i)}, nil
}

func ConsumeUnaryOp(t token.Token) (*ast.UnaryOpOperator, error) {
	switch t.Type {
	case token.LogicalNegationToken:
		return &ast.UnaryOpOperator{Type: t.Type}, nil
	case token.BitwiseComplementToken:
		return &ast.UnaryOpOperator{Type: t.Type}, nil
	case token.NegationToken:
		return &ast.UnaryOpOperator{Type: t.Type}, nil
	default:
		return nil, errors.New(
			fmt.Sprintf("invalid unary operator `%v`", t.Value))
	}
}
