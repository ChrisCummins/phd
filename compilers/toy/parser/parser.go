package parser

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/ChrisCummins/phd/compilers/toy/ast"
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"github.com/golang/glog"
)

// Parser grammar (in Backus-Naur Form):
//     <program> ::= <function>
//     <function> ::= "int" <id> "(" ")" "{" <statement> "}"
//     <statement> ::= "return" <exp> ";"
//     <exp> ::= <logical-and-exp> { "||" <logical-and-exp> }
//     <logical-and-exp> ::= <equality-exp> { "&&" <equality-exp> }
//     <equality-exp> ::= <relational-exp> { ("!=" | "==") <relational-exp> }
//     <relational-exp> ::= <additive-exp> { ("<" | ">" | "<=" | ">=") <additive-exp> }
//     <additive-exp> ::= <term> { ("+" | "-") <term> }
//     <term> ::= <factor> { ("*" | "/") <factor> }
//     <factor> ::= "(" <exp> ")" | <unary_op> <factor> | <int>
//     <unary_op> ::= "!" | "~" | "-"

// <program> ::= <function>
func Parse(ts token.TokenStream) (*ast.Program, error) {
	glog.V(2).Infof("Parse() <- %s", ts.Peek())

	function, err := ParseFunction(ts)
	if err != nil {
		return nil, err
	}
	if ts.Next() {
		return nil, errors.New("expected EOF")
	}
	return &ast.Program{Function: function}, nil
}

// <function> ::= "int" <id> "(" ")" "{" <statement> "}"
func ParseFunction(ts token.TokenStream) (*ast.Function, error) {
	glog.V(2).Infof("ParseFunction() <- %s", ts.Peek())
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

// <statement> ::= "return" <exp> ";"
func ParseStatement(ts token.TokenStream) (*ast.ReturnStatement, error) {
	glog.V(2).Infof("ParseStatement() <- %s", ts.Peek())
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

// <exp> ::= <term> { ("+" | "-") <term> }
func ParseExpression(ts token.TokenStream) (ast.Expression, error) {
	term, err := ParseTerm(ts)
	if err != nil {
		return nil, err
	}

	for {
		if ts.Peek().Type != token.AdditionToken &&
			ts.Peek().Type != token.NegationToken {
			break
		}
		if !ts.Next() {
			panic("unreachable!")
		}
		op, err := consumeBinaryOp(ts.Value())
		if err != nil {
			return nil, err
		}
		//if !ts.Next() {
		//	return nil, errors.New("ran out of tokens")
		//}
		next_term, err := ParseTerm(ts)
		if err != nil {
			return nil, err
		}
		term = &ast.BinaryOp{Operator: op, Term: term, NextTerm: next_term}
	}

	return term, nil
}

// <term> ::= <factor> { ("*" | "/") <factor> }
func ParseTerm(ts token.TokenStream) (ast.Expression, error) {
	glog.V(2).Infof("ParseTerm() <- %s", ts.Peek())

	factor, err := ParseFactor(ts)
	if err != nil {
		return nil, err
	}

	for {
		if ts.Peek().Type != token.MultiplicationToken &&
			ts.Peek().Type != token.DivisionToken {
			break
		}
		if !ts.Next() {
			panic("unreachable!")
		}
		op, err := consumeBinaryOp(ts.Value())
		if err != nil {
			return nil, err
		}
		//if !ts.Next() {
		//	return nil, errors.New("ran out of tokens")
		//}
		next_factor, err := ParseFactor(ts)
		if err != nil {
			return nil, err
		}
		factor = &ast.BinaryOp{Operator: op, Term: factor, NextTerm: next_factor}
	}

	return factor, nil
}

// <factor> ::= "(" <exp> ")" | <unary_op> <factor> | <int>
func ParseFactor(ts token.TokenStream) (ast.Expression, error) {
glog.V(2).Infof("ParseFactor() <- %s", ts.Peek())

	if !ts.Next() {
		return nil, errors.New("ran out of tokens")
	}

	if ts.Value().Type == token.OpenParenthesisToken {
		// <factor> ::= "(" <exp> ")"
		exp, err := ParseExpression(ts)
		if err != nil {
			return nil, err
		}
		if !ts.Next() || ts.Value().Type != token.CloseParenthesisToken {
			return nil, errors.New("expected closing parenthesis")
		}
		return exp, nil
	}

	if isUnaryOp(ts.Value()) {
		// <factor> ::= <unary_op> <factor>
		op, err := consumeUnaryOp(ts.Value())
		if err != nil {
			return nil, err
		}
		factor, err := ParseFactor(ts)
		if err != nil {
			return nil, err
		}
		return &ast.UnaryOp{Operator: op, Expression: factor}, nil
	}

	if ts.Value().Type == token.NumberToken {
		// <factor> ::= <int>
		return consumeIntegerLiteral(ts.Value())
	}

	return nil, errors.New(fmt.Sprintf("invalid token `%v`", ts.Value()))
}

func consumeIntegerLiteral(t token.Token) (*ast.Int32Literal, error) {
	i, err := strconv.Atoi(t.Value)
	if err != nil {
		return nil, err
	}
	return &ast.Int32Literal{Value: int32(i)}, nil
}

func isUnaryOp(t token.Token) bool {
	switch t.Type {
	case token.LogicalNegationToken:
		return true
	case token.BitwiseComplementToken:
		return true
	case token.NegationToken:
		return true
	default:
		return false
	}
}

// <unary_op> ::= "!" | "~" | "-"
func consumeUnaryOp(t token.Token) (*ast.UnaryOpOperator, error) {
	if !isUnaryOp(t) {
		return nil, errors.New(
			fmt.Sprintf("invalid unary operator `%v`", t.Value))
	}
	return &ast.UnaryOpOperator{Type: t.Type}, nil
}

func isBinaryOp(t token.Token) bool {
	switch t.Type {
	case token.NegationToken:
		return true
	case token.AdditionToken:
		return true
	case token.MultiplicationToken:
		return true
	case token.DivisionToken:
		return true
	default:
		return false
	}
}

func consumeBinaryOp(t token.Token) (*ast.BinaryOpOperator, error) {
	if !isBinaryOp(t) {
		return nil, errors.New(
			fmt.Sprintf("invalid binary operator `%v`", t.Value))
	}
	return &ast.BinaryOpOperator{Type: t.Type}, nil
}
