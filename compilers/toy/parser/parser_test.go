package parser

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"github.com/stretchr/testify/assert"
	"testing"
)

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestLexMultiDigit(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "100"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal(int32(100), p.Function.Statement.Expression.Value)
}

func TestLexNewlines(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal(int32(0), p.Function.Statement.Expression.Value)
}

func TestLexNoNewlines(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal(int32(0), p.Function.Statement.Expression.Value)
}

func TestLexReturn0(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal(int32(0), p.Function.Statement.Expression.Value)
}

func TestLexReturn2(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "2"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal(int32(2), p.Function.Statement.Expression.Value)
}

func TestLexSpaces(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal(int32(0), p.Function.Statement.Expression.Value)
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/invalid

func TestLexMissingParen(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestLexMissingRetval(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestLexNoBrace(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestLexNoSemicolon(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestLexNoSpace(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.IdentifierToken, "return0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestLexWrongCase(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.IdentifierToken, "RETURN"},
		token.Token{token.NumberToken, "0"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}
