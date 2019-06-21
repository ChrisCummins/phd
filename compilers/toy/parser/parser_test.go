package parser

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"github.com/stretchr/testify/assert"
	"testing"
)

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestParseMultiDigit(t *testing.T) {
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

func TestParseNewlines(t *testing.T) {
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

func TestParseNoNewlines(t *testing.T) {
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

func TestParseReturn0(t *testing.T) {
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

func TestParseReturn2(t *testing.T) {
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

func TestParseSpaces(t *testing.T) {
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

func TestParseMissingParen(t *testing.T) {
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

func TestParseMissingRetval(t *testing.T) {
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

func TestParseNoBrace(t *testing.T) {
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

func TestParseNoSemicolon(t *testing.T) {
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

func TestParseNoSpace(t *testing.T) {
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

func TestParseWrongCase(t *testing.T) {
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
