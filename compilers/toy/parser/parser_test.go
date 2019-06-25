package parser

import (
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestParseFactorNumber(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.NumberToken, "100"},
	})
	p, err := ParseFactor(ts)

	assert.Nil(err)
	assert.NotNil(p)
	assert.Equal("Int<100>", p.String())
}

func TestParseFactorUnaryOp(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.NegationToken, "-"},
		token.Token{token.NumberToken, "100"},
	})
	p, err := ParseFactor(ts)

	assert.Nil(err)
	assert.NotNil(p)
	assert.Equal("- Int<100>", p.String())
}

func TestParseTermNumberMultiplication(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.NumberToken, "1"},
		token.Token{token.MultiplicationToken, "*"},
		token.Token{token.NumberToken, "2"},
	})

	p, err := ParseTerm(ts)

	assert.Nil(err)
	assert.NotNil(p)
	assert.Equal("Int<1> * Int<2>", p.String())
}

func TestParseExpressionNumericalAddition(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.NumberToken, "1"},
		token.Token{token.AdditionToken, "+"},
		token.Token{token.NumberToken, "2"},
	})

	p, err := ParseExpression(ts)

	assert.Nil(err)
	assert.NotNil(p)
	assert.Equal("Int<1> + Int<2>", p.String())
}

func TestParseStatementWithParenthesis(t *testing.T) {
	fmt.Println(">>>>>>>>>>>>>>>>>>>>>> TEST")
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.NumberToken, "1"},
		token.Token{token.AdditionToken, "+"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.NumberToken, "2"},
		token.Token{token.MultiplicationToken, "*"},
		token.Token{token.NumberToken, "3"},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.SemicolonToken, ";"},
	})

	p, err := ParseStatement(ts)

	assert.Nil(err)
	assert.NotNil(p)
	assert.Equal("RETURN Int<1> + Int<2> * Int<3>", p.String())
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestParseMultiDigit(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	assert.Equal("Int<100>", p.Function.Statement.Expression.String())
}

func TestParseNewlines(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	assert.Equal("Int<0>", p.Function.Statement.Expression.String())
}

func TestParseNoNewlines(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	assert.Equal("Int<0>", p.Function.Statement.Expression.String())
}

func TestParseReturn0(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	assert.Equal("Int<0>", p.Function.Statement.Expression.String())
}

func TestParseReturn2(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	assert.Equal("Int<2>", p.Function.Statement.Expression.String())
}

func TestParseSpaces(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	assert.Equal("Int<0>", p.Function.Statement.Expression.String())
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/invalid

func TestParseMissingParen(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
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
	ts := token.NewSliceTokenStream([]token.Token{
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
	ts := token.NewSliceTokenStream([]token.Token{
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
	ts := token.NewSliceTokenStream([]token.Token{
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
	ts := token.NewSliceTokenStream([]token.Token{
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
	ts := token.NewSliceTokenStream([]token.Token{
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

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_2/valid

func TestParseBitwise(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
		token.Token{token.NumberToken, "12"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal("! Int<12>", p.Function.Statement.Expression.String())
}

func TestParseBitwiseZero(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.BitwiseComplementToken, "~"},
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
	assert.Equal("~ Int<0>", p.Function.Statement.Expression.String())
}

func TestParseNeg(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NegationToken, "-"},
		token.Token{token.NumberToken, "5"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal("- Int<5>", p.Function.Statement.Expression.String())
}

func TestParseNestedOps(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
		token.Token{token.NegationToken, "-"},
		token.Token{token.NumberToken, "3"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal("! - Int<3>", p.Function.Statement.Expression.String())
}

func TestParseNestedOps2(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NegationToken, "-"},
		token.Token{token.BitwiseComplementToken, "~"},
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
	assert.Equal("- ~ Int<0>", p.Function.Statement.Expression.String())
}

func TestParseNotFive(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
		token.Token{token.NumberToken, "5"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)

	assert.Equal(nil, err)
	assert.NotEqual(nil, p.Function)
	assert.Equal("main", p.Function.Identifier)
	assert.NotEqual(nil, p.Function.Statement)
	assert.NotEqual(nil, p.Function.Statement.Expression)
	assert.Equal("! Int<5>", p.Function.Statement.Expression.String())
}

func TestParseNotZero(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
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
	assert.Equal("! Int<0>", p.Function.Statement.Expression.String())
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_2/invalid

func TestParseMissingConst(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestParseMissingSemicolon(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
		token.Token{token.NumberToken, "5"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestParseNestedMissingConst(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.LogicalNegationToken, "!"},
		token.Token{token.BitwiseComplementToken, "~"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}

func TestParseWrongOrder(t *testing.T) {
	assert := assert.New(t)
	ts := token.NewSliceTokenStream([]token.Token{
		token.Token{token.IntKeywordToken, "int"},
		token.Token{token.IdentifierToken, "main"},
		token.Token{token.OpenParenthesisToken, "("},
		token.Token{token.CloseParenthesisToken, ")"},
		token.Token{token.OpenBraceToken, "{"},
		token.Token{token.ReturnKeywordToken, "return"},
		token.Token{token.NumberToken, "4"},
		token.Token{token.NegationToken, "-"},
		token.Token{token.SemicolonToken, ";"},
		token.Token{token.CloseBraceToken, "}"},
	})
	p, err := Parse(ts)
	assert.NotNil(err)
	assert.Nil(p)
}
