package token

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTokenStream(t *testing.T) {
	assert := assert.New(t)

	tokens := []Token{
		Token{IntKeywordToken, "int"},
		Token{IdentifierToken, "main"},
		Token{OpenParenthesisToken, "("},
		Token{CloseParenthesisToken, ")"},
		Token{OpenBraceToken, "{"},
		Token{ReturnKeywordToken, "return"},
		Token{NumberToken, "100"},
		Token{SemicolonToken, ";"},
		Token{CloseBraceToken, "}"},
	}

	ts := NewTokenStream(tokens)
	assert.True(ts.Next())
	assert.Equal(Token{IntKeywordToken, "int"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{IdentifierToken, "main"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{OpenParenthesisToken, "("}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{OpenBraceToken, "{"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{NumberToken, "100"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{SemicolonToken, ";"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(Token{CloseBraceToken, "}"}, ts.Value())
	assert.False(ts.Next())
}
