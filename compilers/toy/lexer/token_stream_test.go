package lexer

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLexTokenStreamMultiDigit(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 100;
}`
	ts := NewLexerTokenStream(Lex(input))
	assert.True(ts.Next())
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.NumberToken, "100"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, ts.Value())
	assert.True(ts.Next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, ts.Value())
	assert.False(ts.Next())
}
