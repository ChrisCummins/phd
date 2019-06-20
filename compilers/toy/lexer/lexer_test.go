package lexer

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/valid

func TestLexMultiDigit(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 100;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "100"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexNewlines(t *testing.T) {
	assert := assert.New(t)
	input := `
int 
main
(   
)
{
return
0
;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexNoNewlines(t *testing.T) {
	assert := assert.New(t)
	input := `int main(){return 0;}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexReturn0(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 0;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexReturn2(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 2;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "2"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexSpaces(t *testing.T) {
	assert := assert.New(t)
	input := `   int   main    (  )  {   return  0 ; }`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/invalid

func TestLexMissingParen(t *testing.T) {
	assert := assert.New(t)
	input := `int main( {
    return 0;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexMissingRetval(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexNoBrace(t *testing.T) {
	assert := assert.New(t)
	input := `int main {
    return 0;
`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexNoSemicolon(t *testing.T) {
	assert := assert.New(t)
	input := `int main {
    return 0
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{ReturnKeywordToken, "return"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexNoSpace(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return0;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{IdentifierToken, "return0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}

func TestLexWrongCase(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    RETURN 0;
}`
	next := Lex(input).NextToken
	assert.Equal(Token{IntKeywordToken, "int"}, next())
	assert.Equal(Token{IdentifierToken, "main"}, next())
	assert.Equal(Token{OpenParanethesisToken, "("}, next())
	assert.Equal(Token{CloseParenthesisToken, ")"}, next())
	assert.Equal(Token{OpenBraceToken, "{"}, next())
	assert.Equal(Token{IdentifierToken, "RETURN"}, next())
	assert.Equal(Token{NumberToken, "0"}, next())
	assert.Equal(Token{SemicolonToken, ";"}, next())
	assert.Equal(Token{CloseBraceToken, "}"}, next())
	assert.Equal(EofToken, next().Type)
}
