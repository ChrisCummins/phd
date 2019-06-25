package lexer

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
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
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "100"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
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
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNoNewlines(t *testing.T) {
	assert := assert.New(t)
	input := `int main(){return 0;}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexReturn0(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexReturn2(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 2;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "2"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexSpaces(t *testing.T) {
	assert := assert.New(t)
	input := `   int   main    (  )  {   return  0 ; }`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_1/invalid

func TestLexMissingParen(t *testing.T) {
	assert := assert.New(t)
	input := `int main( {
    return 0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexMissingRetval(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNoBrace(t *testing.T) {
	assert := assert.New(t)
	input := `int main {
    return 0;
`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNoSemicolon(t *testing.T) {
	assert := assert.New(t)
	input := `int main {
    return 0
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNoSpace(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "return0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexWrongCase(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    RETURN 0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "RETURN"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_2/valid

func TestLexBitwise(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !12;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.NumberToken, "12"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexBitwiseZero(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return ~0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.BitwiseComplementToken, "~"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNeg(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return -5;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NegationToken, "-"}, next())
	assert.Equal(token.Token{token.NumberToken, "5"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNestedOps(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !-3;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.NegationToken, "-"}, next())
	assert.Equal(token.Token{token.NumberToken, "3"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNestedOps2(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return -~0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NegationToken, "-"}, next())
	assert.Equal(token.Token{token.BitwiseComplementToken, "~"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNotFive(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !5;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.NumberToken, "5"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNotZero(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !0;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.NumberToken, "0"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_2/invalid

func TestLexMissingConst(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexMissingSemicolon(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !5
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.NumberToken, "5"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexNestedMissingConst(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return !~;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.LogicalNegationToken, "!"}, next())
	assert.Equal(token.Token{token.BitwiseComplementToken, "~"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

func TestLexWrongOrder(t *testing.T) {
	assert := assert.New(t)
	input := `int main() {
    return 4-;
}`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "4"}, next())
	assert.Equal(token.Token{token.NegationToken, "-"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}

// Test inputs from github.com/nlsandler/write_a_c_compiler/stage_3/valid

func TestLexAssociativity(t *testing.T) {
	assert := assert.New(t)
	input := `int main() { return 1 - 2 - 3; }`
	next := Lex(input).NextToken
	assert.Equal(token.Token{token.IntKeywordToken, "int"}, next())
	assert.Equal(token.Token{token.IdentifierToken, "main"}, next())
	assert.Equal(token.Token{token.OpenParenthesisToken, "("}, next())
	assert.Equal(token.Token{token.CloseParenthesisToken, ")"}, next())
	assert.Equal(token.Token{token.OpenBraceToken, "{"}, next())
	assert.Equal(token.Token{token.ReturnKeywordToken, "return"}, next())
	assert.Equal(token.Token{token.NumberToken, "1"}, next())
	assert.Equal(token.Token{token.NegationToken, "-"}, next())
	assert.Equal(token.Token{token.NumberToken, "2"}, next())
	assert.Equal(token.Token{token.NegationToken, "-"}, next())
	assert.Equal(token.Token{token.NumberToken, "3"}, next())
	assert.Equal(token.Token{token.SemicolonToken, ";"}, next())
	assert.Equal(token.Token{token.CloseBraceToken, "}"}, next())
	assert.Equal(token.EofToken, next().Type)
}
