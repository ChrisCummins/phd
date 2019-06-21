package lexer

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
)

type LexerTokenStream struct {
	lex   *Lexer
	value token.Token
}

func NewLexerTokenStream(lex *Lexer) *LexerTokenStream {
	return &LexerTokenStream{lex: lex}
}

func (ts *LexerTokenStream) Next() bool {
	ts.value = ts.lex.NextToken()
	switch ts.value.Type {
	case token.ErrorToken:
		return false
	case token.EofToken:
		return false
	}

	return true
}

func (ts *LexerTokenStream) Value() token.Token {
	return ts.value
}
