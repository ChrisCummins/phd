package lexer

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
)

type LexerTokenStream struct {
	lex     *Lexer
	prev    token.Token
	current token.Token
	next    token.Token
}

func NewLexerTokenStream(lex *Lexer) *LexerTokenStream {
	ts := &LexerTokenStream{lex: lex, current: token.Token{Type: token.EofToken}}
	ts.next = ts.lex.NextToken()
	return ts
}

func (ts *LexerTokenStream) Next() bool {
	ts.prev = ts.current
	ts.current = ts.next

	switch ts.current.Type {
	case token.ErrorToken:
		return false
	case token.EofToken:
		return false
	}

	ts.next = ts.lex.NextToken()
	return true
}

func (ts *LexerTokenStream) Value() token.Token {
	return ts.current
}

func (ts *LexerTokenStream) backup() {
	ts.next = ts.current
	ts.current = ts.prev
}

func (ts *LexerTokenStream) Peek() token.Token {
	return ts.next
}
