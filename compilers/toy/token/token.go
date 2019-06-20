package token

import "fmt"

// The type of a token.
type TokenType uint8

// A token type.
type Token struct {
	Type  TokenType
	Value string
}

// A list of token types.
const (
	ErrorToken TokenType = iota
	EofToken
	IdentifierToken
	NumberToken
	// Punctuation.
	OpenBraceToken        // {
	CloseBraceToken       // }
	OpenParanethesisToken // (
	CloseParenthesisToken // )
	SemicolonToken        // ;
	// Keywords.
	IntKeywordToken    // int
	ReturnKeywordToken // return
)

// String returns a stringified representation of a token.
func (t Token) String() string {
	switch t.Type {
	case ErrorToken:
		return t.Value
	case EofToken:
		return "EOF"
	}

	// In general case, truncate string representation.
	if len(t.Value) > 10 {
		return fmt.Sprintf("%.10q...", t.Value)
	}
	return fmt.Sprintf("%q", t.Value)
}
