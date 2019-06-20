package lexer

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type Lexer struct {
	input         string
	startPosition int        // Start of current rune.
	position      int        // Current position in the input.
	width         int        // Width of the last rune read.
	tokens        chan Token // Channel of scanned tokens.
	state         stateFunction
}

// Emit a token back to the client.
func (lexer *Lexer) emit(t TokenType) {
	lexer.tokens <- Token{t, lexer.input[lexer.startPosition:lexer.position]}
	lexer.startPosition = lexer.position
}

// Report an error and exit.
func (lexer *Lexer) errorf(format string, args ...interface{}) stateFunction {
	// Set the text to the error message.
	lexer.tokens <- Token{
		ErrorToken,
		fmt.Sprintf(format, args...),
	}
	return nil // End the lexing loop.
}

func (lexer *Lexer) run() {
	for state := lexStartState; state != nil; {
		state = state(lexer)
	}
	// No more tokens will be delivered.
	close(lexer.tokens)
}

func (lexer *Lexer) next() rune {
	if lexer.position >= len(lexer.input) {
		lexer.width = 0
		return eofRune
	}
	r, width := utf8.DecodeRuneInString(lexer.input[lexer.position:])
	lexer.width = width
	lexer.position += lexer.width
	return r
}

func (lexer *Lexer) ignore() {
	lexer.startPosition = lexer.position
}

func (lexer *Lexer) backup() {
	lexer.position -= lexer.width
}

func (lexer *Lexer) peek() rune {
	rune := lexer.next()
	lexer.backup()
	return rune
}

// accept consumes the next rune if it is from the valid set.
func (lexer *Lexer) accept(valid string) bool {
	if strings.IndexRune(valid, lexer.next()) >= 0 {
		return true
	}
	lexer.backup()
	return false
}

// acceptRun consumes a run of runes from the valid set.
func (lexer *Lexer) acceptRun(valid string) {
	for strings.IndexRune(valid, lexer.next()) >= 0 {

	}
	lexer.backup()
}

func (lexer *Lexer) NextToken() Token {
	for {
		select {
		case token := <-lexer.tokens:
			return token
		default:
			if lexer.state == nil {
				return Token{EofToken, ""}
			}
			lexer.state = lexer.state(lexer)
		}
	}
	panic("unreachable!")
}

func Lex(input string) *Lexer {
	return &Lexer{
		input:  input,
		state:  lexStartState,
		tokens: make(chan Token, 2), // Two items sufficient.
	}
}
