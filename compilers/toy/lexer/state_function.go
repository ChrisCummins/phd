package lexer

import (
	"github.com/ChrisCummins/phd/compilers/toy/token"
	"strings"
	"unicode"
	"unicode/utf8"
)

type stateFunction func(*Lexer) stateFunction

const eofRune = rune(0)

func isIdentifierRune(r rune) bool {
	return unicode.IsDigit(r) || unicode.IsLetter(r) || r == '_'
}

func identifierLookAhead(lexer *Lexer, prefix string) bool {
	// No room to look-ahead.
	if len(lexer.input) <= lexer.position+len(prefix) {
		return false
	}

	// Check if next character is part of an identifier.
	r, _ := utf8.DecodeRuneInString(
		lexer.input[lexer.position+len(prefix):])
	if isIdentifierRune(r) {
		return true
	}

	return false
}

// The initial state function.
func lexStartState(lexer *Lexer) stateFunction {
	for {
		candidateToken := lexer.input[lexer.position:]

		// FIXME: This lookahead logic for int and return is overly convoluted.
		if strings.HasPrefix(candidateToken, "int") {
			if identifierLookAhead(lexer, "int") {
				return lexIdentifier
			}
			return emit(3, token.IntKeywordToken, lexStartState, lexer)
		} else if strings.HasPrefix(candidateToken, "return") {
			if identifierLookAhead(lexer, "return") {
				return lexIdentifier
			}
			return emit(len("return"), token.ReturnKeywordToken, lexStartState, lexer)
		} else if strings.HasPrefix(candidateToken, "&&") {
			return emit(2, token.AndToken, lexStartState, lexer)
		} else if strings.HasPrefix(candidateToken, "||") {
			return emit(2, token.OrToken, lexStartState, lexer)
		} else if strings.HasPrefix(candidateToken, "!=") {
			return emit(2, token.NotEqualToken, lexStartState, lexer)
		} else if strings.HasPrefix(candidateToken, "<=") {
			return emit(2, token.LessThanOrEqualToken, lexStartState, lexer)
		} else if strings.HasPrefix(candidateToken, ">=") {
			return emit(2, token.GreaterThanOrEqualToken, lexStartState, lexer)
		}

		if lexer.peek() == eofRune {
			return nil
		}

		// TODO: There's some confusion here. One of the switches looks at the
		// current character, the other looks at next(). Remove one of these.
		switch r, _ := utf8.DecodeRuneInString(lexer.input[lexer.position:]); {
		case r == '{':
			return emit(1, token.OpenBraceToken, lexStartState, lexer)
		case r == '}':
			return emit(1, token.CloseBraceToken, lexStartState, lexer)
		case r == '(':
			return emit(1, token.OpenParenthesisToken, lexStartState, lexer)
		case r == ')':
			return emit(1, token.CloseParenthesisToken, lexStartState, lexer)
		case r == ';':
			return emit(1, token.SemicolonToken, lexStartState, lexer)
		case r == '!':
			return emit(1, token.LogicalNegationToken, lexStartState, lexer)
		case r == '~':
			return emit(1, token.BitwiseComplementToken, lexStartState, lexer)
		case r == '-':
			return emit(1, token.NegationToken, lexStartState, lexer)
		case r == '+':
			return emit(1, token.AdditionToken, lexStartState, lexer)
		case r == '*':
			return emit(1, token.MultiplicationToken, lexStartState, lexer)
		case r == '/':
			return emit(1, token.DivisionToken, lexStartState, lexer)
		case r == '<':
			return emit(1, token.LessThanToken, lexStartState, lexer)
		case r == '>':
			return emit(1, token.GreaterThanToken, lexStartState, lexer)
		}

		switch r := lexer.next(); {
		case unicode.IsSpace(r) || r == '\n':
			lexer.ignore()
		case unicode.IsDigit(r):
			lexer.Backup()
			return lexNumber
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_':
			lexer.Backup()
			return lexIdentifier
		default:
			return lexer.errorf("illegal character: `%v`", string(r))
		}
	}
	panic("unreachable!")
}

func lexNumber(lexer *Lexer) stateFunction {
	digits := "0123456789"
	// Is it hex or octal?
	if lexer.accept("0") && lexer.accept("xX") {
		digits = "0123456789abcdefABCDEF"
	} else if lexer.accept("0") {
		digits = "01234567"
	}
	lexer.acceptRun(digits)
	if lexer.accept(".") {
		lexer.acceptRun(digits)
	}
	if lexer.accept("eE") {
		lexer.accept("+-")
		lexer.acceptRun("0123456789")
	}

	if unicode.IsDigit(lexer.peek()) || unicode.IsLetter(lexer.peek()) {
		lexer.next()
		return lexer.errorf("Bad number syntax: %q",
			lexer.input[lexer.startPosition:lexer.position])
	}

	lexer.emit(token.NumberToken)
	return lexStartState
}

func lexIdentifier(lexer *Lexer) stateFunction {
	for {
		r := lexer.peek()
		if r == eofRune {
			return lexer.errorf("Unterminated identifier")
		}
		if !(unicode.IsDigit(r) || unicode.IsLetter(r) || r == '_') {
			break
		}
		lexer.next()
	}

	r := lexer.peek()
	if !(unicode.IsSpace(r) || r == '\n' || r == '(' || r == ')' ||
		r == ';') {
		lexer.next()
		return lexer.errorf("Bad identifier: %v",
			lexer.input[lexer.startPosition:lexer.position])
	}

	lexer.emit(token.IdentifierToken)
	return lexStartState
}

func emit(len int, t token.TokenType, nextState stateFunction,
	lexer *Lexer) func(lexer *Lexer) stateFunction {
	return func(lexer *Lexer) stateFunction {
		lexer.position += len
		lexer.emit(t)
		return nextState
	}
}
