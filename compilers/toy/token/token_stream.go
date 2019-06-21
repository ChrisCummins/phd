package token

type TokenStream struct {
	tokens   []Token
	position int
}

func NewTokenStream(tokens []Token) *TokenStream {
	return &TokenStream{tokens: tokens}
}

func (i *TokenStream) Next() bool {
	i.position++
	return i.position <= len(i.tokens)
}

func (i *TokenStream) Value() Token {
	if i.position > len(i.tokens) {
		panic("iterator ended")
	}
	// We increment the position before we get the value, so i.position needs to
	// be negatively offset.
	return i.tokens[i.position-1]
}
