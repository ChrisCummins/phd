package token

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTokenString(t *testing.T) {
	assert := assert.New(t)
	assert.Equal(`EOF`, Token{EofToken, ""}.String())
	assert.Equal(`"abc"`, Token{IdentifierToken, "abc"}.String())
	assert.Equal(`"0123456789"...`,
		Token{IdentifierToken, "01234567890123456789"}.String())
}
