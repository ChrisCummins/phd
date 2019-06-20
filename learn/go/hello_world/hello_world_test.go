package main

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSayHello(t *testing.T) {
	assert.Equal(t, 0, SayHello(), "Woops!")
}

func TestCallLibrary(t *testing.T) {
	assert.Equal(t, int32(42), CallLibrary())
}
