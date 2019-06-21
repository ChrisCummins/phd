package main

import (
	"flag"
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/lexer"
	"github.com/ChrisCummins/phd/compilers/toy/parser"
	"io/ioutil"
	"os"
)

var FLAGS_i = flag.String("i", "", "Path to input file.")
var FLAGS_o = flag.String("o", "", "Path to output file.")

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func main() {
	flag.Parse()
	inputPath := FLAGS_i
	outputPath := FLAGS_o

	if *inputPath == "" {
		// TODO: Check for `.c` suffix.
		fmt.Fprintf(os.Stderr, "Input -i not provided")
		os.Exit(1)
	}
	if *outputPath == "" {
		// TODO: Get basename of source file and append suffix.
		*outputPath = *inputPath + `.s`
	}

	data, err := ioutil.ReadFile(*inputPath)
	check(err)

	ts := lexer.NewLexerTokenStream(lexer.Lex(string(data)))

	ast, err := parser.Parse(ts)
	check(err)

	as, err := ast.GenerateAssembly()
	check(err)

	err = ioutil.WriteFile(*outputPath, []byte(as), 0644)
	check(err)
}
