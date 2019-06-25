package main

import (
	"flag"
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/lexer"
	"github.com/ChrisCummins/phd/compilers/toy/parser"
	"io/ioutil"
	"os"
	"strings"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func main() {
	flag.Parse()
	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "No input file provided.\n")
		os.Exit(1)
	} else if flag.NArg() > 1 {
		fmt.Fprintf(os.Stderr, "No support for multiple input files.\n")
		os.Exit(1)
	}
	inputPath := flag.Arg(0)

	if !strings.HasSuffix(inputPath, ".c") {
		fmt.Fprintf(os.Stderr, "Unrecognised file type\n")
		os.Exit(1)
	}

	outputPath := strings.TrimSuffix(inputPath, ".c") + ".s"

	data, err := ioutil.ReadFile(inputPath)
	check(err)

	ts := lexer.NewLexerTokenStream(lexer.Lex(string(data)))

	ast, err := parser.Parse(ts)
	check(err)

	as, err := ast.GenerateAssembly()
	check(err)

	err = ioutil.WriteFile(outputPath, []byte(as), 0644)
	check(err)
}