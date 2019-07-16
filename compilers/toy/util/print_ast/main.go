// Pretty-print the AST to stdout.
//     Usage: $0 <source_file>
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/ChrisCummins/phd/compilers/toy/lexer"
	"github.com/ChrisCummins/phd/compilers/toy/parser"
	"github.com/golang/glog"
)

func check(e error) {
	if e != nil {
		fmt.Fprintf(os.Stderr, "fatal: %s\n", e)
		os.Exit(1)
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

	data, err := ioutil.ReadFile(inputPath)
	check(err)

	ts := lexer.NewLexerTokenStream(lexer.Lex(string(data)))

	ast, err := parser.Parse(ts)
	check(err)

	fmt.Println(ast)

	glog.Flush()
}
