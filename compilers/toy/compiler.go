package main

import (
	"bytes"
	"flag"
	"fmt"
	"github.com/ChrisCummins/phd/compilers/toy/lexer"
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
		fmt.Fprintf(os.Stderr, "Input -i not provided")
		os.Exit(1)
	}
	if *outputPath == "" {
		*outputPath = *inputPath + `.s`
	}

	data, err := ioutil.ReadFile(*inputPath)
	check(err)

	output := bytes.Buffer{}
	l := lexer.Lex(string(data))

	stop := false
	for {
		if stop == true {
			break
		}
		token := l.NextToken()
		switch token.Type {
		case lexer.ErrorToken:
			fmt.Fprintf(os.Stderr, "error in lexer!: %v\n", token.Value)
			stop = true
		case lexer.EofToken:
			stop = true
		default:
			fmt.Println("Lexed:", token)
			output.WriteString(token.Value)
			output.WriteRune('\n')
		}
	}

	err = ioutil.WriteFile(*outputPath, []byte(output.String()), 0644)
	check(err)
}
