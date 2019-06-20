package main

import (
	"flag"
	"fmt"
	"github.com/ChrisCummins/phd/learn/go/first_library"
)

var FLAGS_name = flag.String(
	"name", "", "The name of the person to say hello to.")

func SayHello() int {
	if *FLAGS_name != "" {
		fmt.Printf("Hello, %v\n", *FLAGS_name)
	} else {
		fmt.Println("Hello, world!")
	}
	return 0
}

func CallLibrary() int32 {
	return first_library.ComputeTheAnswerToLifeTheUniverseAndEverything()
}

func main() {
	flag.Parse()
	SayHello()
}
