package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ozim-ai/langid-go/langid"
)

func main() {
	// Parse command line flags
	var (
		languages = flag.String("l", "", "comma-separated set of target ISO639 language codes (e.g en,de)")
		normalize = flag.Bool("n", false, "normalize confidence scores to probability values")
		verbose   = flag.Bool("v", false, "increase verbosity")
		help      = flag.Bool("h", false, "show this help message and exit")
	)
	flag.Parse()

	if *help {
		printUsage()
		return
	}

	// Initialize language identifier with embedded model
	identifier, err := langid.NewEmbedded()
	if err != nil {
		log.Fatalf("Failed to initialize language identifier: %v", err)
	}
	defer identifier.Close()

	// Set normalization if requested
	if *normalize {
		identifier.SetNormalizeProbs(true)
	}

	// Set language constraints if specified
	if *languages != "" {
		langList := strings.Split(*languages, ",")
		for i, lang := range langList {
			langList[i] = strings.TrimSpace(lang)
		}
		identifier.SetLanguages(langList)
	}

	// Check if input is from pipe or file
	stat, err := os.Stdin.Stat()
	if err != nil {
		log.Fatalf("Failed to check stdin: %v", err)
	}

	if stat.Mode()&os.ModeCharDevice == 0 {
		// Input is from pipe or file
		processPipedInput(identifier, *verbose)
	} else {
		// Interactive mode
		processInteractiveInput(identifier, *verbose)
	}
}

func processPipedInput(identifier *langid.LanguageIdentifier, verbose bool) {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}

		language, confidence := identifier.Classify(text)
		if verbose {
			fmt.Printf("%s\t%.6f\n", language, confidence)
		} else {
			fmt.Printf("%s %.6f\n", language, confidence)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading input: %v", err)
	}
}

func processInteractiveInput(identifier *langid.LanguageIdentifier, verbose bool) {
	fmt.Println("Language Identifier (Go version - Embedded Model)")
	fmt.Println("Enter text to classify (Ctrl+C to exit):")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			break
		}

		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}

		language, confidence := identifier.Classify(text)
		if verbose {
			fmt.Printf("('%s', %.6f)\n", language, confidence)
		} else {
			fmt.Printf("('%s', %.6f)\n", language, confidence)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading input: %v", err)
	}
}

func printUsage() {
	fmt.Println("Usage: langid [options]")
	fmt.Println()
	fmt.Println("Options:")
	fmt.Println("  -h, --help            show this help message and exit")
	fmt.Println("  -l LANGS, --langs=LANGS")
	fmt.Println("                        comma-separated set of target ISO639 language codes")
	fmt.Println("                        (e.g en,de)")
	fmt.Println("  -n, --normalize       normalize confidence scores to probability values")
	fmt.Println("  -v                    increase verbosity (repeat for greater effect)")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  echo 'Hello world' | ./langid")
	fmt.Println("  ./langid < README.md")
	fmt.Println("  echo 'Hello world' | ./langid -l en,de,fr")
	fmt.Println("  echo 'Hello world' | ./langid -n")
}
