package main

import (
	"fmt"
	"log"

	"github.com/ozim-ai/langid-go/langid"
)

func main() {
	// Test model loading
	identifier, err := langid.New("data")
	if err != nil {
		log.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test simple classification
	text := "Hello world"
	language, confidence := identifier.Classify(text)
	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Language: %s\n", language)
	fmt.Printf("Confidence: %f\n", confidence)

	// Test with constrained languages
	identifier.SetLanguages([]string{"en", "de", "fr"})
	language, confidence = identifier.Classify(text)
	fmt.Printf("Constrained Language: %s\n", language)
	fmt.Printf("Constrained Confidence: %f\n", confidence)
}
