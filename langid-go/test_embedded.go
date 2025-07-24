package main

import (
	"fmt"
	"log"

	"github.com/ozim-ai/langid-go/langid"
)

func main() {
	// Test embedded model
	fmt.Println("Testing embedded model...")

	identifier, err := langid.NewEmbedded()
	if err != nil {
		log.Fatalf("Failed to create embedded LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test some classifications
	tests := []struct {
		text     string
		expected string
	}{
		{"Hello world", "en"},
		{"你好吗？", "zh"},
		{"Bonjour le monde", "fr"},
		{"Hola mundo", "es"},
		{"こんにちは世界", "ja"},
		{"안녕하세요 세계", "ko"},
		{"مرحبا بالعالم", "ar"},
	}

	for _, test := range tests {
		language, confidence := identifier.Classify(test.text)
		fmt.Printf("Text: %s\n", test.text)
		fmt.Printf("  Detected: %s (%.6f)\n", language, confidence)
		fmt.Printf("  Expected: %s\n", test.expected)
		fmt.Printf("  Match: %t\n", language == test.expected)
		fmt.Println()
	}
}
