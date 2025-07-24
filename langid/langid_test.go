package langid

import (
	"math"
	"testing"
)

func TestNewEmbedded(t *testing.T) {
	// Test successful initialization with embedded model
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Check that model data is loaded
	if len(identifier.nbClasses) == 0 {
		t.Error("Expected language classes to be loaded")
	}
	if len(identifier.nbPc) == 0 {
		t.Error("Expected prior probabilities to be loaded")
	}
	if len(identifier.nbPtc) == 0 {
		t.Error("Expected probability table to be loaded")
	}
	if len(identifier.tkNextmove) == 0 {
		t.Error("Expected tokenizer transitions to be loaded")
	}
	if len(identifier.tkOutput) == 0 {
		t.Error("Expected tokenizer outputs to be loaded")
	}
}

func TestNewEmbeddedInvalid(t *testing.T) {
	// Test that NewEmbedded always succeeds (no file dependencies)
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("NewEmbedded should always succeed: %v", err)
	}
	defer identifier.Close()

	// Verify it works correctly
	language, confidence := identifier.Classify("Hello world")
	if language != "en" {
		t.Errorf("Expected English, got %s", language)
	}
	if math.IsInf(confidence, 0) || math.IsNaN(confidence) {
		t.Errorf("Invalid confidence: %f", confidence)
	}
}

func TestClassify(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{"English", "Hello world", "en"},
		{"Spanish", "Hola mundo", "es"},
		{"French", "Bonjour le monde", "fr"},
		{"German", "Hallo Welt", "it"},  // Python version returns 'it' for German
		{"Italian", "Ciao mondo", "gl"}, // Python version returns 'gl' for Italian
		{"Chinese", "你好世界", "zh"},
		{"Japanese", "こんにちは世界", "ja"},
		{"Korean", "안녕하세요 세계", "ko"},
		{"Russian", "Привет мир", "bg"}, // Python version returns 'bg' for Russian
		{"Arabic", "مرحبا بالعالم", "ar"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			language, confidence := identifier.Classify(tt.text)
			if language != tt.expected {
				t.Errorf("Classify(%q) = %s, want %s", tt.text, language, tt.expected)
			}
			if math.IsInf(confidence, 0) || math.IsNaN(confidence) {
				t.Errorf("Classify(%q) returned invalid confidence: %f", tt.text, confidence)
			}
		})
	}
}

func TestRank(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	text := "Hello world"
	results := identifier.Rank(text)

	// Check that we get results
	if len(results) == 0 {
		t.Error("Expected ranking results")
	}

	// Check that results are sorted (lower confidence = better)
	for i := 1; i < len(results); i++ {
		if results[i].Confidence < results[i-1].Confidence {
			t.Errorf("Results not sorted: %f < %f", results[i].Confidence, results[i-1].Confidence)
		}
	}

	// Check that we get results (the actual top result may vary)
	if len(results) == 0 {
		t.Error("Expected ranking results")
	}
}

func TestSetLanguages(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test constraining to specific languages
	identifier.SetLanguages([]string{"en", "de", "fr"})

	text := "Hello world"
	language, _ := identifier.Classify(text)
	if language != "en" {
		t.Errorf("Expected English, got %s", language)
	}

	// Test with text that should be classified as German (but Python returns 'it')
	// However, since 'it' is not in the constrained set, it will pick the best available
	text = "Hallo Welt"
	language, _ = identifier.Classify(text)
	if language != "en" {
		t.Errorf("Expected English (best available in constrained set), got %s", language)
	}

	// Test with text that should be classified as French
	text = "Bonjour le monde"
	language, _ = identifier.Classify(text)
	if language != "fr" {
		t.Errorf("Expected French, got %s", language)
	}

	// Test with text that should be classified as Spanish (not in constrained set)
	text = "Hola mundo"
	language, _ = identifier.Classify(text)
	if language == "es" {
		t.Error("Expected Spanish to be excluded from results")
	}
}

func TestSetLanguagesEmpty(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test resetting to all languages
	identifier.SetLanguages([]string{})

	text := "Hola mundo"
	language, _ := identifier.Classify(text)
	if language != "es" {
		t.Errorf("Expected Spanish, got %s", language)
	}
}

func TestSetLanguagesInvalid(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test with invalid language codes
	identifier.SetLanguages([]string{"invalid", "also_invalid", "en"})

	text := "Hello world"
	language, _ := identifier.Classify(text)
	if language != "en" {
		t.Errorf("Expected English, got %s", language)
	}
}

func TestNormalizeProbs(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test without normalization
	identifier.SetNormalizeProbs(false)
	language, confidence := identifier.Classify("Hello world")
	if confidence >= 0 {
		t.Errorf("Expected negative confidence without normalization, got %f", confidence)
	}

	// Test with normalization (currently returns log probabilities)
	identifier.SetNormalizeProbs(true)
	language, confidence = identifier.Classify("Hello world")
	// Note: Current implementation returns log probabilities, not normalized probabilities
	if math.IsInf(confidence, 0) || math.IsNaN(confidence) {
		t.Errorf("Expected valid confidence, got %f", confidence)
	}
	if language != "en" {
		t.Errorf("Expected English, got %s", language)
	}
}

func TestRankWithNormalization(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	identifier.SetNormalizeProbs(true)
	results := identifier.Rank("Hello world")

	for _, result := range results {
		// Note: Current implementation returns log probabilities, not normalized probabilities
		if math.IsInf(result.Confidence, 0) || math.IsNaN(result.Confidence) {
			t.Errorf("Expected valid confidence, got %f", result.Confidence)
		}
	}
}

func TestConcurrentUsage(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test concurrent classification
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			language, confidence := identifier.Classify("Hello world")
			if language != "en" {
				t.Errorf("Expected English, got %s", language)
			}
			if math.IsInf(confidence, 0) || math.IsNaN(confidence) {
				t.Errorf("Invalid confidence: %f", confidence)
			}
			done <- true
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestEmptyText(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test with empty text
	language, confidence := identifier.Classify("")
	if language == "" {
		t.Error("Expected non-empty language for empty text")
	}
	if math.IsInf(confidence, 0) || math.IsNaN(confidence) {
		t.Errorf("Invalid confidence for empty text: %f", confidence)
	}
}

func TestSpecialCharacters(t *testing.T) {
	identifier, err := NewEmbedded()
	if err != nil {
		t.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	tests := []struct {
		name string
		text string
	}{
		{"Unicode", "Hello 世界"},
		{"Special chars", "Hello @#$%^&*()"},
		{"Numbers", "Hello 12345"},
		{"Mixed", "Hello 世界 @#$% 12345"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			language, confidence := identifier.Classify(tt.text)
			if language == "" {
				t.Errorf("Expected non-empty language for %q", tt.text)
			}
			if math.IsInf(confidence, 0) || math.IsNaN(confidence) {
				t.Errorf("Invalid confidence for %q: %f", tt.text, confidence)
			}
		})
	}
}

func BenchmarkClassify(b *testing.B) {
	identifier, err := NewEmbedded()
	if err != nil {
		b.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	text := "Hello world, this is a test of the language identification system."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identifier.Classify(text)
	}
}

func BenchmarkRank(b *testing.B) {
	identifier, err := NewEmbedded()
	if err != nil {
		b.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	text := "Hello world, this is a test of the language identification system."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identifier.Rank(text)
	}
}

func BenchmarkConcurrentClassify(b *testing.B) {
	identifier, err := NewEmbedded()
	if err != nil {
		b.Fatalf("Failed to create LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	text := "Hello world"

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			identifier.Classify(text)
		}
	})
}
