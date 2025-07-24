package tests

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/ozim-ai/langid-go/langid"
)

func TestPythonCompatibility(t *testing.T) {
	// Test cases that should produce identical results
	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{"English", "Hello world", "en"},
		{"Spanish", "Hola mundo", "es"},
		{"French", "Bonjour le monde", "fr"},
		{"German", "Hallo Welt", "de"},
		{"Italian", "Ciao mondo", "it"},
		{"Chinese", "你好世界", "zh"},
		{"Japanese", "こんにちは世界", "ja"},
		{"Korean", "안녕하세요 세계", "ko"},
		{"Russian", "Привет мир", "ru"},
		{"Arabic", "مرحبا بالعالم", "ar"},
		{"Portuguese", "Olá mundo", "pt"},
		{"Dutch", "Hallo wereld", "nl"},
		{"Swedish", "Hej världen", "sv"},
		{"Norwegian", "Hei verden", "no"},
		{"Danish", "Hej verden", "da"},
		{"Finnish", "Hei maailma", "fi"},
		{"Polish", "Witaj świecie", "pl"},
		{"Czech", "Ahoj světe", "cs"},
		{"Hungarian", "Helló világ", "hu"},
		{"Romanian", "Salut lume", "ro"},
		{
			name:     "Chinese greeting",
			text:     "你好吗？",
			expected: "zh",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test Go implementation
			identifier, err := langid.New("../data")
			if err != nil {
				t.Fatalf("Failed to create Go LanguageIdentifier: %v", err)
			}
			defer identifier.Close()

			goLang, goConf := identifier.Classify(tt.text)

			// Test Python implementation
			pythonLang, pythonConf := runPythonClassify(tt.text)
			if pythonLang == "" {
				t.Skipf("Python test skipped for %s (Python not available)", tt.name)
			}

			// Compare languages
			if goLang != pythonLang {
				t.Errorf("Language mismatch for %q: Go=%s, Python=%s", tt.text, goLang, pythonLang)
			}

			// Compare confidence scores (allow small differences due to floating point precision)
			diff := abs(goConf - pythonConf)
			if diff > 0.01 {
				t.Errorf("Confidence mismatch for %q: Go=%.6f, Python=%.6f, diff=%.6f",
					tt.text, goConf, pythonConf, diff)
			}

			// Verify expected language
			if goLang != tt.expected {
				t.Errorf("Expected %s for %q, got %s", tt.expected, tt.text, goLang)
			}
		})
	}
}

func TestPythonRankCompatibility(t *testing.T) {
	text := "Hello world"

	// Get Go ranking
	identifier, err := langid.New("../data")
	if err != nil {
		t.Fatalf("Failed to create Go LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	goResults := identifier.Rank(text)

	// Get Python ranking
	pythonResults := runPythonRank(text)
	if len(pythonResults) == 0 {
		t.Skip("Python test skipped (Python not available)")
	}

	// Compare top 5 results
	compareCount := min(5, len(goResults), len(pythonResults))
	for i := 0; i < compareCount; i++ {
		if goResults[i].Language != pythonResults[i].Language {
			t.Errorf("Rank %d mismatch: Go=%s, Python=%s", i,
				goResults[i].Language, pythonResults[i].Language)
		}
	}
}

func TestPythonConstrainedLanguages(t *testing.T) {
	// Test with language constraints
	identifier, err := langid.New("../data")
	if err != nil {
		t.Fatalf("Failed to create Go LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Constrain to specific languages
	identifier.SetLanguages([]string{"en", "de", "fr"})

	text := "Hello world"
	goLang, goConf := identifier.Classify(text)

	// Test Python with same constraints
	pythonLang, pythonConf := runPythonClassifyWithLanguages(text, []string{"en", "de", "fr"})
	if pythonLang == "" {
		t.Skip("Python test skipped (Python not available)")
	}

	// Compare results
	if goLang != pythonLang {
		t.Errorf("Constrained language mismatch: Go=%s, Python=%s", goLang, pythonLang)
	}

	diff := abs(goConf - pythonConf)
	if diff > 0.01 {
		t.Errorf("Constrained confidence mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
			goConf, pythonConf, diff)
	}
}

func TestPythonNormalization(t *testing.T) {
	identifier, err := langid.New("../data")
	if err != nil {
		t.Fatalf("Failed to create Go LanguageIdentifier: %v", err)
	}
	defer identifier.Close()

	// Test with normalization enabled
	identifier.SetNormalizeProbs(true)

	text := "Hello world"
	goLang, goConf := identifier.Classify(text)

	// Test Python with normalization
	pythonLang, pythonConf := runPythonClassifyNormalized(text)
	if pythonLang == "" {
		t.Skip("Python test skipped (Python not available)")
	}

	// Compare results
	if goLang != pythonLang {
		t.Errorf("Normalized language mismatch: Go=%s, Python=%s", goLang, pythonLang)
	}

	// Normalized probabilities should be between 0 and 1
	if goConf < 0 || goConf > 1 {
		t.Errorf("Go normalized confidence out of range: %f", goConf)
	}
	if pythonConf < 0 || pythonConf > 1 {
		t.Errorf("Python normalized confidence out of range: %f", pythonConf)
	}

	diff := abs(goConf - pythonConf)
	if diff > 0.01 {
		t.Errorf("Normalized confidence mismatch: Go=%.6f, Python=%.6f, diff=%.6f",
			goConf, pythonConf, diff)
	}
}

// Helper functions to run Python tests

func runPythonClassify(text string) (string, float64) {
	cmd := exec.Command("python", "-c",
		fmt.Sprintf("import langid; result = langid.classify('%s'); print('%%s %%.6f' %% result)", text))

	output, err := cmd.Output()
	if err != nil {
		return "", 0
	}

	parts := strings.Fields(string(output))
	if len(parts) != 2 {
		return "", 0
	}

	lang := parts[0]
	conf, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return "", 0
	}

	return lang, conf
}

func runPythonRank(text string) []struct {
	Language   string
	Confidence float64
} {
	cmd := exec.Command("python", "-c",
		fmt.Sprintf("import langid; result = langid.rank('%s'); print('\\\\n'.join(['%%s %%.6f' %% (r[0], r[1]) for r in result[:5]]))", text))

	output, err := cmd.Output()
	if err != nil {
		return nil
	}

	var results []struct {
		Language   string
		Confidence float64
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")

	for _, line := range lines {
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}

		conf, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			continue
		}

		results = append(results, struct {
			Language   string
			Confidence float64
		}{
			Language:   parts[0],
			Confidence: conf,
		})
	}

	return results
}

func runPythonClassifyWithLanguages(text string, languages []string) (string, float64) {
	langStr := strings.Join(languages, "','")
	cmd := exec.Command("python", "-c",
		fmt.Sprintf("import langid; langid.set_languages(['%s']); result = langid.classify('%s'); print('%%s %%.6f' %% result)", langStr, text))

	output, err := cmd.Output()
	if err != nil {
		return "", 0
	}

	parts := strings.Fields(string(output))
	if len(parts) != 2 {
		return "", 0
	}

	lang := parts[0]
	conf, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return "", 0
	}

	return lang, conf
}

func runPythonClassifyNormalized(text string) (string, float64) {
	cmd := exec.Command("python", "-c",
		fmt.Sprintf("from langid.langid import LanguageIdentifier; identifier = LanguageIdentifier.from_binary_files('data', norm_probs=True); result = identifier.classify('%s'); print('%%s %%.6f' %% result)", text))

	output, err := cmd.Output()
	if err != nil {
		return "", 0
	}

	parts := strings.Fields(string(output))
	if len(parts) != 2 {
		return "", 0
	}

	lang := parts[0]
	conf, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return "", 0
	}

	return lang, conf
}

// Utility functions

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b, c int) int {
	if a < b && a < c {
		return a
	}
	if b < c {
		return b
	}
	return c
}
