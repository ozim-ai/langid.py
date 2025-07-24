//go:generate go run ../generate.go

package langid

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// LanguageScore represents a language classification result
type LanguageScore struct {
	Language   string  // ISO 639-1 language code
	Confidence float64 // Confidence score (lower = higher confidence)
}

// LanguageIdentifier holds the model data and provides classification methods
type LanguageIdentifier struct {
	mu sync.RWMutex // For thread-safe operations

	// Model data
	nbPtc      [][]float32         // Probability table (7480×97 matrix)
	nbPc       []float32           // Prior probabilities (97 vector)
	nbClasses  []string            // Language codes (97 codes)
	tkNextmove []uint16            // Tokenizer state transitions (2,334,208 elements)
	tkOutput   map[uint32][]uint16 // Tokenizer output mappings (8,656 entries)

	// Configuration
	normProbs    bool     // Whether to normalize probabilities
	languages    []string // Constrained language set
	allLanguages []string // All available languages
}

// New creates a new LanguageIdentifier instance from files
func New(modelDir string) (*LanguageIdentifier, error) {
	li := &LanguageIdentifier{}

	if err := li.loadModel(modelDir); err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Initialize with all languages
	li.allLanguages = make([]string, len(li.nbClasses))
	copy(li.allLanguages, li.nbClasses)
	li.languages = li.allLanguages

	return li, nil
}

// NewEmbedded creates a new LanguageIdentifier instance using embedded model data
func NewEmbedded() (*LanguageIdentifier, error) {
	li := &LanguageIdentifier{}

	if err := li.loadEmbeddedModel(); err != nil {
		return nil, fmt.Errorf("failed to load embedded model: %w", err)
	}

	// Initialize with all languages
	li.allLanguages = make([]string, len(li.nbClasses))
	copy(li.allLanguages, li.nbClasses)
	li.languages = li.allLanguages

	return li, nil
}

// Close releases resources associated with the identifier
func (li *LanguageIdentifier) Close() error {
	// No cleanup needed for embedded model
	return nil
}

// SetNormalizeProbs sets whether to normalize probabilities
func (li *LanguageIdentifier) SetNormalizeProbs(normalize bool) {
	li.normProbs = normalize
}

// SetLanguages constrains the language set for classification
func (li *LanguageIdentifier) SetLanguages(languages []string) {
	li.languages = languages
}

// Classify identifies the most likely language for the given text
func (li *LanguageIdentifier) Classify(text string) (string, float64) {
	li.mu.RLock()
	defer li.mu.RUnlock()

	// Extract features
	features := li.instance2fv(text)

	// Calculate probabilities
	probs := li.nbClassprobs(features)

	// Find the best language
	bestLang := ""
	bestScore := float32(math.Inf(-1))

	for i, lang := range li.nbClasses {
		// Check if language is in constrained set
		if len(li.languages) > 0 {
			found := false
			for _, allowedLang := range li.languages {
				if allowedLang == lang {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		score := probs[i]
		if score > bestScore {
			bestScore = score
			bestLang = lang
		}
	}

	return bestLang, float64(bestScore)
}

// Rank returns a ranked list of languages with confidence scores
func (li *LanguageIdentifier) Rank(text string) []LanguageScore {
	li.mu.RLock()
	defer li.mu.RUnlock()

	// Extract features
	features := li.instance2fv(text)

	// Calculate probabilities
	probs := li.nbClassprobs(features)

	// Create results
	var results []LanguageScore
	for i, lang := range li.nbClasses {
		// Check if language is in constrained set
		if len(li.languages) > 0 {
			found := false
			for _, allowedLang := range li.languages {
				if allowedLang == lang {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		results = append(results, LanguageScore{
			Language:   lang,
			Confidence: float64(probs[i]),
		})
	}

	// Sort by confidence (lower = higher confidence)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Confidence < results[j].Confidence
	})

	return results
}

// instance2fv converts text to feature vector
func (li *LanguageIdentifier) instance2fv(text string) []float32 {
	// Convert text to UTF-8 bytes (matching Python behavior)
	textBytes := []byte(text)

	// Initialize feature vector
	features := make([]float32, 7480)

	// Tokenize text
	tokens := li.tokenize(textBytes)

	// Count tokens
	for _, token := range tokens {
		if token < 7480 {
			features[token]++
		}
	}

	return features
}

// nbClassprobs calculates class probabilities
func (li *LanguageIdentifier) nbClassprobs(features []float32) []float32 {
	numClasses := len(li.nbClasses)
	probs := make([]float32, numClasses)

	// Initialize with prior probabilities
	for i := range probs {
		probs[i] = li.nbPc[i]
	}

	// Compute matrix-vector dot product: features * nbPtc
	// This is equivalent to Python's np.dot(fv, self.nb_ptc)
	for j, feature := range features {
		if feature > 0 {
			for i := 0; i < numClasses; i++ {
				probs[i] += li.nbPtc[j][i] * feature
			}
		}
	}

	return probs
}

// tokenize tokenizes text using the DFA
func (li *LanguageIdentifier) tokenize(text []byte) []uint16 {
	var tokens []uint16
	state := uint32(0)

	for _, letter := range text {
		// Get next state
		nextState := li.tkNextmove[state*256+uint32(letter)]
		state = uint32(nextState)

		// Check for output
		if outputs, exists := li.tkOutput[state]; exists {
			tokens = append(tokens, outputs...)
		}
	}

	return tokens
}

// loadEmbeddedModel loads the model from embedded data
func (li *LanguageIdentifier) loadEmbeddedModel() error {
	// Load nb_ptc from embedded data
	if err := li.loadNbPtcFromBytes(embeddedNbPtc); err != nil {
		return fmt.Errorf("failed to load embedded nb_ptc: %w", err)
	}

	// Load nb_pc from embedded data
	if err := li.loadNbPcFromBytes(embeddedNbPc); err != nil {
		return fmt.Errorf("failed to load embedded nb_pc: %w", err)
	}

	// Load nb_classes from embedded data
	if err := li.loadNbClassesFromBytes(embeddedNbClasses); err != nil {
		return fmt.Errorf("failed to load embedded nb_classes: %w", err)
	}

	// Load tk_nextmove from embedded data
	if err := li.loadTkNextmoveFromBytes(embeddedTkNextmove); err != nil {
		return fmt.Errorf("failed to load embedded tk_nextmove: %w", err)
	}

	// Load tk_output from embedded data
	if err := li.loadTkOutputFromBytes(embeddedTkOutput); err != nil {
		return fmt.Errorf("failed to load embedded tk_output: %w", err)
	}

	return nil
}

// loadModel loads the model from files
func (li *LanguageIdentifier) loadModel(modelDir string) error {
	// Load nb_ptc.bin (probability table)
	if err := li.loadNbPtc(filepath.Join(modelDir, "nb_ptc.bin")); err != nil {
		return fmt.Errorf("failed to load nb_ptc: %w", err)
	}

	// Load nb_pc.bin (prior probabilities)
	if err := li.loadNbPc(filepath.Join(modelDir, "nb_pc.bin")); err != nil {
		return fmt.Errorf("failed to load nb_pc: %w", err)
	}

	// Load nb_classes.json (language codes)
	if err := li.loadNbClasses(filepath.Join(modelDir, "nb_classes.json")); err != nil {
		return fmt.Errorf("failed to load nb_classes: %w", err)
	}

	// Load tk_nextmove.bin (tokenizer state transitions)
	if err := li.loadTkNextmove(filepath.Join(modelDir, "tk_nextmove.bin")); err != nil {
		return fmt.Errorf("failed to load tk_nextmove: %w", err)
	}

	// Load tk_output.bin (tokenizer output mappings)
	if err := li.loadTkOutput(filepath.Join(modelDir, "tk_output.bin")); err != nil {
		return fmt.Errorf("failed to load tk_output: %w", err)
	}

	return nil
}

// loadNbPtc loads the probability table from file
func (li *LanguageIdentifier) loadNbPtc(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", filename, err)
	}
	return li.loadNbPtcFromBytes(data)
}

// loadNbPc loads the prior probabilities from file
func (li *LanguageIdentifier) loadNbPc(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", filename, err)
	}
	return li.loadNbPcFromBytes(data)
}

// loadNbClasses loads the language codes from file
func (li *LanguageIdentifier) loadNbClasses(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", filename, err)
	}
	return li.loadNbClassesFromBytes(data)
}

// loadTkNextmove loads the tokenizer state transitions from file
func (li *LanguageIdentifier) loadTkNextmove(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", filename, err)
	}
	return li.loadTkNextmoveFromBytes(data)
}

// loadTkOutput loads the tokenizer output mappings from file
func (li *LanguageIdentifier) loadTkOutput(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", filename, err)
	}
	return li.loadTkOutputFromBytes(data)
}

// loadNbPtcFromBytes loads the probability table from byte data
func (li *LanguageIdentifier) loadNbPtcFromBytes(data []byte) error {
	// Read shape information (little-endian)
	if len(data) < 8 {
		return fmt.Errorf("insufficient data for nb_ptc shape")
	}
	rows := binary.LittleEndian.Uint32(data[0:4])
	cols := binary.LittleEndian.Uint32(data[4:8])
	data = data[8:]

	// Read data as float32 array
	if len(data) < int(rows*cols*4) {
		return fmt.Errorf("insufficient data for nb_ptc matrix")
	}

	// Reshape to 2D array
	li.nbPtc = make([][]float32, rows)
	for i := range li.nbPtc {
		li.nbPtc[i] = make([]float32, cols)
		for j := range li.nbPtc[i] {
			offset := (i*int(cols) + j) * 4
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			li.nbPtc[i][j] = math.Float32frombits(bits)
		}
	}

	return nil
}

// loadNbPcFromBytes loads the prior probabilities from byte data
func (li *LanguageIdentifier) loadNbPcFromBytes(data []byte) error {
	// Read length (little-endian)
	if len(data) < 4 {
		return fmt.Errorf("insufficient data for nb_pc length")
	}
	length := binary.LittleEndian.Uint32(data[0:4])
	data = data[4:]

	// Read data as float32 array
	if len(data) < int(length*4) {
		return fmt.Errorf("insufficient data for nb_pc vector")
	}

	li.nbPc = make([]float32, length)
	for i := range li.nbPc {
		offset := i * 4
		bits := binary.LittleEndian.Uint32(data[offset : offset+4])
		li.nbPc[i] = math.Float32frombits(bits)
	}

	return nil
}

// loadNbClassesFromBytes loads the language codes from byte data
func (li *LanguageIdentifier) loadNbClassesFromBytes(data []byte) error {
	return json.Unmarshal(data, &li.nbClasses)
}

// loadTkNextmoveFromBytes loads the tokenizer state transitions from byte data
func (li *LanguageIdentifier) loadTkNextmoveFromBytes(data []byte) error {
	// Read length (little-endian)
	if len(data) < 4 {
		return fmt.Errorf("insufficient data for tk_nextmove length")
	}
	length := binary.LittleEndian.Uint32(data[0:4])
	data = data[4:]

	// Read data as uint16 array
	if len(data) < int(length*2) {
		return fmt.Errorf("insufficient data for tk_nextmove array")
	}

	li.tkNextmove = make([]uint16, length)
	for i := range li.tkNextmove {
		offset := i * 2
		li.tkNextmove[i] = binary.LittleEndian.Uint16(data[offset : offset+2])
	}

	return nil
}

// loadTkOutputFromBytes loads the tokenizer output mappings from byte data
func (li *LanguageIdentifier) loadTkOutputFromBytes(data []byte) error {
	// Read number of entries (little-endian)
	if len(data) < 4 {
		return fmt.Errorf("insufficient data for tk_output entries")
	}
	numEntries := binary.LittleEndian.Uint32(data[0:4])
	data = data[4:]

	li.tkOutput = make(map[uint32][]uint16)
	offset := 0

	for i := uint32(0); i < numEntries; i++ {
		if offset+8 > len(data) {
			return fmt.Errorf("insufficient data for tk_output entry %d", i)
		}

		// Read key (uint32)
		key := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4

		// Read value length (uint32)
		valueLength := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4

		// Read values (uint16 array)
		if offset+int(valueLength*2) > len(data) {
			return fmt.Errorf("insufficient data for tk_output values")
		}

		values := make([]uint16, valueLength)
		for j := uint32(0); j < valueLength; j++ {
			valueOffset := offset + int(j*2)
			values[j] = binary.LittleEndian.Uint16(data[valueOffset : valueOffset+2])
		}
		offset += int(valueLength * 2)

		li.tkOutput[key] = values
	}

	return nil
}
