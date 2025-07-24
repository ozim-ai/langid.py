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

// New creates a new LanguageIdentifier instance
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

// Classify identifies the most likely language for the given text
func (li *LanguageIdentifier) Classify(text string) (string, float64) {
	li.mu.RLock()
	defer li.mu.RUnlock()

	// Extract features from text
	features := li.instance2fv(text)

	// Calculate language probabilities
	probs := li.nbClassprobs(features)

	// Find the best language (highest probability = least negative log prob)
	bestLang := ""
	bestScore := float32(math.Inf(-1))

	for i, lang := range li.nbClasses {
		// Skip if not in constrained set
		if !li.isLanguageAllowed(lang) {
			continue
		}

		score := probs[i]
		if score > bestScore {
			bestScore = score
			bestLang = lang
		}
	}

	if li.normProbs {
		// Normalize to 0-1 range
		bestScore = float32(li.normalizeProbability(probs, bestLang))
	}

	return bestLang, float64(bestScore)
}

// Rank returns a ranked list of languages with confidence scores
func (li *LanguageIdentifier) Rank(text string) []LanguageScore {
	li.mu.RLock()
	defer li.mu.RUnlock()

	// Extract features from text
	features := li.instance2fv(text)

	// Calculate language probabilities
	probs := li.nbClassprobs(features)

	// Create results for allowed languages only
	var results []LanguageScore
	for i, lang := range li.nbClasses {
		if !li.isLanguageAllowed(lang) {
			continue
		}

		score := probs[i]
		if li.normProbs {
			score = float32(li.normalizeProbability(probs, lang))
		}

		results = append(results, LanguageScore{
			Language:   lang,
			Confidence: float64(score),
		})
	}

	// Sort by confidence (lower = better)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Confidence < results[j].Confidence
	})

	return results
}

// SetLanguages constrains the language set for classification
func (li *LanguageIdentifier) SetLanguages(languages []string) {
	li.mu.Lock()
	defer li.mu.Unlock()

	if len(languages) == 0 {
		// Reset to all languages
		li.languages = li.allLanguages
		return
	}

	// Validate languages
	validLanguages := make([]string, 0, len(languages))
	for _, lang := range languages {
		if li.isValidLanguage(lang) {
			validLanguages = append(validLanguages, lang)
		}
	}

	li.languages = validLanguages
}

// Close releases resources associated with the identifier
func (li *LanguageIdentifier) Close() error {
	li.mu.Lock()
	defer li.mu.Unlock()

	// Clear large data structures
	li.nbPtc = nil
	li.nbPc = nil
	li.nbClasses = nil
	li.tkNextmove = nil
	li.tkOutput = nil

	return nil
}

// SetNormalizeProbs enables or disables probability normalization
func (li *LanguageIdentifier) SetNormalizeProbs(normalize bool) {
	li.mu.Lock()
	defer li.mu.Unlock()
	li.normProbs = normalize
}

// loadModel loads the binary model files
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

// loadNbPtc loads the probability table from binary file
func (li *LanguageIdentifier) loadNbPtc(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Read shape information (little-endian)
	var rows, cols uint32
	if err := binary.Read(file, binary.LittleEndian, &rows); err != nil {
		return err
	}
	if err := binary.Read(file, binary.LittleEndian, &cols); err != nil {
		return err
	}

	// Read data as float32 array
	data := make([]float32, rows*cols)
	if err := binary.Read(file, binary.LittleEndian, &data); err != nil {
		return err
	}

	// Reshape to 2D array
	li.nbPtc = make([][]float32, rows)
	for i := range li.nbPtc {
		li.nbPtc[i] = data[i*int(cols) : (i+1)*int(cols)]
	}

	return nil
}

// loadNbPc loads the prior probabilities from binary file
func (li *LanguageIdentifier) loadNbPc(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Read length (little-endian)
	var length uint32
	if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
		return err
	}

	// Read data as float32 array
	li.nbPc = make([]float32, length)
	if err := binary.Read(file, binary.LittleEndian, &li.nbPc); err != nil {
		return err
	}

	return nil
}

// loadNbClasses loads the language codes from JSON file
func (li *LanguageIdentifier) loadNbClasses(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&li.nbClasses); err != nil {
		return err
	}

	return nil
}

// loadTkNextmove loads the tokenizer state transitions from binary file
func (li *LanguageIdentifier) loadTkNextmove(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Read length (little-endian)
	var length uint32
	if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
		return err
	}

	// Read data as uint16 array
	li.tkNextmove = make([]uint16, length)
	if err := binary.Read(file, binary.LittleEndian, &li.tkNextmove); err != nil {
		return err
	}

	return nil
}

// loadTkOutput loads the tokenizer output mappings from binary file
func (li *LanguageIdentifier) loadTkOutput(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Read number of entries (little-endian)
	var numEntries uint32
	if err := binary.Read(file, binary.LittleEndian, &numEntries); err != nil {
		return err
	}

	li.tkOutput = make(map[uint32][]uint16, numEntries)

	// Read each entry: key, value_length, values
	for i := uint32(0); i < numEntries; i++ {
		var key uint32
		if err := binary.Read(file, binary.LittleEndian, &key); err != nil {
			return err
		}

		var valueLength uint32
		if err := binary.Read(file, binary.LittleEndian, &valueLength); err != nil {
			return err
		}

		values := make([]uint16, valueLength)
		if err := binary.Read(file, binary.LittleEndian, &values); err != nil {
			return err
		}

		li.tkOutput[key] = values
	}

	return nil
}

// instance2fv converts text to feature vector
func (li *LanguageIdentifier) instance2fv(text string) []float32 {
	// Create feature vector (7480 features)
	features := make([]float32, 7480)

	// Convert text to UTF-8 bytes (matching Python behavior)
	textBytes := []byte(text)

	// Count the number of times we enter each state
	state := uint32(0)
	stateCount := make(map[uint32]int)

	for _, letter := range textBytes {
		// Calculate state index: (state << 8) + letter
		stateIndex := (state << 8) + uint32(letter)
		if stateIndex < uint32(len(li.tkNextmove)) {
			state = uint32(li.tkNextmove[stateIndex])
			stateCount[state]++
		}
	}

	// Update all the productions corresponding to the state
	for state, count := range stateCount {
		if outputs, exists := li.tkOutput[state]; exists {
			for _, index := range outputs {
				if uint32(index) < uint32(len(features)) {
					features[index] += float32(count)
				}
			}
		}
	}

	return features
}

// nbClassprobs calculates language probabilities
func (li *LanguageIdentifier) nbClassprobs(features []float32) []float32 {
	numClasses := len(li.nbPc)
	probs := make([]float32, numClasses)

	// Initialize with prior probabilities
	copy(probs, li.nbPc)

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

// isLanguageAllowed checks if a language is in the constrained set
func (li *LanguageIdentifier) isLanguageAllowed(lang string) bool {
	if len(li.languages) == 0 {
		return true // No constraints
	}

	for _, allowed := range li.languages {
		if allowed == lang {
			return true
		}
	}
	return false
}

// isValidLanguage checks if a language code is valid
func (li *LanguageIdentifier) isValidLanguage(lang string) bool {
	for _, valid := range li.allLanguages {
		if valid == lang {
			return true
		}
	}
	return false
}

// normalizeProbability normalizes a probability to 0-1 range
func (li *LanguageIdentifier) normalizeProbability(probs []float32, targetLang string) float64 {
	// Find the target language index
	targetIdx := -1
	for i, lang := range li.nbClasses {
		if lang == targetLang {
			targetIdx = i
			break
		}
	}

	if targetIdx == -1 {
		return 0.0
	}

	// Convert to exponential form for numerical stability
	maxProb := float32(math.Inf(-1))
	for _, prob := range probs {
		if prob > maxProb {
			maxProb = prob
		}
	}

	// Calculate normalized probability
	targetProb := probs[targetIdx]
	expTarget := math.Exp(float64(targetProb - maxProb))

	sum := float64(0)
	for _, prob := range probs {
		sum += math.Exp(float64(prob - maxProb))
	}

	if sum == 0 {
		return 0.0
	}

	return expTarget / sum
}
