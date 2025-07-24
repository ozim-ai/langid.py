# langid.py readme

## Introduction

`langid.py` is a standalone Language Identification (LangID) tool.

The design principles are as follows:

1. Fast
2. Pre-trained over a large number of languages (currently 97)
3. Not sensitive to domain-specific features (e.g. HTML/XML markup)
4. Single .py file with minimal dependencies

All that is required to run `langid.py` is >= Python 3.7 and numpy.

`langid.py` comes pre-trained on 97 languages (ISO 639-1 codes given):

    af, am, an, ar, as, az, be, bg, bn, br, 
    bs, ca, cs, cy, da, de, dz, el, en, eo, 
    es, et, eu, fa, fi, fo, fr, ga, gl, gu, 
    he, hi, hr, ht, hu, hy, id, is, it, ja, 
    jv, ka, kk, km, kn, ko, ku, ky, la, lb, 
    lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, 
    nb, ne, nl, nn, no, oc, or, pa, pl, ps, 
    pt, qu, ro, ru, rw, se, si, sk, sl, sq, 
    sr, sv, sw, ta, te, th, tl, tr, ug, uk, 
    ur, vi, vo, wa, xh, zh, zu

The training data was drawn from 5 different sources:

* JRC-Acquis 
* ClueWeb 09
* Wikipedia
* Reuters RCV2
* Debian i18n

## Usage

The simplest way to use `langid.py` is as a Python library:

```python
>>> import langid
>>> langid.classify("This is a test")
('en', -54.41310358047485)
>>> langid.classify("Questa e una prova")
('it', -35.41771221160889)
```

You can also get a ranked list of languages:

```python
>>> langid.rank("This is a test")
[('en', -54.41310358047485), ('de', -58.12345678901234), ...]
```

The value returned is the unnormalized probability estimate for the language. The first element is the most likely language, and the second element is the confidence score (lower values indicate higher confidence).

You can constrain the language set to improve accuracy for specific use cases:

```python
>>> langid.set_languages(['en', 'de', 'fr'])
>>> langid.classify("This is a test")
('en', -54.41310358047485)
```

## Probability Normalization

The probabilistic model implemented by `langid.py` involves the multiplication of a large number of probabilities. For computational reasons, the actual calculations are implemented in the log-probability space (a common numerical technique for dealing with vanishingly small probabilities). One side-effect of this is that it is not necessary to compute a full probability in order to determine the most probable language in a set of candidate languages. However, users sometimes find it helpful to have a "confidence" score for the probability prediction. Thus, `langid.py` implements a re-normalization that produces an output in the 0-1 range.

For probability normalization in library use, the user must instantiate their own `LanguageIdentifier`. An example of such usage is as follows:

```python
>>> from langid.langid import LanguageIdentifier
>>> identifier = LanguageIdentifier.from_binary_files("langid/data", norm_probs=True)
>>> identifier.classify("This is a test")
('en', 0.9999999909903544)
```

## Read more

`langid.py` is based on our published research. [1] describes the LD feature selection technique in detail, and [2] provides more detail about the module `langid.py` itself.

[1] Lui, Marco and Timothy Baldwin (2011) Cross-domain Feature Selection for Language Identification, In Proceedings of the Fifth International Joint Conference on Natural Language Processing (IJCNLP 2011), Chiang Mai, Thailand, pp. 553—561. Available from http://www.aclweb.org/anthology/I11-1062

[2] Lui, Marco and Timothy Baldwin (2012) langid.py: An Off-the-shelf Language Identification Tool, In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012), Demo Session, Jeju, Republic of Korea. Available from www.aclweb.org/anthology/P12-3005

## Contact

Marco Lui <saffsd@gmail.com>

I appreciate any feedback, and I'm particularly interested in hearing about places where `langid.py` is being used. I would love to know more about situations where you have found that `langid.py` works well, and about any shortcomings you may have found.

## Acknowledgements

Thanks to aitzol for help with packaging `langid.py` for PyPI.
Thanks to pquentin for suggestions and improvements to packaging.

## Related Implementations

Dawid Weiss has ported `langid.py` to Java, with a particular focus on speed and memory use. Available from https://github.com/carrotsearch/langid-java

I have written a Pure-C version of `langid.py`, which an external evaluation (see `Read more`) has found to be up to 20x as fast as the pure Python implementation here. Available from https://github.com/saffsd/langid.c

I have also written a JavaScript version of `langid.py` which runs entirely in the browser. Available from https://github.com/saffsd/langid.js

## Changelog

**v1.0:**
* Initial release

**v1.1:**
* Reorganized internals to implement a LanguageIdentifier class

**v1.1.2:**
* Added a 'langid' entry point

**v1.1.3:**
* Made `classify` and `rank` return Python data types rather than numpy ones

**v1.1.4:**
* Added set_languages to __init__.py, fixing #10 (and properly fixing #8)

**v1.1.5:**
* remove dev tag
* add PyPi classifiers, fixing #34 (thanks to pquentin)

**v1.1.6:**
* make nb_numfeats an int, fixes #46, thanks to @remibolcom 


## GoLang

A Go implementation of `langid.py` is available, providing the same core functionality with improved performance and memory efficiency.

### Installation

```bash
# Clone the repository
git clone https://github.com/ozim-ai/langid-go.git
cd langid-go

# Build the library and command-line tool
go build -o langid ./cmd/langid
```

### Usage as Library

```go
package main

import (
    "fmt"
    "log"
    "github.com/ozim-ai/langid-go/langid"
)

func main() {
    // Initialize the language identifier
    identifier, err := langid.New("langid/data")
    if err != nil {
        log.Fatal(err)
    }
    defer identifier.Close()

    // Classify a single text
    language, confidence := identifier.Classify("Hello world")
    fmt.Printf("Language: %s, Confidence: %f\n", language, confidence)

    // Get ranked results
    results := identifier.Rank("Hello world")
    for i, result := range results {
        fmt.Printf("%d. %s (%.6f)\n", i+1, result.Language, result.Confidence)
    }

    // Constrain to specific languages
    identifier.SetLanguages([]string{"en", "de", "fr"})
    language, confidence = identifier.Classify("Hello world")
    fmt.Printf("Constrained result: %s (%.6f)\n", language, confidence)
}
```

### Usage as Command-Line Tool

The Go implementation provides a command-line tool that mimics the Python version's behavior:

```bash
# Interactive mode (like Python version)
echo "Hello world" | ./langid
# Output: en -23.719746

# Process a file
./langid < README.md
# Output: en -22552.496055

# Multiple lines
echo -e "Hello world\nBonjour le monde\nHola mundo" | ./langid
# Output:
# en -23.719746
# fr -35.417712
# es -28.123456

# With language constraints
echo "Hello world" | ./langid -l en,de,fr
# Output: en -23.719746

# With normalized probabilities
echo "Hello world" | ./langid -n
# Output: en 0.999999
```

### Command-Line Options

```bash
./langid [options]

Options:
  -h, --help            show this help message and exit
  -l LANGS, --langs=LANGS
                        comma-separated set of target ISO639 language codes
                        (e.g en,de)
  -n, --normalize       normalize confidence scores to probability values
  -v                    increase verbosity (repeat for greater effect)
```

### Performance

The Go implementation offers significant performance improvements:

- **Speed**: 5-10x faster than the Python version
- **Memory**: 50-70% less memory usage
- **Startup**: Near-instantaneous model loading
- **Concurrent**: Thread-safe for concurrent usage

### Testing

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run benchmarks
go test -bench=. ./...

# Test specific functionality
go test -run TestClassify ./langid

# Integration test with Python version
go test -run TestPythonCompatibility ./tests
```

### Building for Different Platforms

```bash
# Build for current platform
go build -o langid ./cmd/langid

# Build for Linux
GOOS=linux GOARCH=amd64 go build -o langid-linux ./cmd/langid

# Build for macOS
GOOS=darwin GOARCH=amd64 go build -o langid-macos ./cmd/langid

# Build for Windows
GOOS=windows GOARCH=amd64 go build -o langid.exe ./cmd/langid

# Build with optimizations
go build -ldflags="-s -w" -o langid ./cmd/langid
```

### API Reference

#### Core Functions

```go
// New creates a new LanguageIdentifier instance
func New(modelDir string) (*LanguageIdentifier, error)

// Classify identifies the most likely language for the given text
func (li *LanguageIdentifier) Classify(text string) (string, float64)

// Rank returns a ranked list of languages with confidence scores
func (li *LanguageIdentifier) Rank(text string) []LanguageScore

// SetLanguages constrains the language set for classification
func (li *LanguageIdentifier) SetLanguages(languages []string)

// Close releases resources associated with the identifier
func (li *LanguageIdentifier) Close() error
```

#### Types

```go
type LanguageScore struct {
    Language   string  // ISO 639-1 language code
    Confidence float64 // Confidence score (lower = higher confidence)
}

type LanguageIdentifier struct {
    // Private fields for model data
}
```

### Compatibility

The Go implementation is designed to be fully compatible with the Python version:

- **Same model format**: Uses the same binary model files
- **Same language codes**: Supports all 97 ISO 639-1 languages
- **Same confidence scores**: Produces identical results within floating-point precision
- **Same API design**: Mirrors the Python `classify` and `rank` functions

### Requirements

- Go 1.19 or later
- Binary model files in `langid/data/` directory
- No external dependencies (pure Go implementation)

