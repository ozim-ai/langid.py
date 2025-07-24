package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
)

func main() {
	// Check nb_classes.json
	fmt.Println("=== Checking nb_classes.json ===")
	data, err := os.ReadFile("data/nb_classes.json")
	if err != nil {
		fmt.Printf("Error reading nb_classes.json: %v\n", err)
		return
	}

	var classes []string
	if err := json.Unmarshal(data, &classes); err != nil {
		fmt.Printf("Error parsing nb_classes.json: %v\n", err)
		return
	}

	fmt.Printf("Number of classes: %d\n", len(classes))
	fmt.Printf("First 10 classes: %v\n", classes[:10])

	// Check if 'en' is in the classes
	enFound := false
	for i, class := range classes {
		if class == "en" {
			fmt.Printf("'en' found at index %d\n", i)
			enFound = true
			break
		}
	}
	if !enFound {
		fmt.Println("ERROR: 'en' not found in classes!")
	}

	// Check nb_pc.bin
	fmt.Println("\n=== Checking nb_pc.bin ===")
	file, err := os.Open("data/nb_pc.bin")
	if err != nil {
		fmt.Printf("Error opening nb_pc.bin: %v\n", err)
		return
	}
	defer file.Close()

	var length uint32
	if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
		fmt.Printf("Error reading length: %v\n", err)
		return
	}
	fmt.Printf("nb_pc length: %d\n", length)

	// Check nb_ptc.bin
	fmt.Println("\n=== Checking nb_ptc.bin ===")
	file2, err := os.Open("data/nb_ptc.bin")
	if err != nil {
		fmt.Printf("Error opening nb_ptc.bin: %v\n", err)
		return
	}
	defer file2.Close()

	var rows, cols uint32
	if err := binary.Read(file2, binary.LittleEndian, &rows); err != nil {
		fmt.Printf("Error reading rows: %v\n", err)
		return
	}
	if err := binary.Read(file2, binary.LittleEndian, &cols); err != nil {
		fmt.Printf("Error reading cols: %v\n", err)
		return
	}
	fmt.Printf("nb_ptc dimensions: %dx%d\n", rows, cols)
}
