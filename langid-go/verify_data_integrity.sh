#!/bin/bash

# Data Integrity Verification Script
# Compares binary model files between langid/data and langid-go/data

echo "🔍 Verifying data integrity between langid/data and langid-go/data"
echo "================================================================"

# Function to compare files
compare_files() {
    local file1="$1"
    local file2="$2"
    local filename=$(basename "$1")
    
    echo -n "Checking $filename... "
    
    if [ ! -f "$file1" ]; then
        echo "❌ Missing: $file1"
        return 1
    fi
    
    if [ ! -f "$file2" ]; then
        echo "❌ Missing: $file2"
        return 1
    fi
    
    # Compare file sizes
    size1=$(stat -f%z "$file1" 2>/dev/null || stat -c%s "$file1" 2>/dev/null)
    size2=$(stat -f%z "$file2" 2>/dev/null || stat -c%s "$file2" 2>/dev/null)
    
    if [ "$size1" != "$size2" ]; then
        echo "❌ Size mismatch: $size1 vs $size2 bytes"
        return 1
    fi
    
    # Compare file contents
    if cmp -s "$file1" "$file2"; then
        echo "✅ Identical ($size1 bytes)"
        return 0
    else
        echo "❌ Content differs"
        return 1
    fi
}

# Function to show file details
show_file_details() {
    local file="$1"
    local filename=$(basename "$file")
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    local md5=$(md5sum "$file" 2>/dev/null || md5 -q "$file" 2>/dev/null)
    echo "  📄 $filename: $size bytes, MD5: $md5"
}

# Check each file
echo ""
echo "📊 File Comparison:"
echo "-------------------"

errors=0

# Compare binary files
for file in nb_ptc.bin nb_pc.bin tk_nextmove.bin tk_output.bin; do
    if ! compare_files "langid/data/$file" "langid-go/data/$file"; then
        ((errors++))
    fi
done

# Compare JSON file (text-based, so we'll do a different check)
echo -n "Checking nb_classes.json... "
if [ -f "langid/data/nb_classes.json" ] && [ -f "langid-go/data/nb_classes.json" ]; then
    if cmp -s "langid/data/nb_classes.json" "langid-go/data/nb_classes.json"; then
        size=$(stat -f%z "langid/data/nb_classes.json" 2>/dev/null || stat -c%s "langid/data/nb_classes.json" 2>/dev/null)
        echo "✅ Identical ($size bytes)"
    else
        echo "❌ Content differs"
        ((errors++))
    fi
else
    echo "❌ Missing file"
    ((errors++))
fi

echo ""
echo "📋 File Details:"
echo "----------------"

echo "langid/data/:"
for file in langid/data/*.bin langid/data/*.json; do
    show_file_details "$file"
done

echo ""
echo "langid-go/data/:"
for file in langid-go/data/*.bin langid-go/data/*.json; do
    show_file_details "$file"
done

echo ""
echo "📈 Summary:"
echo "-----------"

if [ $errors -eq 0 ]; then
    echo "✅ All files are identical! Data integrity verified."
    echo "🎯 Total files compared: 5"
    echo "✅ Successful comparisons: 5"
    echo "❌ Failed comparisons: 0"
else
    echo "❌ Found $errors file(s) with differences!"
    echo "🎯 Total files compared: 5"
    echo "✅ Successful comparisons: $((5 - errors))"
    echo "❌ Failed comparisons: $errors"
fi

echo ""
echo "🔧 Next Steps:"
if [ $errors -eq 0 ]; then
    echo "✅ Data integrity confirmed - the Go implementation is using identical model files"
    echo "✅ The classification issue is in the Go code logic, not the model data"
else
    echo "❌ Fix the data differences before proceeding"
    echo "❌ Copy the correct files from langid/data/ to langid-go/data/"
fi 