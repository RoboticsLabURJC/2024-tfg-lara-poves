#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <file>"
  exit 1
fi

file="$1"

if [ ! -f "$file" ]; then
  echo "File $file doesn't exist"
  exit 1
fi

while IFS= read -r line; do
  carac=$(echo -n "$line" | wc -m)
  if [ "$carac" -gt 90 ]; then
    echo "Line '$line': $caracteres"
  fi
done < "$file"
