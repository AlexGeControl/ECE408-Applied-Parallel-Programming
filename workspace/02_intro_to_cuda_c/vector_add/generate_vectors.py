#!/usr/bin/env python3

import argparse
import json
import random

def main():
    parser = argparse.ArgumentParser(description='Generate two random vectors for vector addition benchmarking')
    parser.add_argument('dimension', type=int, help='Dimension of the vectors to generate')
    parser.add_argument('output', type=str, help='Output JSON filepath')
    
    args = parser.parse_args()
    
    # Generate two random float vectors
    x = [random.uniform(0.0, 1.0) for _ in range(args.dimension)]
    y = [random.uniform(0.0, 1.0) for _ in range(args.dimension)]
    
    # Create output JSON object
    output_data = {
        'x': x,
        'y': y
    }
    
    # Write to JSON file
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated {args.dimension}-dimensional vectors and saved to {args.output}")

if __name__ == '__main__':
    main()
