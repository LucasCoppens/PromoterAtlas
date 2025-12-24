#!/usr/bin/env python
"""
DEPRECATED: This script is deprecated in favor of the unified annotate.py script.

This wrapper maintains backward compatibility but will be removed in a future version.
Please use: python scripts/annotate.py --input your_genome.gb --output annotated.gb

For GenBank files, the new script works identically to this one.
"""

import warnings
import sys
from pathlib import Path

# Issue deprecation warning
warnings.warn(
    "\n" + "="*70 + "\n"
    "DEPRECATION WARNING: annotate_genbank.py is deprecated.\n"
    "Please use 'annotate.py' instead:\n\n"
    "  python scripts/annotate.py --input genome.gb --output annotated.gb\n\n"
    "This wrapper will be removed in a future version.\n"
    "="*70,
    DeprecationWarning,
    stacklevel=2
)

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import and run the unified annotate script
import annotate

# Force format to genbank if not specified
if "--format" not in sys.argv:
    sys.argv.extend(["--format", "genbank"])

if __name__ == "__main__":
    annotate.main()
