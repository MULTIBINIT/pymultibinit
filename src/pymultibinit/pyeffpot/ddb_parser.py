"""
DDB file parser wrapper.

This module provides compatibility by wrapping ddb_parser_complete.
It is recommended to use ddb_parser_complete directly for new code.
"""
from .ddb_parser_complete import read_ddb
from .datastructures import UnitcellData

# Alias for backward compatibility if needed
DDBParser = None # The complete parser is in ddb_parser_complete
