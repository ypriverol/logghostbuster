"""EBI-specific feature extraction function.

This module contains the EBI-specific feature extraction logic that uses
EBI-tailored extractors (yearly patterns, time-of-day, country-level features).
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List

from ...utils import logger
from ..schema import LogSchema, EBI_SCHEMA
from ..base import BaseFeatureExtractor
from .ebi_extractors import (
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)


def extract_location_features(
    conn, 
    input_parquet: str,
    schema: Optional[LogSchema] = None,
    custom_extractors: Optional[List[BaseFeatureExtractor]] = None
):
    """
    EBI-specific feature extraction function.
    
    Extracts behavioral features per location using EBI-tailored extractors:
    - YearlyPatternExtractor: Yearly temporal patterns
    - TimeOfDayExtractor: Time-of-day patterns
    - CountryLevelExtractor: Country-level aggregations
    
    Features capture patterns that distinguish:
    - Bots: many users, low downloads/user, high hourly user density, irregular time patterns
    - Mirrors: few users, high downloads/user, systematic patterns, regular time patterns
    
    Args:
        conn: Database connection (DuckDB)
        input_parquet: Path to input parquet file
        schema: LogSchema defining field mappings (defaults to EBI_SCHEMA)
        custom_extractors: Optional list of custom feature extractors to apply
    
    Returns:
        DataFrame with extracted features
    """
    if schema is None:
        schema = EBI_SCHEMA
    
    # Import generic extraction function
    from ..extraction import extract_location_features as generic_extract
    
    # EBI-specific extractors (order matters - yearly must come before country-level)
    ebi_extractors = [
        YearlyPatternExtractor(schema),
        TimeOfDayExtractor(schema),
        CountryLevelExtractor(schema),
    ]
    
    # Use generic extraction function with EBI extractors
    return generic_extract(conn, input_parquet, schema, ebi_extractors, custom_extractors)
