"""
EBI-specific feature extractors and convenience functions.

This module contains feature extractors and convenience functions specific to EBI log formats.
"""

from ..base import BaseFeatureExtractor
from .ebi_extraction import extract_location_features
from .ebi_extractors import (
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)
from ..schema import EBI_SCHEMA
import pandas as pd


def extract_location_features_ebi(conn, input_parquet):
    """
    Convenience function for EBI log format (backward compatibility).
    
    This uses EBI-specific extraction logic with EBI schema and EBI-tailored extractors.
    
    Args:
        conn: Database connection (DuckDB)
        input_parquet: Path to input parquet file
    
    Returns:
        DataFrame with extracted features using EBI schema and EBI extractors
    """
    return extract_location_features(conn, input_parquet, schema=EBI_SCHEMA)


# Example: EBI-specific extractor (if needed in the future)
class EBIProjectPatternExtractor(BaseFeatureExtractor):
    """
    Extract project-specific patterns for EBI logs.
    
    This is an example of a provider-specific extractor that could
    analyze patterns specific to EBI's accession/project structure.
    """
    
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        """
        Extract EBI-specific project patterns.
        
        For now, this is a placeholder. Add EBI-specific logic here
        if needed in the future.
        """
        # Example: Could analyze accession patterns, project distribution, etc.
        # For now, just return the dataframe unchanged
        return df
