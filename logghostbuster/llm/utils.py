"""LLM utilities for location canonical naming."""

import os
import json
import re

from ..utils import logger


def get_llm_canonical_name(group_members):
    """
    Use Hugging Face transformers to determine canonical name for a group of nearby locations.
    Falls back to geographic center if transformers is unavailable.
    """
    # Prepare location info
    locations_info = []
    for loc in group_members:
        locations_info.append({
            'city': loc['city'] or 'Unknown',
            'country': loc['country'],
            'coordinates': f"{loc['lat']:.4f}, {loc['lon']:.4f}",
            'downloads': f"{loc['total_downloads']:,}"
        })
    
    # Create prompt
    prompt = f"""You are analyzing download hub locations that are geographically close and likely represent the same institution or research center.

Locations to analyze:
{json.dumps(locations_info, indent=2)}

These locations are within 10km of each other in the same country. Identify:
1. The most appropriate canonical name for this group (likely institution name)
2. If these represent a known research institution (e.g., "European Bioinformatics Institute (EBI)" for Hinxton/Sawston)

Respond in JSON format:
{{
  "canonical_name": "Institution or location name",
  "reasoning": "Brief explanation"
}}

If unsure, use the city name of the location with the most downloads."""
    
    # Try Hugging Face transformers (primary method)
    try:
        from transformers import pipeline
        
        model_name = os.getenv('HF_MODEL', 'microsoft/DialoGPT-medium')  # Can use llama2, mistral, etc.
        
        # Use a smaller, faster model if available
        try:
            generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto",
                max_new_tokens=200,
                temperature=0.3
            )
        except Exception as e:
            # Fall back to geographic center if model loading fails
            logger.warning(f"Could not load {model_name}: {e}, using geographic center instead")
            return group_members[0]['geo_location']
        
        full_prompt = f"You are a geographic and institutional data analyzer. Respond only with valid JSON.\n\n{prompt}"
        result_text = generator(full_prompt, return_full_text=False)[0]['generated_text']
        
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            canonical_name = result.get('canonical_name', group_members[0]['city'] or group_members[0]['geo_location'])
            logger.info(f"Hugging Face model grouped {len(group_members)} locations as: {canonical_name}")
            
            top_member = max(group_members, key=lambda x: x['total_downloads'])
            return top_member['geo_location']
        else:
            logger.warning("Could not extract JSON from Hugging Face response, using geographic center")
            return group_members[0]['geo_location']
            
    except ImportError:
        logger.warning("transformers package not installed, using geographic center instead")
    except Exception as e:
        logger.warning(f"Hugging Face grouping failed: {e}, using geographic center instead")
    
    # Final fallback: use geographic center
    logger.info(f"Using geographic center for {len(group_members)} locations (no LLM available)")
    return group_members[0]['geo_location']
