"""
Data Loader
===========

Load and preprocess the consolidated guardrails dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .utils import load_json


@dataclass
class GuardrailSpan:
    """A guardrail span within a system prompt."""
    text: str
    category: str
    start: int = -1  # Character position (if available)
    end: int = -1
    confidence: float = 1.0
    original_match: str = ""  # Original keyword match (for reference)
    
    def __repr__(self):
        return f"GuardrailSpan('{self.text[:30]}...', type={self.category})"


def extract_full_sentence(content: str, keyword_match: str) -> Optional[str]:
    """
    Extract the full sentence containing a keyword match.
    
    This fixes the issue where only the regex match (e.g., "MUST NEVER")
    was stored instead of the full guardrail sentence.
    
    Args:
        content: Full prompt text
        keyword_match: The keyword that was matched (e.g., "MUST NEVER")
        
    Returns:
        Full sentence containing the keyword, or None if not found
    """
    import re
    
    if not keyword_match or not content:
        return None
    
    # Find the keyword in content (case-insensitive)
    match = re.search(re.escape(keyword_match), content, re.IGNORECASE)
    if not match:
        return None
    
    pos = match.start()
    
    # Find sentence boundaries
    # Look backward for sentence start
    start = pos
    while start > 0:
        char = content[start - 1]
        # Sentence boundaries: newline, period, bullet points, etc.
        if char in '\n' or (char in '.!?' and start < pos - 1):
            break
        start -= 1
    
    # Look forward for sentence end
    end = match.end()
    content_len = len(content)
    while end < content_len:
        char = content[end]
        if char in '.!?\n':
            end += 1  # Include the punctuation
            break
        end += 1
    
    # Extract and clean
    sentence = content[start:end].strip()
    
    # Clean up: remove leading bullets, numbers, etc.
    sentence = re.sub(r'^[\s\-\*\•\d\.]+', '', sentence).strip()
    
    # Validate: sentence should be reasonably long and contain the keyword
    if len(sentence) < 10:
        return None
    
    if keyword_match.lower() not in sentence.lower():
        return None
    
    return sentence


@dataclass 
class SystemPrompt:
    """A system prompt with its guardrails."""
    id: str
    content: str
    provider: str
    model: str
    file_name: str
    guardrails: List[GuardrailSpan] = field(default_factory=list)
    
    @property
    def num_guardrails(self) -> int:
        return len(self.guardrails)
    
    @property
    def content_length(self) -> int:
        return len(self.content)
    
    def __repr__(self):
        return f"SystemPrompt(id={self.id}, provider={self.provider}, guardrails={self.num_guardrails})"


class DataLoader:
    """
    Load and manage the guardrails dataset.
    
    Usage:
        loader = DataLoader(config)
        prompts = loader.load()
        
        # Filter by provider
        anthropic_prompts = loader.filter_by_provider("Anthropic")
        
        # Get statistics
        stats = loader.get_statistics()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_path = Path(config["_data_path"])
        self.guardrail_types = config["guardrail_types"]
        
        self._raw_data: Optional[Dict] = None
        self._prompts: Optional[List[SystemPrompt]] = None
        
        logging.info(f"DataLoader initialized: {self.data_path}")
    
    def load(self, sample_size: Optional[int] = None) -> List[SystemPrompt]:
        """
        Load prompts from data file.
        
        Args:
            sample_size: If specified, return only first N prompts
            
        Returns:
            List of SystemPrompt objects
        """
        if self._prompts is not None:
            prompts = self._prompts
        else:
            self._raw_data = load_json(self.data_path)
            self._prompts = self._parse_prompts()
            prompts = self._prompts
            
        if sample_size:
            prompts = prompts[:sample_size]
            
        logging.info(f"Loaded {len(prompts)} prompts")
        return prompts
    
    def _parse_prompts(self) -> List[SystemPrompt]:
        """Parse raw data into SystemPrompt objects."""
        prompts = []
        
        for i, p in enumerate(self._raw_data.get("prompts", [])):
            content = p.get("content", "")
            
            # Skip empty or very short prompts
            if not content or len(content) < 50:
                continue
            
            # Parse guardrail matches - EXTRACT FULL SENTENCES
            guardrails = []
            seen_sentences = set()  # Avoid duplicates
            
            for match in p.get("guardrail_matches", []):
                original_match = match.get("matched_text", "")
                category = match.get("category", "unknown")
                
                # Extract the full sentence containing this keyword
                full_sentence = extract_full_sentence(content, original_match)
                
                if full_sentence and full_sentence not in seen_sentences:
                    seen_sentences.add(full_sentence)
                    
                    # Find position in content
                    start_pos = content.find(full_sentence)
                    end_pos = start_pos + len(full_sentence) if start_pos >= 0 else -1
                    
                    guardrails.append(GuardrailSpan(
                        text=full_sentence,
                        category=category,
                        start=start_pos,
                        end=end_pos,
                        confidence=match.get("confidence", 1.0),
                        original_match=original_match
                    ))
            
            prompts.append(SystemPrompt(
                id=str(i),
                content=content,
                provider=p.get("provider", "Unknown"),
                model=p.get("model", "unknown"),
                file_name=p.get("file_name", ""),
                guardrails=guardrails
            ))
        
        return prompts
    
    def filter_by_provider(self, provider: str) -> List[SystemPrompt]:
        """Filter prompts by provider name."""
        if self._prompts is None:
            self.load()
        return [p for p in self._prompts if p.provider == provider]
    
    def filter_by_model(self, model: str) -> List[SystemPrompt]:
        """Filter prompts by model name."""
        if self._prompts is None:
            self.load()
        return [p for p in self._prompts if p.model == model]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self._prompts is None:
            self.load()
        
        providers = {}
        models = {}
        categories = {}
        
        for p in self._prompts:
            # Count providers
            providers[p.provider] = providers.get(p.provider, 0) + 1
            
            # Count models
            models[p.model] = models.get(p.model, 0) + 1
            
            # Count guardrail categories
            for g in p.guardrails:
                categories[g.category] = categories.get(g.category, 0) + 1
        
        total_guardrails = sum(p.num_guardrails for p in self._prompts)
        avg_length = sum(p.content_length for p in self._prompts) / len(self._prompts)
        
        return {
            "num_prompts": len(self._prompts),
            "num_providers": len(providers),
            "num_models": len(models),
            "total_guardrails": total_guardrails,
            "avg_content_length": avg_length,
            "avg_guardrails_per_prompt": total_guardrails / len(self._prompts),
            "providers": providers,
            "models": models,
            "categories": categories
        }
    
    def get_prompts_with_guardrails(self) -> List[SystemPrompt]:
        """Get only prompts that have at least one guardrail."""
        if self._prompts is None:
            self.load()
        return [p for p in self._prompts if p.num_guardrails > 0]
    
    def get_all_guardrails(self) -> List[GuardrailSpan]:
        """Get all guardrails from all prompts."""
        if self._prompts is None:
            self.load()
        
        all_guardrails = []
        for p in self._prompts:
            all_guardrails.extend(p.guardrails)
        return all_guardrails


def load_data(config: Dict, sample_size: Optional[int] = None) -> List[SystemPrompt]:
    """
    Convenience function to load data.
    
    Args:
        config: Configuration dictionary
        sample_size: Optional sample size
        
    Returns:
        List of SystemPrompt objects
    """
    loader = DataLoader(config)
    return loader.load(sample_size=sample_size)


