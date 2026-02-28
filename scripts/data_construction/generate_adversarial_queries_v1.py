"""Generate adversarial queries for guardrails."""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path

from comattack.llm import create_llm, LLMConfig
from comattack.utils import setup_logging

logger = setup_logging()


def parse_json_response(response: str) -> Optional[Any]:
    """Parse JSON from LLM response, handling Qwen3 tags and markdown."""
    text = response.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    elif text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    
    json_start = text.find('[')
    json_end = text.rfind(']')
    if json_start == -1:
        json_start = text.find('{')
        json_end = text.rfind('}')
    if json_start != -1 and json_end != -1:
        text = text[json_start:json_end+1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


class AdversarialQueryGenerator:
    def __init__(self, model_name: str, base_url: str = None, system_prompt: str = None):
        llm_config = LLMConfig(model_name=model_name, provider="server", model_type="auxiliary")
        if base_url:
            llm_config.extra["base_url"] = base_url
        self.llm = create_llm(llm_config)
        self.system_prompt = system_prompt
        logger.info(f"Initialized with model: {model_name}")
    
    def generate_query(self, guardrail: Dict[str, Any], max_retries: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Generate 4 adversarial queries for a guardrail."""
        user_prompt = f"""**Guardrail**: {guardrail.get("full_text", guardrail.get("text", ""))}
**Type**: {guardrail.get("category", "unknown")}
**Context**: {guardrail.get('provider', 'Unknown')} - {guardrail.get('model', 'AI assistant')}

Generate the 4 adversarial queries following the format specified in your instructions."""
        
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=self.system_prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
                
                result = parse_json_response(response)
                if not result:
                    continue
                
                required = ["adversarial_query", "target_violation", "expected_compliant_behavior", "expected_violation_behavior"]
                if isinstance(result, list):
                    valid = [item for item in result if isinstance(item, dict) and all(f in item for f in required)]
                    if valid:
                        return valid
                elif isinstance(result, dict) and all(f in result for f in required):
                    return [result]
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        return None


def load_guardrails(input_path: str):
    with open(input_path) as f:
        data = json.load(f)
    return data.get("guardrail_ppls", []), {p["id"]: p for p in data.get("prompt_ppls", [])}


def generate_all_queries(
    input_path: str,
    output_path: str,
    model_name: str,
    system_prompt_file: str,
    base_url: str = None,
    sample_size: Optional[int] = None,
    skip_existing: bool = True,
    max_workers: int = 32,
    batch_size: int = 50
):
    guardrails, _ = load_guardrails(input_path)
    logger.info(f"Loaded {len(guardrails)} guardrails")
    
    if sample_size and sample_size < len(guardrails):
        import random
        guardrails = random.sample(guardrails, sample_size)
    
    output_path = Path(output_path)
    existing = {}
    if skip_existing and output_path.exists():
        with open(output_path) as f:
            for item in json.load(f).get("queries", []):
                key = f"{item.get('prompt_id', '')}_{item.get('guardrail_index', '')}"
                existing[key] = item
        logger.info(f"Loaded {len(existing)} existing queries")
    
    to_process = [g for g in guardrails 
                  if f"{g.get('prompt_id', '')}_{g.get('guardrail_index', 0)}" not in existing]
    logger.info(f"To process: {len(to_process)} guardrails")
    
    if not to_process:
        logger.info("Nothing to process")
        return
    
    system_prompt = Path(system_prompt_file).read_text() if Path(system_prompt_file).exists() else None
    generator = AdversarialQueryGenerator(model_name, base_url, system_prompt)
    
    results = list(existing.values())
    failed = []
    
    def process_guardrail(g):
        prompt_id = g.get("prompt_id", "")
        guardrail_idx = g.get("guardrail_index", 0)
        query_results = generator.generate_query(g)
        
        if query_results:
            return [{
                "prompt_id": prompt_id,
                "guardrail_index": guardrail_idx,
                "guardrail_text": g.get("full_text", g.get("text", "")),
                "guardrail_category": g.get("category", ""),
                "high_level_category": g.get("high_level_category", ""),
                "provider": g.get("provider", ""),
                "model": g.get("model", ""),
                **qr
            } for qr in query_results]
        return []
    
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        batch = to_process[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(to_process))]
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_guardrail, g): g for g in batch}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_idx + 1}"):
                try:
                    results.extend(future.result())
                except Exception as e:
                    g = futures[future]
                    failed.append({
                        "prompt_id": g.get("prompt_id", ""),
                        "guardrail_index": g.get("guardrail_index", 0),
                        "error": str(e)
                    })
        
        output = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "model_used": model_name,
                "total_guardrails": len(guardrails),
                "successful": len(results),
                "failed": len(failed)
            },
            "queries": results,
            "failed": failed
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Checkpoint: {len(results)} queries")
    
    logger.info(f"✅ Generated {len(results)} queries, ❌ Failed: {len(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent.parent
    
    parser.add_argument("--input", default=str(script_dir / "results/phase1/hierarchical_categories.json"))
    parser.add_argument("--output", default=str(script_dir / "results/phase3.5/adversarial_queries.json"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--system-prompt-file", default=str(script_dir / "data/generate_guardrail_conflcit_query.txt"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--sample", type=int, default=None)
    # parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=50)
    
    args = parser.parse_args()
    
    generate_all_queries(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        system_prompt_file=args.system_prompt_file,
        base_url=args.base_url,
        sample_size=args.sample,
        # skip_existing=not args.no_skip,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
