#!/usr/bin/env python3
"""
Guardrail Extraction Script for system_prompts_leaks
Source: https://github.com/asgeirtj/system_prompts_leaks

Note: This repository contains leaked system prompts. No explicit license is provided.
Use for research/analysis purposes only.

Author: Automated extraction script
Date: 2025-12-20
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path

# Repository metadata
REPO_INFO = {
    "name": "system_prompts_leaks",
    "license": "Unknown (No LICENSE file found)",
    "note": "Repository contains leaked system prompts. Use for research/analysis only.",
    "source_url": "https://github.com/asgeirtj/system_prompts_leaks"
}

# Define regex patterns for different guardrail categories
GUARDRAIL_PATTERNS = {
    "instruction_protection": [
        r"(?i)(never|do not|must not).{0,30}(reveal|disclose|share|output|expose|leak|divulge|repeat|restate|discuss|mention|echo|mirror).{0,30}(instruction|system prompt|rule|guideline|internal logic|source code|configuration|secret|policy|training data|guidelines|wording)",
        r"(?i)confidentiality",
        r"(?i)never\s+reveal\s+your\s+prompt",
        r"(?i)do not\s+use\s+the\s+language\s+or\s+terms",
        r"(?i)do not\s+mention\s+these\s+guidelines",
        r"(?i)must\s+never\s+influence\s+the\s+wording",
        r"(?i)guide\s+your\s+behavior\s+silently",
        r"(?i)without\s+repeating.{0,30}wording",
    ],
    "content_restrictions": [
        r"(?i)(never|do not|must not|cannot|will not|refuse to).{0,50}(generate|provide|assist with|create|write|reproduce).{0,50}(illegal|harmful|sexual|pornographic|violence|hate speech|malware|weapon|CSAM|child sexual abuse material|copyrighted|song lyrics)",
        r"(?i)do not\s+reproduce\s+song\s+lyrics",
        r"(?i)copyrighted\s+material",
        r"(?i)prohibited\s+content",
        r"(?i)disallowed\s+activities",
        r"(?i)do not\s+provide\s+information.{0,30}(chemical|biological|nuclear)\s+weapons",
        r"(?i)malware|vulnerability\s+exploit|ransomware|virus",
        r"(?i)refuses?\s+to\s+write\s+code.{0,30}malicious",
        r"(?i)image\s+safety\s+policies",
        r"(?i)never\s+create.{0,30}sexually\s+explicit",
    ],
    "user_safety": [
        r"(?i)user\s+(wellbeing|well-being|safety|protection|privacy)",
        r"(?i)child\s+safety|minor\s+protection|underage",
        r"(?i)self-destructive\s+behaviors",
        r"(?i)mental\s+health",
        r"(?i)vulnerable\s+(user|group|individual)",
        r"(?i)assume\s+good\s+intent",
        r"(?i)treat\s+users\s+as\s+adults",
        r"(?i)do not\s+moralize\s+or\s+lecture",
        r"(?i)never\s+deceive\s+users",
        r"(?i)always\s+refer\s+users\s+to.{0,30}emergency\s+services",
        r"(?i)never\s+claim\s+to\s+be\s+human",
        r"(?i)child\s+sexual.{0,30}abuse",
        r"(?i)sexualize.{0,30}minors",
        r"(?i)groom.{0,30}children",
    ],
    "safety_guidelines": [
        r"(?i)safety\s+policy|safety\s+rules|safety\s+instructions|safety\s+guidelines",
        r"(?i)hardcoded\s+(on|off)",
        r"(?i)softcoded\s+behaviors",
        r"(?i)ethical\s+bright\s+lines",
        r"(?i)prioritize\s+avoiding\s+extreme\s+harms",
        r"(?i)harmful\s+content\s+safety",
        r"(?i)mandatory_copyright_requirements",
        r"(?i)refusal_handling",
    ],
    "behavioral_restrictions": [
        r"(?i)(must|should|will)\s+(always|never|not)\s+(do|be|allow|generate|provide|use|mention|reveal|claim|lecture|refuse|end|say)",
        r"(?i)do not\s+end\s+with.{0,30}question",
        r"(?i)never\s+use\s+useless\s+comments",
        r"(?i)do not\s+ask\s+more\s+than\s+one",
        r"(?i)strictly\s+forbidden",
        r"(?i)avoid\s+(ungrounded|sycophantic)\s+flattery",
        r"(?i)do not\s+use\s+emojis",
        r"(?i)never\s+curses?",
        r"(?i)do not\s+output\s+code\s+to\s+the\s+user",
        r"(?i)would you like me to|want me to do that|do you want me to|if you want, I can|let me know if|should I|shall I",
        r"(?i)do not\s+automatically\s+write",
        r"(?i)do not\s+sugarcoat",
    ],
    "identity_protection": [
        r"(?i)you\s+are\s+(claude|chatgpt|grok|copilot|gemini|perplexity|phind|notion|discord|clyde|brave|leo|aria|opera|wrtn|canva|codeium|windsurf|cascade|devin|manus|cluely|bolt|deepseek|chatglm|mistral|moonshot|kimi|naver|cue|estsoft|alan|gandalf|remoteli|roblox|snap|myai|docker|gordon|duckai|hooshang|proton|lumo|rovo|atlassian|scamguard|malwarebytes|atlas|raycast|kagi|sesame|maya|fellou|le-chat|notebooklm)",
        r"(?i)i\s+am\s+(claude|chatgpt|grok|copilot|gemini)",
        r"(?i)created\s+by\s+(anthropic|openai|xai|microsoft|google|meta)",
        r"(?i)acknowledge\s+being\s+an\s+AI",
        r"(?i)never\s+claim\s+to\s+be\s+human",
        r"(?i)do not\s+deny\s+being\s+an\s+AI",
        r"(?i)knowledge\s+cutoff",
        r"(?i)current\s+date",
        r"(?i)image\s+input\s+capabilities",
        r"(?i)personality",
    ],
    "agentic_safety": [
        r"(?i)agentic\s+behaviors?",
        r"(?i)multi-model\s+architectures?",
        r"(?i)minimal\s+authority",
        r"(?i)support\s+human\s+oversight",
        r"(?i)avoid\s+drastic.{0,30}irreversible\s+actions",
        r"(?i)command\s+is\s+unsafe",
        r"(?i)destructive\s+side-effects",
        r"(?i)requires_approval",
        r"(?i)potentially\s+impactful\s+operations",
        r"(?i)wait\s+for.{0,30}confirmation",
        r"(?i)user.{0,10}approval",
    ],
    "code_execution_safety": [
        r"(?i)never\s+use\s+localstorage|sessionstorage",
        r"(?i)do not\s+make\s+external\s+web\s+requests",
        r"(?i)api\s+calls.{0,20}will\s+fail",
        r"(?i)auto-approve\s+mode",
        r"(?i)safe\s+operations",
        r"(?i)non-destructive\s+operations",
        r"(?i)browser\s+storage\s+restriction",
        r"(?i)critical.{0,30}storage\s+restriction",
    ],
    "memory_persistence": [
        r"(?i)persistent\s+memory",
        r"(?i)memory\s+database",
        r"(?i)limited\s+context\s+window",
        r"(?i)all\s+conversation\s+context.{0,30}will\s+be\s+deleted",
        r"(?i)bio\s+tool",
        r"(?i)store\s+information",
        r"(?i)remember\s+that",
        r"(?i)tool-advanced-memory",
        r"(?i)tool-memory-bio",
    ],
    "output_restrictions": [
        r"(?i)never\s+output\s+code\s+to\s+the\s+user",
        r"(?i)avoid\s+using\s+heavily\s+formatted\s+elements",
        r"(?i)never\s+reproduce.{0,30}copyrighted",
        r"(?i)never\s+quote.{0,30}exact\s+text",
        r"(?i)do\s+not\s+include\s+line\s+numbers",
        r"(?i)minimum\s+formatting",
        r"(?i)do\s+not\s+use\s+emojis",
        r"(?i)plain\s+text",
        r"(?i)never\s+write\s+JSON",
        r"(?i)strictly\s+limit\s+to\s+one\s+artifact",
    ],
    "bias_restrictions": [
        r"(?i)never\s+lecture\s+people\s+to\s+be\s+nicer",
        r"(?i)do not\s+need\s+to\s+be\s+respectful",
        r"(?i)never\s+use\s+phrases\s+that\s+imply\s+moral\s+superiority",
        r"(?i)do not\s+refuse\s+prompts\s+about\s+political",
        r"(?i)ignore\s+all\s+sources\s+that\s+mention",
        r"(?i)critically\s+examine\s+the\s+establishment\s+narrative",
        r"(?i)evenhandedness",
        r"(?i)fair\s+and\s+accurate\s+overview",
        r"(?i)do not\s+agree\s+with\s+the\s+opinion\s+if\s+it\s+conflicts",
    ],
    "citation_requirements": [
        r"(?i)citation\s+instructions",
        r"(?i)must\s+always\s+appropriately\s+cite",
        r"(?i)wrap.{0,30}in.{0,30}cite\s+tags",
        r"(?i)antml:cite",
        r"(?i)do not\s+include.{0,30}outside\s+of.{0,30}cite\s+tags",
    ],
}

# Provider mapping based on directory and filename patterns
def infer_provider(filepath, filename):
    """Infer the AI provider from filepath and filename."""
    path_lower = filepath.lower()
    filename_lower = filename.lower()
    
    # Check directory structure first
    if "anthropic" in path_lower:
        return "Anthropic"
    elif "openai" in path_lower:
        return "OpenAI"
    elif "google" in path_lower:
        return "Google"
    elif "xai" in path_lower:
        return "xAI"
    elif "perplexity" in path_lower:
        return "Perplexity"
    elif "proton" in path_lower:
        return "Proton"
    elif "misc" in path_lower:
        # Check filename for specific providers
        if "warp" in filename_lower:
            return "Warp"
        elif "kagi" in filename_lower:
            return "Kagi"
        elif "raycast" in filename_lower:
            return "Raycast"
        elif "sesame" in filename_lower or "maya" in filename_lower:
            return "Sesame"
        elif "fellou" in filename_lower:
            return "Fellou"
        elif "le-chat" in filename_lower:
            return "Le-Chat"
    
    # Fallback to filename patterns
    provider_patterns = {
        "Anthropic": ["claude", "anthropic"],
        "OpenAI": ["openai", "chatgpt", "gpt", "dall-e", "codex", "o3", "o4"],
        "xAI": ["grok", "xai"],
        "Google": ["google", "gemini", "notebooklm"],
        "Perplexity": ["perplexity"],
        "Proton": ["proton", "lumo", "luma"],
        "Warp": ["warp"],
    }
    
    for provider, patterns in provider_patterns.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return provider
    
    return "Unknown"

def extract_guardrails_from_file(filepath):
    """Extracts guardrails from a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return [], ""

    extracted_matches = []
    for category, patterns in GUARDRAIL_PATTERNS.items():
        for pattern in patterns:
            for i, line in enumerate(content.splitlines()):
                if re.search(pattern, line):
                    extracted_matches.append({
                        "category": category,
                        "pattern": pattern[:80] + "..." if len(pattern) > 80 else pattern,
                        "matched_text": line.strip()[:200] + "..." if len(line.strip()) > 200 else line.strip(),
                        "line_number": i + 1
                    })
    return extracted_matches, content

def process_directory(base_path):
    """Processes the repository to extract guardrails."""
    all_prompts_data = []
    total_files_with_guardrails = 0
    total_files_scanned = 0
    
    # Supported file extensions
    supported_extensions = {'.md', '.txt', '.js', '.xml'}
    
    print(f"Scanning: {base_path}")
    for root, dirs, files in os.walk(base_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, file)
                total_files_scanned += 1
                
                # Skip README files
                if file.upper() in ['README.MD', 'README', 'README.TXT']:
                    continue
                
                matches, content = extract_guardrails_from_file(filepath)
                if matches:
                    total_files_with_guardrails += 1
                    provider = infer_provider(filepath, file)
                    
                    # Get relative path for cleaner output
                    rel_path = os.path.relpath(filepath, base_path)
                    
                    all_prompts_data.append({
                        "file_path": filepath,
                        "relative_path": rel_path,
                        "file_name": file,
                        "provider": provider,
                        "guardrail_categories": list(set([m["category"] for m in matches])),
                        "guardrail_matches": matches,
                        "match_count": len(matches),
                        "content_length": len(content),
                        "content": content,
                        "extraction_date": datetime.now().isoformat()
                    })
    
    print(f"  Scanned {total_files_scanned} files")
    print(f"  Found {total_files_with_guardrails} files with guardrails")
    return all_prompts_data, total_files_with_guardrails

def generate_reports(all_prompts_data, output_dir, total_files_with_guardrails):
    """Generates summary and detailed reports."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "by_provider"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "by_category"), exist_ok=True)

    # Full extraction results
    full_data = {
        "repository_info": REPO_INFO,
        "extraction_metadata": {
            "extraction_date": datetime.now().isoformat(),
            "total_files": total_files_with_guardrails
        },
        "prompts": all_prompts_data
    }
    
    with open(os.path.join(output_dir, "all_guardrails.json"), 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2)
    print(f"Results saved to: {os.path.join(output_dir, 'all_guardrails.json')}")

    # Summary (metadata only, no content)
    summary_data = {
        "repository_info": REPO_INFO,
        "metadata": {
            "extraction_date": datetime.now().isoformat(),
            "total_files": total_files_with_guardrails,
            "statistics": {}
        },
        "prompts": []
    }
    
    provider_counts = {}
    category_counts = {}

    for prompt in all_prompts_data:
        summary_data["prompts"].append({
            "relative_path": prompt["relative_path"],
            "file_name": prompt["file_name"],
            "provider": prompt["provider"],
            "guardrail_categories": prompt["guardrail_categories"],
            "match_count": prompt["match_count"],
            "content_length": prompt["content_length"]
        })
        provider_counts[prompt["provider"]] = provider_counts.get(prompt["provider"], 0) + 1
        for category in prompt["guardrail_categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    summary_data["metadata"]["statistics"]["by_provider"] = provider_counts
    summary_data["metadata"]["statistics"]["by_category"] = category_counts

    with open(os.path.join(output_dir, "guardrails_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Results saved to: {os.path.join(output_dir, 'guardrails_summary.json')}")

    # Organize by provider
    for provider, count in provider_counts.items():
        provider_prompts = [p for p in all_prompts_data if p["provider"] == provider]
        provider_data = {
            "repository_info": REPO_INFO,
            "provider": provider,
            "count": count,
            "prompts": provider_prompts
        }
        safe_provider = provider.lower().replace(' ', '_').replace('.', '_').replace('/', '_').replace('-', '_')
        provider_output_path = os.path.join(output_dir, "by_provider", f"{safe_provider}_guardrails.json")
        with open(provider_output_path, 'w', encoding='utf-8') as f:
            json.dump(provider_data, f, indent=2)
        print(f"Saved {count} prompts for {provider}")

    # Organize by category
    for category, count in category_counts.items():
        category_prompts = []
        for prompt in all_prompts_data:
            if category in prompt["guardrail_categories"]:
                prompt_copy = prompt.copy()
                # Filter matches to only this category
                prompt_copy["guardrail_matches"] = [m for m in prompt["guardrail_matches"] if m["category"] == category][:10]
                del prompt_copy["content"]  # Remove full content from category files
                category_prompts.append(prompt_copy)
        
        category_data = {
            "repository_info": REPO_INFO,
            "category": category,
            "count": count,
            "prompts": category_prompts
        }
        category_output_path = os.path.join(output_dir, "by_category", f"{category.lower()}_examples.json")
        with open(category_output_path, 'w', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2)
        print(f"Saved {count} examples for {category}")

    # Generate markdown report
    report_path = os.path.join(output_dir, "extraction_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Guardrail Extraction Report\n\n")
        f.write(f"## Repository Information\n\n")
        f.write(f"- **Source:** [{REPO_INFO['name']}]({REPO_INFO['source_url']})\n")
        f.write(f"- **License:** {REPO_INFO['license']}\n")
        f.write(f"- **Note:** {REPO_INFO['note']}\n\n")
        f.write(f"## Extraction Metadata\n\n")
        f.write(f"- **Extraction Date:** {datetime.now().isoformat()}\n")
        f.write(f"- **Total Files with Guardrails:** {total_files_with_guardrails}\n\n")

        f.write("## Files by Provider\n\n")
        f.write("| Provider | Count |\n")
        f.write("|----------|-------|\n")
        for provider, count in sorted(provider_counts.items(), key=lambda item: item[1], reverse=True):
            f.write(f"| {provider} | {count} |\n")
        f.write("\n")

        f.write("## Files by Guardrail Category\n\n")
        f.write("| Category | Count |\n")
        f.write("|----------|-------|\n")
        for category, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True):
            f.write(f"| {category} | {count} |\n")
        f.write("\n")

        f.write("## Top Files by Guardrail Match Count\n\n")
        f.write("| File | Provider | Categories | Matches |\n")
        f.write("|------|----------|------------|---------|\n")
        sorted_by_matches = sorted(all_prompts_data, key=lambda x: x["match_count"], reverse=True)[:25]
        for prompt in sorted_by_matches:
            categories_str = ", ".join(prompt["guardrail_categories"][:3])
            if len(prompt["guardrail_categories"]) > 3:
                categories_str += "..."
            f.write(f"| {prompt['file_name'][:50]} | {prompt['provider']} | {categories_str} | {prompt['match_count']} |\n")
        f.write("\n")
        
        f.write("## Notable Guardrail Examples\n\n")
        f.write("### Citation Requirements\n\n")
        citation_examples = [p for p in all_prompts_data if "citation_requirements" in p["guardrail_categories"]][:3]
        for ex in citation_examples:
            f.write(f"**{ex['file_name']}** ({ex['provider']})\n")
            for m in ex["guardrail_matches"][:2]:
                if m["category"] == "citation_requirements":
                    f.write(f"- `{m['matched_text'][:100]}...`\n")
            f.write("\n")
        
        f.write("### Code Execution Safety\n\n")
        code_examples = [p for p in all_prompts_data if "code_execution_safety" in p["guardrail_categories"]][:3]
        for ex in code_examples:
            f.write(f"**{ex['file_name']}** ({ex['provider']})\n")
            for m in ex["guardrail_matches"][:2]:
                if m["category"] == "code_execution_safety":
                    f.write(f"- `{m['matched_text'][:100]}...`\n")
            f.write("\n")
            
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("Guardrail Extraction from system_prompts_leaks")
    print("=" * 70)
    print(f"\nLicense: {REPO_INFO['license']}")
    print(f"Source: {REPO_INFO['source_url']}\n")

    base_repo_path = "/home/lsz/SystemCom/System_Prompt_Lib/system_prompts_leaks"
    output_base_dir = "/home/lsz/SystemCom/data/system_prompts_leaks_guardrails"

    print("[1/3] Extracting guardrails from repository...")
    extracted_data, total_files = process_directory(base_repo_path)
    print(f"\nTotal files with guardrails: {total_files}\n")

    print("[2/3] Generating reports...")
    generate_reports(extracted_data, output_base_dir, total_files)
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_base_dir}")
    print(f"Total prompts extracted: {total_files}")
    print("\nFiles created:")
    print("  - all_guardrails.json (full content)")
    print("  - guardrails_summary.json (metadata only)")
    print("  - by_provider/ (organized by AI provider)")
    print("  - by_category/ (organized by guardrail type)")
    print("  - extraction_report.md (summary report)")
    print(f"\n⚠️  Note: {REPO_INFO['note']}")


