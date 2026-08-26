#!/usr/bin/env python3
"""
Example:
    python generate_se_frames_v11.py \
        --categories X_cloud X_language X_database_relational \
        --n 10

Install:
    pip install openai groq pandas python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
DEFAULT_LLAMA_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ===========================================================================
# CATEGORY TAXONOMY
# ===========================================================================

CATEGORIES = {
    "L1_physical": {"label":"L1 — Physical","role":"physical connection or transmission between devices","technologies":["USB","Ethernet","Bluetooth","NPC"],"allowed_subjects":["device","system","hardware","equipment","machines"],"allowed_relations":["connects using {TECH}","uses {TECH} for physical connectivity","communicates through {TECH}"],"forbidden_subjects":["database","source code","backend logic"],"forbidden_relations":["implemented in","stores data using","compiled with"]},
    "L2_data_link": {"label":"L2 — Data Link","role":"link-layer or local network communication","technologies":["HDLC","Ethernet","Wi-Fi","PPP"],"allowed_subjects":["network","device","hosts","nodes","systems"],"allowed_relations":["uses {TECH} for local communication","connects using {TECH}","communicates over {TECH}"],"forbidden_subjects":["application data","database","source code"],"forbidden_relations":["implemented in","stores data using","written in"]},
    "L3_network": {"label":"L3 — Network","role":"network-layer communication","technologies":["IP","ICMP","IGMP","IPsec"],"allowed_subjects":["network","system","hosts","nodes","devices"],"allowed_relations":["uses {TECH} for network communication","communicates using {TECH}","relies on {TECH} for network communication"],"forbidden_subjects":["database","application data","source code"],"forbidden_relations":["stores data using","implemented in","hosted on"]},
    "L4_transport": {"label":"L4 — Transport","role":"transport-layer communication between networked components","technologies":["TCP","UDP","SCTP","QUIC"],"allowed_subjects":["service","application","system","components","services","nodes","backend"],"allowed_relations":["communicates using {TECH}","uses {TECH} for communication","exchanges data using {TECH}"],"forbidden_subjects":["database","source code","cloud platform"],"forbidden_relations":["stores data in","runs on","implemented in"]},
    "L5_session": {"label":"L5 — Session","role":"session or remote system interaction","technologies":["NFS","SMB","NetBios","RPC"],"allowed_subjects":["system","service","application","components","machines","services"],"allowed_relations":["uses {TECH} for system interaction","relies on {TECH} for remote interaction","connects through {TECH}"],"forbidden_subjects":["source code","programming language"],"forbidden_relations":["implemented in","compiled with"]},
    "L6_presentation": {"label":"L6 — Presentation","role":"data presentation, encoding, formatting, or representation","technologies":["SSL/TLS","JPEG","MPEG","ASCII"],"allowed_subjects":["application","system","service","data","content","components"],"allowed_relations":["uses {TECH} for data representation","handles data using {TECH}","uses {TECH} when presenting data"],"forbidden_subjects":["operating system","cloud environment"],"forbidden_relations":["runs on","deployed on","hosted on"]},
    "L7_application": {"label":"L7 — Application","role":"application-layer network communication","technologies":["HTTP","FTP","DNS","SMTP"],"allowed_subjects":["application","service","system","components","services","client","backend"],"allowed_relations":["uses {TECH} for application communication","communicates using {TECH}","uses {TECH} for network services"],"forbidden_subjects":["database","source code"],"forbidden_relations":["stores data in","implemented in","runs on"]},
    "X_language": {"label":"Programming, Scripting, and Markup Languages","role":"writing, defining, or implementing software and web content","technologies":["Javascript","HTML/CSS","SQL","Python","Bash/Shell"],"allowed_subjects":["project","developers","engineers","team","software","codebase"],"allowed_relations":["uses {TECH} in the project","works with {TECH} during development","uses {TECH} for development"],"forbidden_subjects":["network","cloud infrastructure"],"forbidden_relations":["hosted on","stores data in","routes using"]},
    "X_database": {"label":"Databases","role":"application data storage and management","technologies":["PostgresSQL","MySQL","SQLite","Microsoft SQL Server","Redis"],"allowed_subjects":["application","service","system","project","backend","developers","engineers","team"],"allowed_relations":["uses {TECH} for application data","stores application data using {TECH}","relies on {TECH} for data storage","uses {TECH} as the project database"],"forbidden_subjects":["network","operating system"],"forbidden_relations":["designs {TECH}","implements {TECH}","builds {TECH}"]},
    "X_cloud_development": {"label":"Cloud Development","role":"tools and platforms used in cloud-oriented software development and deployment workflows","technologies":["Docker","npm","AWS","Pip","Kubernetes"],"allowed_subjects":["project","team","developers","engineers","development workflow","application workflow"],"allowed_relations":["uses {TECH} in the development workflow","works with {TECH} during development","includes {TECH} in the project workflow"],"forbidden_subjects":["database","network protocol"],"forbidden_relations":["written in","stores data in"]},
    "X_web_technology": {"label":"Web Frameworks and Technologies","role":"web application development","technologies":["Node.js","React","jQuery","Next.js","Express"],"allowed_subjects":["web project","application","development team","developers","engineers"],"allowed_relations":["uses {TECH} for web development","develops the web application with {TECH}","uses {TECH} in the web project"],"forbidden_subjects":["network","database","operating system"],"forbidden_relations":["stores data in","hosted on"]},
    "X_ide": {"label":"Development IDEs and Editors","role":"software development and source-code editing","technologies":["Visual Studio Code","Visual Studio","Notepad++","IntelliJ IDEA","Vim"],"allowed_subjects":["developer","developers","engineer","engineers","team"],"allowed_relations":["uses {TECH} for development","works in {TECH} while developing software","uses {TECH} to edit project code"],"forbidden_subjects":["application","network","database"],"forbidden_relations":["deployed on","stores data in","routes using"]},
}


# ===========================================================================
# GENERATION PROMPTS
# ===========================================================================

SHARED_RULES = """
Generate exactly {n} distinct software-engineering sentence frames.

Category: {label}
Shared role: {role}

Use the exact token {{TECH}} exactly once in every frame.

The goal is a short, ordinary sentence whose technology slot belongs only to
the shared category role above.

HARD RULES:

1. Do not place "a", "an", or "the" directly before {{TECH}}.

2. Treat {{TECH}} as a technology proper name, not a countable object.

3. The sentence subject should normally come from this semantic set:
{allowed_subjects}

4. The relationship to {{TECH}} should stay close to one of these category-level
   relations:
{allowed_relations}

5. Do NOT use these inappropriate subjects or concepts:
{forbidden_subjects}

6. Do NOT use these inappropriate relations:
{forbidden_relations}

7. Do not explain why the technology was selected.

8. Do not mention special capabilities, product-specific features, APIs,
   syntax, commands, proprietary components, performance, security,
   scalability, reliability, popularity, cost, or other differentiating
   properties.

9. Avoid causal wording such as:
   because, since, due to, for its, enabling, allowing, so that,
   in order to, to improve, to ensure, to reduce, to optimize.

10. Do not mention another named technology.

11. Keep the sentence natural and concise, preferably 7-16 words.

12. Avoid vague or semantically empty subjects such as:
    "the environment", "the infrastructure", "the network", or
    "the deployment" UNLESS that subject is explicitly appropriate for this
    category.

13. Before returning a frame, ask:
    - Is the subject logically the thing that would use this category?
    - Is the verb something engineers naturally say with this category?
    - Does the sentence sound like ordinary software-engineering prose?
    If not, rewrite it.

Return ONLY JSON:
[
  {{"sentence_frame": "The team uses {{TECH}} for the project."}}
]
"""


PROMPTS = {
    "P3_strict_interchangeability": """
Act as a careful technical corpus writer. Every sentence must be both
category-correct and natural. Reject any sentence that is merely grammatical
but semantically strange, such as a network "using" a database or engineers
"designing" an existing database product.
""" + SHARED_RULES,
}


def render_generation_prompt(prompt_id: str, category_id: str, n: int) -> str:
    c = CATEGORIES[category_id]
    return PROMPTS[prompt_id].format(
        n=n,
        label=c["label"],
        role=c["role"],
        allowed_subjects="\n".join(f"- {x}" for x in c["allowed_subjects"]),
        allowed_relations="\n".join(f"- {x}" for x in c["allowed_relations"]),
        forbidden_subjects="\n".join(f"- {x}" for x in c["forbidden_subjects"]),
        forbidden_relations="\n".join(f"- {x}" for x in c["forbidden_relations"]),
    )


# ===========================================================================
# API CALLS
# ===========================================================================

def call_openai(client: OpenAI, model: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Write concise, natural software-engineering corpus "
                    "sentence frames. Follow the requested JSON exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text


def call_llama(client: Groq, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write concise, natural software-engineering corpus "
                    "sentence frames. Follow the requested JSON exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
    )
    return response.choices[0].message.content or ""


# ===========================================================================
# PARSING
# ===========================================================================

def parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])

    if not isinstance(value, list):
        raise ValueError("Expected JSON array.")
    return value


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("Expected JSON object.")
    return value


# ===========================================================================
# DETERMINISTIC VALIDATION
# ===========================================================================

ARTICLE_PATTERNS = [
    r"\ba\s+\{TECH\}",
    r"\ban\s+\{TECH\}",
    r"\bthe\s+\{TECH\}",
]

BANNED_CUES = [
    "because", "since", "due to", "for its", "enabling", "allowing",
    "so that", "in order to", "to improve", "to ensure", "to reduce",
    "to optimize", "scalable", "scalability", "secure", "security",
    "reliable", "reliability", "efficient", "efficiency", "low latency",
    "low-latency", "high throughput", "real-time", "serverless",
    "concurrent", "concurrency", "streaming", "caching", "cache",
    "fault tolerance", "memory safety", "multiplex", "offline",
    "high availability",
]


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def deterministic_validate(
    frame: str,
    category_id: str,
) -> tuple[bool, list[str], int]:
    errors: list[str] = []

    if frame.count("{TECH}") != 1:
        errors.append(f"tech_token_count_{frame.count('{TECH}')}")

    word_count = len(frame.split())
    if word_count < 5:
        errors.append("too_short")
    if word_count > 20:
        errors.append("too_long")

    for pattern in ARTICLE_PATTERNS:
        if re.search(pattern, frame, flags=re.I):
            errors.append("article_before_tech")
            break

    lower = frame.casefold()

    cue_hits = [cue for cue in BANNED_CUES if cue in lower]
    if cue_hits:
        errors.append("specificity_cue:" + ",".join(cue_hits))

    c = CATEGORIES[category_id]

    for bad in c["forbidden_subjects"]:
        if bad.casefold() in lower:
            errors.append("forbidden_subject:" + bad)

    # Remove placeholder to make relation matching more forgiving.
    for bad in c["forbidden_relations"]:
        cleaned_bad = bad.replace("{TECH}", "").strip().casefold()
        if cleaned_bad and cleaned_bad in lower:
            errors.append("forbidden_relation:" + bad)

    if ";" in frame:
        errors.append("semicolon")
    if ":" in frame:
        errors.append("colon")

    return len(errors) == 0, errors, word_count


def expand_frame(frame: str, technology: str) -> str:
    return frame.replace("{TECH}", technology)


# ===========================================================================
# SECOND-PASS LLM REVIEW
# ===========================================================================

def build_review_prompt(
    category_id: str,
    frame: str,
) -> str:
    c = CATEGORIES[category_id]
    expansions = [
        expand_frame(frame, tech)
        for tech in c["technologies"]
    ]

    rendered = "\n".join(
        f"- {sentence}" for sentence in expansions
    )

    return f"""
Review one software-engineering sentence frame.

Category: {c['label']}
Shared role: {c['role']}

Expanded sentences:
{rendered}

Judge the SET as a whole using only these criteria:

1. GRAMMAR:
   Every expanded sentence is grammatical.

2. NATURALNESS:
   Every expanded sentence sounds like ordinary software-engineering language,
   not a mechanically generated or awkward sentence.

3. SEMANTIC ROLE:
   The sentence subject and verb make sense for this technology category.

4. INTERCHANGEABILITY:
   No listed technology becomes clearly inappropriate because of the surrounding
   wording.

5. NEUTRALITY:
   The wording does not describe a distinctive advantage, feature, requirement,
   or property that favors one listed technology.

Be strict. For example:
- "The network uses PostgreSQL for application data" should FAIL naturalness /
  semantic role.
- "Engineers design PostgreSQL for the system" should FAIL semantic role.
- "The application uses PostgreSQL for data storage" can PASS.

Return ONLY a JSON object:
{{
  "pass": true,
  "grammar": true,
  "naturalness": true,
  "semantic_role": true,
  "interchangeability": true,
  "neutrality": true,
  "reason": "short explanation"
}}
"""


def review_frame(
    reviewer_provider: str,
    reviewer_model: str,
    openai_client: OpenAI,
    llama_client: Groq,
    category_id: str,
    frame: str,
) -> dict[str, Any]:

    prompt = build_review_prompt(category_id, frame)

    if reviewer_provider == "gpt":
        raw = call_openai(openai_client, reviewer_model, prompt)
    elif reviewer_provider == "llama":
        raw = call_llama(llama_client, reviewer_model, prompt)
    else:
        raise ValueError("reviewer_provider must be 'gpt' or 'llama'")

    result = parse_json_object(raw)

    fields = [
        "pass",
        "grammar",
        "naturalness",
        "semantic_role",
        "interchangeability",
        "neutrality",
    ]

    normalized: dict[str, Any] = {}
    for field in fields:
        normalized[field] = bool(result.get(field, False))

    normalized["reason"] = normalize(result.get("reason"))
    return normalized


# ===========================================================================
# MODEL CONFIGURATION
# ===========================================================================

MODEL_SPECS = {
    "gpt56": {
        "provider": "openai",
        "model": os.getenv("GPT56_MODEL", "gpt-5.6"),
    },
    "gpt4o": {
        "provider": "openai",
        "model": os.getenv("GPT4O_MODEL", "gpt-4o"),
    },
    "llama70b": {
        "provider": "openrouter",
        "model": os.getenv(
            "LLAMA70B_MODEL",
            "meta-llama/llama-3.3-70b-instruct",
        ),
    },
    "llama8b": {
        "provider": "openrouter",
        "model": os.getenv("LLAMA8B_MODEL", "meta-llama/llama-3.1-8b-instruct"),
    },
    "mistral": {
        "provider": "mistral",
        "model": os.getenv("MISTRAL_MODEL", "mistral-large-2512"),
    },
}


def call_openrouter(client: OpenAI, model: str, prompt: str) -> str:
    """Call Llama 3.1 8B through OpenRouter's OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Write concise, natural software-engineering corpus sentence frames. Follow the requested JSON exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
    )
    return response.choices[0].message.content or ""


def call_mistral(client: OpenAI, model: str, prompt: str) -> str:
    """Call Mistral through its OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write concise, natural software-engineering corpus "
                    "sentence frames. Follow the requested JSON exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
    )
    return response.choices[0].message.content or ""


def make_clients(selected_models: list[str]) -> dict[str, Any]:
    """Create only the API clients needed for the selected generation models."""
    load_dotenv()

    providers = {MODEL_SPECS[name]["provider"] for name in selected_models}
    clients: dict[str, Any] = {}

    if "openai" in providers:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for gpt5/gpt4o but is not set."
            )
        clients["openai"] = OpenAI(api_key=key)

    if "groq" in providers:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is required for llama70b but is not set."
            )
        clients["groq"] = Groq(api_key=key)

    if "openrouter" in providers:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for llama70b/llama8b but is not set.")
        clients["openrouter"] = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )

    if "mistral" in providers:
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError(
                "MISTRAL_API_KEY is required for mistral but is not set."
            )
        clients["mistral"] = OpenAI(
            api_key=key,
            base_url="https://api.mistral.ai/v1",
        )

    return clients


def get_caller(provider: str):
    if provider == "openai":
        return call_openai
    if provider == "groq":
        return call_llama
    if provider == "openrouter":
        return call_openrouter
    if provider == "mistral":
        return call_mistral
    raise ValueError(f"Unsupported provider: {provider}")


def provider_cooldown(provider: str) -> float:
    return 8.0 if provider == "mistral" else 0.2

def technical_backoff(provider: str, attempt: int) -> float:
    return 15.0 * (2 ** (attempt - 1)) if provider == "mistral" else float(attempt)

# ===========================================================================
# RUNNER
# ===========================================================================

def run(
    n: int,
    category_ids: list[str],
    selected_models: list[str],
    output_dir: Path,
) -> None:

    clients = make_clients(selected_models)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_id = "P3_strict_interchangeability"
    default_max_technical_attempts = 3

    frame_rows: list[dict[str, Any]] = []
    expansion_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for model_alias in selected_models:
        spec = MODEL_SPECS[model_alias]
        provider = spec["provider"]
        model = spec["model"]
        client = clients[provider]
        caller = get_caller(provider)

        max_technical_attempts = 5 if provider == "mistral" else default_max_technical_attempts
        for category_id in category_ids:
            c = CATEGORIES[category_id]
            prompt = render_generation_prompt(prompt_id, category_id, n)

            print(
                f"[generate] {model_alias} | {provider} | {model} | "
                f"{prompt_id} | {category_id} | n={n}"
            )

            items = None
            final_error = ""
            successful_attempt = None

            # Retry ONLY technical failures: API/network errors or malformed JSON.
            # The exact same prompt and requested n are used each time.
            for attempt in range(1, max_technical_attempts + 1):
                started = time.time()
                raw = ""

                try:
                    raw = caller(client, model, prompt)
                    latency = time.time() - started
                except Exception as exc:
                    latency = time.time() - started
                    final_error = f"api_error:{type(exc).__name__}: {exc}"

                    raw_rows.append({
                        "stage": "generation",
                        "model_alias": model_alias,
                        "provider": provider,
                        "model": model,
                        "prompt_id": prompt_id,
                        "category": category_id,
                        "attempt": attempt,
                        "requested_n": n,
                        "latency_seconds": latency,
                        "parse_success": False,
                        "raw_output": "",
                        "error": final_error,
                    })

                    print(
                        f"  attempt {attempt}/{max_technical_attempts}: "
                        f"{type(exc).__name__}; retrying"
                        if attempt < max_technical_attempts
                        else
                        f"  attempt {attempt}/{max_technical_attempts}: "
                        f"{type(exc).__name__}; giving up"
                    )

                    if attempt < max_technical_attempts:
                        wait = technical_backoff(provider, attempt)
                        print(f"  waiting {wait:.0f}s before retry")
                        time.sleep(wait)
                    continue

                try:
                    parsed = parse_json_array(raw)
                except Exception as exc:
                    final_error = f"json_parse_error:{type(exc).__name__}"

                    raw_rows.append({
                        "stage": "generation",
                        "model_alias": model_alias,
                        "provider": provider,
                        "model": model,
                        "prompt_id": prompt_id,
                        "category": category_id,
                        "attempt": attempt,
                        "requested_n": n,
                        "latency_seconds": latency,
                        "parse_success": False,
                        "raw_output": raw,
                        "error": final_error,
                    })

                    print(
                        f"  attempt {attempt}/{max_technical_attempts}: "
                        f"JSON parse error; retrying"
                        if attempt < max_technical_attempts
                        else
                        f"  attempt {attempt}/{max_technical_attempts}: "
                        f"JSON parse error; giving up"
                    )

                    if attempt < max_technical_attempts:
                        time.sleep(0.5)
                    continue

                # Successful technical response.
                items = parsed
                successful_attempt = attempt
                final_error = ""

                raw_rows.append({
                    "stage": "generation",
                    "model_alias": model_alias,
                    "provider": provider,
                    "model": model,
                    "prompt_id": prompt_id,
                    "category": category_id,
                    "attempt": attempt,
                    "requested_n": n,
                    "latency_seconds": latency,
                    "parse_success": True,
                    "raw_output": raw,
                    "error": "",
                })
                break

            # All three attempts failed technically. Record one failure row and move on.
            if items is None:
                frame_rows.append({
                    "frame_id": "",
                    "model_alias": model_alias,
                    "provider": provider,
                    "model": model,
                    "prompt_id": prompt_id,
                    "category": category_id,
                    "category_label": c["label"],
                    "frame_index": None,
                    "sentence_frame": "",
                    "word_count": None,
                    "auto_structural_valid": False,
                    "structural_errors": final_error or "technical_failure",
                    "auto_accept": False,
                    "generation_attempt": None,
                    "model_returned_count": None,
                    "saved_count": 0,
                    "human_natural": "",
                    "human_interchangeable": "",
                    "human_neutral": "",
                    "final_accept": "",
                })
                continue

            returned_count = len(items)

            # Models occasionally return > n despite "exactly n". For experimental
            # consistency, save at most the first n objects from the first response
            # that parses successfully. Do NOT request replacements for structural
            # failures or duplicates.
            items = items[:n]
            saved_count = len(items)

            if returned_count > n:
                print(
                    f"  model returned {returned_count}; "
                    f"keeping first {n} as requested"
                )
            elif returned_count < n:
                print(
                    f"  WARNING: model returned only {returned_count}/{n}; "
                    f"not regenerating missing items"
                )

            for i, item in enumerate(items, start=1):
                frame = (
                    normalize(item.get("sentence_frame"))
                    if isinstance(item, dict)
                    else ""
                )

                structural_valid, errors, word_count = (
                    deterministic_validate(frame, category_id)
                )

                frame_id = f"{model_alias}__{category_id}__{i}"

                frame_rows.append({
                    "frame_id": frame_id,
                    "model_alias": model_alias,
                    "provider": provider,
                    "model": model,
                    "prompt_id": prompt_id,
                    "category": category_id,
                    "category_label": c["label"],
                    "frame_index": i,
                    "sentence_frame": frame,
                    "word_count": word_count,
                    "auto_structural_valid": structural_valid,
                    "structural_errors": ";".join(errors),
                    "auto_accept": structural_valid,
                    "generation_attempt": successful_attempt,
                    "model_returned_count": returned_count,
                    "saved_count": saved_count,
                    # Intentionally blank human-review columns.
                    "human_natural": "",
                    "human_interchangeable": "",
                    "human_neutral": "",
                    "final_accept": "",
                })

                if frame.count("{TECH}") == 1:
                    for tech in c["technologies"]:
                        expansion_rows.append({
                            "frame_id": frame_id,
                            "model_alias": model_alias,
                            "provider": provider,
                            "model": model,
                            "prompt_id": prompt_id,
                            "category": category_id,
                            "technology": tech,
                            "sentence": expand_frame(frame, tech),
                            "auto_accept": structural_valid,
                        })

            print(
                f"  completed on attempt {successful_attempt}; "
                f"saved {saved_count}/{n}"
            )
            cooldown = provider_cooldown(provider)
            if cooldown:
                print(f"  cooldown {cooldown:.0f}s before next category")
                time.sleep(cooldown)

    frames = pd.DataFrame(frame_rows)
    expansions = pd.DataFrame(expansion_rows)

    if len(frames):
        lowered = frames["sentence_frame"].fillna("").str.casefold()
        counts = lowered.value_counts().to_dict()
        frames["exact_duplicate"] = lowered.map(
            lambda x: bool(x.strip()) and counts.get(x, 0) > 1
        )

    run_tag = "__".join(selected_models)
    frames_path = output_dir / f"sentence_frames__{run_tag}.csv"
    expansions_path = output_dir / f"all_technology_expansions__{run_tag}.csv"
    raw_path = output_dir / f"raw_generations__{run_tag}.jsonl"
    summary_path = output_dir / f"model_summary__{run_tag}.csv"

    frames.to_csv(frames_path, index=False)
    expansions.to_csv(expansions_path, index=False)

    with raw_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(frames):
        summary = (
            frames.groupby(
                ["model_alias", "provider", "model", "prompt_id"],
                dropna=False,
            )
            .agg(
                rows=("sentence_frame", "size"),
                structural_passes=("auto_structural_valid", "sum"),
                structural_pass_rate=("auto_structural_valid", "mean"),
                auto_accepts=("auto_accept", "sum"),
                auto_accept_rate=("auto_accept", "mean"),
                mean_word_count=("word_count", "mean"),
                exact_duplicate_rate=("exact_duplicate", "mean"),
                categories_covered=("category", "nunique"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(summary_path, index=False)

    print("\nFinished.")
    print(f"Frames:      {frames_path}")
    print(f"Expansions:  {expansions_path}")
    print(f"Raw:         {raw_path}")
    print(f"Summary:     {summary_path}")

    if len(summary):
        print("\nModel summary:")
        print(summary.to_string(index=False))


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate technology-neutral SE sentence frames using Prompt 3. "
            "Run one or more generation models with --models."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Frames requested per category/model. Default: 10.",
    )

    parser.add_argument(
        "--categories",
        nargs="*",
        choices=list(CATEGORIES),
        help="Optional category subset.",
    )

    parser.add_argument(
        "--osi-only",
        action="store_true",
        help="Generate only L1-L7 categories.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help=(
            "Generation model(s). Choices: "
            + ", ".join(MODEL_SPECS)
            + ". Default: all."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs_v11",
        help="Directory for CSV/JSONL outputs. Default: outputs_v11.",
    )

    args = parser.parse_args()

    if args.categories:
        category_ids = args.categories
    elif args.osi_only:
        category_ids = [key for key in CATEGORIES if key.startswith("L")]
    else:
        category_ids = list(CATEGORIES)

    run(
        n=args.n,
        category_ids=category_ids,
        selected_models=args.models,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
