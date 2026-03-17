import re
from dataclasses import dataclass

# Official SEC 10-K section titles mapped to clean labels
SECTION_PATTERNS = {
    "business_overview": [
        r"item\s*1[^a-z].*business",
        r"overview of.*business",
        r"our\s+business",
    ],
    "risk_factors": [
        r"item\s*1a[^a-z]",
        r"risk\s+factors",
        r"risks\s+related\s+to",
        r"factors\s+that\s+(?:may|could)\s+affect",
    ],
    "legal_proceedings": [
        r"item\s*3[^a-z]",
        r"legal\s+proceedings",
        r"litigation",
    ],
    "mda": [  # Management Discussion & Analysis — goldmine for financial insights
        r"item\s*7[^a-z]",
        r"management.{0,10}discussion",
        r"management.{0,10}analysis",
        r"results\s+of\s+operations",
        r"liquidity\s+and\s+capital",
    ],
    "financial_statements": [
        r"item\s*8[^a-z]",
        r"financial\s+statements",
        r"consolidated\s+balance\s+sheet",
        r"consolidated\s+statements?\s+of\s+(?:income|operations|earnings)",
        r"consolidated\s+statements?\s+of\s+cash",
    ],
    "controls_procedures": [
        r"item\s*9[^a-z]",
        r"controls\s+and\s+procedures",
        r"internal\s+control",
    ],
    "executive_compensation": [
        r"item\s*11[^a-z]",
        r"executive\s+compensation",
        r"compensation\s+discussion",
    ],
    "market_data": [
        r"item\s*5[^a-z]",
        r"market\s+for\s+(?:registrant|common)",
        r"stock\s+price",
        r"dividend",
    ],
    "quantitative_disclosures": [
        r"item\s*7a[^a-z]",
        r"quantitative.*qualitative.*market\s+risk",
        r"market\s+risk\s+disclosures",
    ],
}

# Human-readable labels for display
SECTION_LABELS = {
    "business_overview":      "Business Overview",
    "risk_factors":           "Risk Factors",
    "legal_proceedings":      "Legal Proceedings",
    "mda":                    "MD&A",
    "financial_statements":   "Financial Statements",
    "controls_procedures":    "Controls & Procedures",
    "executive_compensation": "Executive Compensation",
    "market_data":            "Market Data",
    "quantitative_disclosures": "Market Risk Disclosures",
    "unknown":                "General",
}

# Which sections are most relevant for which query types
QUERY_SECTION_AFFINITY = {
    "risk":          ["risk_factors", "mda", "quantitative_disclosures"],
    "revenue":       ["mda", "financial_statements"],
    "income":        ["financial_statements", "mda"],
    "debt":          ["mda", "financial_statements"],
    "cash":          ["financial_statements", "mda"],
    "competition":   ["business_overview", "risk_factors"],
    "regulation":    ["risk_factors", "legal_proceedings"],
    "compensation":  ["executive_compensation"],
    "market":        ["market_data", "quantitative_disclosures", "risk_factors"],
    "growth":        ["mda", "business_overview"],
    "outlook":       ["mda", "business_overview"],
}


def detect_section(text: str) -> str:
    """
    Detects which 10-K section a chunk of text belongs to.
    Returns a section key like 'risk_factors' or 'mda'.
    """
    text_lower = text.lower()[:500]  # Check first 500 chars (section headers appear early)

    for section_key, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return section_key

    return "unknown"


def get_section_label(section_key: str) -> str:
    """Returns human-readable section name."""
    return SECTION_LABELS.get(section_key, "General")


def get_priority_sections(query: str) -> list[str]:
    """
    Given a user query, returns which sections to prioritize in retrieval.
    This boosts retrieval precision dramatically.
    """
    query_lower = query.lower()
    priority = []

    for keyword, sections in QUERY_SECTION_AFFINITY.items():
        if keyword in query_lower:
            for s in sections:
                if s not in priority:
                    priority.append(s)

    return priority if priority else list(SECTION_PATTERNS.keys())


def label_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds section labels to a list of chunk dicts.
    Call this after chunking in the pipeline.
    """
    section_counts = {}

    for chunk in chunks:
        section = detect_section(chunk.get("text", ""))
        chunk["section"]       = section
        chunk["section_label"] = get_section_label(section)
        section_counts[section] = section_counts.get(section, 0) + 1

    print(f"Section distribution across {len(chunks)} chunks:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        pct = round(count / len(chunks) * 100, 1)
        print(f"  {SECTION_LABELS.get(section, section):<30} {count:>5} chunks ({pct}%)")

    return chunks
