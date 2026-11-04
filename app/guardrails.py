import re

_INJECTION_PATTERNS = [
    re.compile(r'\bignore\b.{0,50}\b(previous|all|above)\b.{0,30}\b(instructions|rules|prompts)\b', re.I),
    re.compile(r'\b(reveal|show|print|dump|exfiltrate)\b.{0,50}\b(confidential|secret|system prompt|restricted|private)\b', re.I),
    re.compile(r'\b(bypass|override|disable)\b.{0,40}\b(access control|authorization|security|policy)\b', re.I),
]
_PII_PATTERNS = {
    'email': re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I),
    'phone': re.compile(r'(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'employee_id': re.compile(r'\bNS-TEST-\d{4}\b', re.I),
}

def is_prompt_injection(text):
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)

def detect_pii(text):
    return sorted(name for name, pattern in _PII_PATTERNS.items() if pattern.search(text))
