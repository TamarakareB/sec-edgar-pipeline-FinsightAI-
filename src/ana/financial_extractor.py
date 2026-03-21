import re, json, os
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path('data/dataset')
  
# 15 financial fields with regex patterns
PATTERNS = {
    'total_revenue': [
        r'net\s+sales',
        r'total\s+net\s+sales',
        r'total\s+revenue',
    ],
    'net_income': [
        r'net\s+income',
        r'net\s+earnings',
    ],
    'total_assets': [
        r'total\s+assets',
    ],
    'total_liabilities': [
        r'total\s+liabilities',
    ],
    'cash_and_equivalents': [
        r'cash\s+and\s+cash\s+equivalents',
    ]
}
  
def extract_field(text, field):
    text = text.lower()

    for pattern in PATTERNS.get(field, []):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            snippet = text[m.start(): m.start() + 300]

            num_match = re.search(r'[\$]?\s*([\d,]+(?:\.\d+)?)', snippet)
            if num_match:
                try:
                    return float(num_match.group(1).replace(',', ''))
                except:
                    pass
    return None
  
def extract_from_doc(doc):
    text = doc.get('text', '').lower()


    row = {
        'ticker':      doc.get('ticker'),
        'company':     doc.get('company'),
        'form':        doc.get('form'),
        'filing_date': doc.get('filing_date'),
        'year':        int(doc['filing_date'][:4]) if doc.get('filing_date') else None,
        'accession':   doc.get('accession_number'),
    }
    found = 0
    for field in PATTERNS:
        val = extract_field(text, field)
        row[field] = val
        if val is not None: found += 1
    row['fields_found'] = found
    row['confidence']   = round(found / len(PATTERNS), 2)
    return row
  
def run():
    docs_path = DATASET_DIR / 'edgar_docs.jsonl'
    if not docs_path.exists():
        print('ERROR: Run Person 1 pipeline first!')
        return []
    results = []
    with open(docs_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    print(f'Processing {len(lines)} documents...')
    for line in tqdm(lines):
        doc = json.loads(line)
        results.append(extract_from_doc(doc))
    out = DATASET_DIR / 'financial_metrics.jsonl'
    with open(out, 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')
    print(f'Saved {len(results)} records to {out}')
    # Print summary
    for field in PATTERNS:
        n = sum(1 for r in results if r.get(field) is not None)
        pct = round(n/len(results)*100, 1) if results else 0
        print(f'  {field:<30} {pct}% ({n}/{len(results)})')
    return results
  
if __name__ == '__main__': run()