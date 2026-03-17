import re, json, os
  from pathlib import Path
  from tqdm import tqdm
  
  DATASET_DIR = Path('data/dataset')
  
  # 15 financial fields with regex patterns
  PATTERNS = {
      'total_revenue': [
          r'(?:total\s+)?(?:net\s+)?revenues?\s*[\$]?\s*([\d,]+(?:\.\d+)?)',
          r'net\s+sales\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'net_income': [
          r'net\s+(?:income|earnings)\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'gross_profit': [
          r'gross\s+profit\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'operating_income': [
          r'(?:operating\s+)?income\s+from\s+operations\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'eps_basic': [
          r'basic\s+.*?earnings\s+per\s+share\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'eps_diluted': [
          r'diluted\s+.*?earnings\s+per\s+share\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'total_assets': [
          r'total\s+assets\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'total_liabilities': [
          r'total\s+liabilities\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'total_equity': [
          r'total\s+(?:stockholders|shareholders).{0,10}equity\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'long_term_debt': [
          r'long[- ]term\s+debt\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'cash_and_equivalents': [
          r'cash\s+and\s+cash\s+equivalents\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'operating_cash_flow': [
          r'(?:net\s+)?cash\s+(?:provided\s+by|from)\s+operating\s+activities\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'capital_expenditures': [
          r'capital\s+expenditures?\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'research_and_development': [
          r'research\s+and\s+development\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
      'dividend_per_share': [
          r'dividends?\s+(?:declared\s+)?per\s+(?:common\s+)?share\s*[\$]?\s*([\d,]+(?:\.\d+)?)'],
  }
  
  def extract_field(text, field):
      for pattern in PATTERNS.get(field, []):
          m = re.search(pattern, text.lower())
          if m:
              try: return float(m.group(1).replace(',', ''))
              except: pass
      return None
  
  def extract_from_doc(doc):
      text = doc.get('text', '')[:60000]  # Use first 60k chars
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
      with open(docs_path) as f:
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