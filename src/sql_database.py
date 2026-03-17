import sqlite3, json
from pathlib import Path

DB_PATH      = Path('data/sql/finsightai.db')
METRICS_PATH = Path('data/dataset/financial_metrics.jsonl')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDS = ['total_revenue','net_income','gross_profit','operating_income',
          'eps_basic','eps_diluted','total_assets','total_liabilities',
          'total_equity','long_term_debt','cash_and_equivalents',
          'operating_cash_flow','capital_expenditures',
          'research_and_development','dividend_per_share']

def create_tables(conn):
    cur = conn.cursor()
    field_cols = '\n'.join([f'    {f} REAL,' for f in FIELDS])
    cur.execute(f'''CREATE TABLE IF NOT EXISTS financial_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, company TEXT, form TEXT,
        filing_date TEXT, year INTEGER, accession TEXT,
        {field_cols}
        fields_found INTEGER, confidence REAL,
        UNIQUE(ticker, filing_date, form))
    ''')
    cur.execute('''CREATE TABLE IF NOT EXISTS companies (
        ticker TEXT PRIMARY KEY, company TEXT, sector TEXT)''')
    cur.execute('''CREATE INDEX IF NOT EXISTS idx_ticker ON financial_metrics(ticker)''')
    cur.execute('''CREATE INDEX IF NOT EXISTS idx_year   ON financial_metrics(year)''')
    conn.commit()
    print('Tables created')
  
def load_metrics(conn):
    if not METRICS_PATH.exists():
        print('Run financial_extractor.py first!')
        return 0
    cur  = conn.cursor()
    cols = ', '.join(FIELDS)
    vals = ', '.join(['?' for _ in FIELDS])
    loaded = 0
    with open(METRICS_PATH) as f:
        for line in f:
            r = json.loads(line)
            field_vals = [r.get(field) for field in FIELDS]
            try:
                cur.execute(f'''INSERT OR REPLACE INTO financial_metrics
                    (ticker,company,form,filing_date,year,accession,{cols},fields_found,confidence)
                    VALUES (?,?,?,?,?,?,{vals},?,?)''',
                    [r.get('ticker'),r.get('company'),r.get('form'),
                     r.get('filing_date'),r.get('year'),r.get('accession')]
                    + field_vals + [r.get('fields_found'), r.get('confidence')])
                loaded += 1
            except Exception as e: print(f'Skip: {e}')
    conn.commit()
    print(f'Loaded {loaded} records')
    return loaded
  
def sample_queries(conn):
    cur = conn.cursor()
    print('\nTop 5 by Revenue (most recent 10-K):')
    cur.execute('''SELECT ticker, year, total_revenue FROM financial_metrics
        WHERE form='10-K' AND total_revenue IS NOT NULL
        ORDER BY total_revenue DESC LIMIT 5''')
    for row in cur.fetchall(): print(f'  {row[0]} ({row[1]}): ${row[2]:,.0f}M')

    print('\nApple metrics over time:')
    cur.execute('''SELECT year, total_revenue, net_income, eps_diluted
        FROM financial_metrics WHERE ticker='AAPL' AND form='10-K'
        ORDER BY year''')
    for row in cur.fetchall():
        print(f'  {row[0]}: Revenue=${row[1]:,.0f}M  NI=${row[2]:,.0f}M  EPS=${row[3]}')
  
if __name__ == '__main__':
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    load_metrics(conn)
    sample_queries(conn)
    conn.close()
    print(f'\nDatabase saved: {DB_PATH}')
    print('Open with DB Browser for SQLite: https://sqlitebrowser.org')
