import os
import pandas as pd

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1].strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s

def prepare_benchmark():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    exp_path = os.path.join(data_dir, "Promise_Exp.csv")
    nfr_path = os.path.join(data_dir, "nfr.csv")
    fnfc_path = os.path.join(data_dir, "FNFC.csv")

    exp = pd.read_csv(exp_path, encoding='latin-1')
    nfr = pd.read_csv(nfr_path, encoding='latin-1')
    fnfc = pd.read_csv(fnfc_path, encoding='latin-1')

    df1 = pd.DataFrame({
        'requirement': exp['Requirement'].apply(clean_text),
        'category': exp['Type'].str.strip(),
        'source': 'PROMISE_exp'
    })

    df2 = pd.DataFrame({
        'requirement': nfr['RequirementText'].apply(clean_text),
        'category': nfr['class'].str.strip(),
        'source': 'PROMISE_nfr'
    })

    df3 = pd.DataFrame({
        'requirement': fnfc['text'].apply(clean_text),
        'category': fnfc['class'].str.strip(),
        'source': 'FNFC'
    })

    combined = pd.concat([df1, df2, df3], ignore_index=True)
    # Filter empty or trivial texts
    combined = combined[combined['requirement'].str.len() > 5]
    
    # Deduplicate based on lowercase requirement text
    combined['norm_text'] = combined['requirement'].str.lower()
    dedup = combined.drop_duplicates(subset=['norm_text']).drop(columns=['norm_text']).reset_index(drop=True)

    print(f"Loaded PROMISE_exp: {len(df1)} rows")
    print(f"Loaded PROMISE_nfr: {len(df2)} rows")
    print(f"Loaded FNFC:        {len(df3)} rows")
    print(f"Total raw:          {len(combined)} rows")
    print(f"Total deduplicated: {len(dedup)} unique authentic requirements")
    print("\nClass distribution (deduplicated):")
    print(dedup['category'].value_counts())

    # Save to data/promise_fnfc_benchmark.csv
    out_path = os.path.join(data_dir, "promise_fnfc_benchmark.csv")
    dedup.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\nSaved unified authentic corpus to: {out_path}")

if __name__ == "__main__":
    prepare_benchmark()
