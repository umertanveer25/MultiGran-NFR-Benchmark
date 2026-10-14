import pandas as pd

df_exp = pd.read_csv(r'B:\Dataset\Promise_Exp.csv', encoding='latin-1')
df_nfr = pd.read_csv(r'B:\Dataset\nfr.csv', encoding='latin-1')
df_fnfc = pd.read_csv(r'B:\Dataset\FNFC.csv', encoding='latin-1')

print(f"1. Promise_Exp.csv: {len(df_exp)} rows, cols: {df_exp.columns.tolist()}")
print(f"2. nfr.csv:         {len(df_nfr)} rows, cols: {df_nfr.columns.tolist()}")
print(f"3. FNFC.csv:        {len(df_fnfc)} rows, cols: {df_fnfc.columns.tolist()}")

t_exp = set(df_exp['Requirement'].astype(str).str.strip().str.lower())
t_nfr = set(df_nfr['RequirementText'].astype(str).str.strip().str.strip("'").str.strip().str.lower())
t_fnfc = set(df_fnfc['text'].astype(str).str.strip().str.lower())

print("\nOverlap Analysis:")
print(f" - Overlap between Promise_Exp and nfr: {len(t_exp.intersection(t_nfr))} requirements")
print(f" - Overlap between FNFC and Promise_Exp: {len(t_fnfc.intersection(t_exp))} requirements")
print(f" - Overlap between FNFC and nfr:         {len(t_fnfc.intersection(t_nfr))} requirements")

# Combined unique requirements across all 3
all_unique = t_exp.union(t_nfr).union(t_fnfc)
print(f"\nTotal UNIQUE authentic requirements across all 3 datasets: {len(all_unique)} requirements!")
