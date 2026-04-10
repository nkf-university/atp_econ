"""
The Electoral Cost of the Prime-Ministerial Office in OECD Parliamentary Democracies:
Evidence from 20 Countries, 1990–2020
═══════════════════════════════════════════════════════════════════════════════════════

RESEARCH QUESTION:
  How large is the electoral cost borne specifically by PM parties in modern OECD
  parliamentary democracies, and does it differ from the all-party benchmark of
  −2.25 pp (Nannestad & Paldam, 2002)?

MEASUREMENT APPROACH — N&P-COMPARABLE DESIGN:
  Ce = Ve − Le-1   for the INCUMBENT PM party at each election e.

  Following Nannestad & Paldam (2002):
    - Ge = the party holding the PM role going INTO election e (pre-election incumbent)
    - Ve  = Ge's vote share AT election e
    - Le-1 = Ge's vote share AT the PREVIOUS election (e-1)
    - Ce = Ve − Le-1

  CRITICAL DESIGN CHOICE — COMPARABILITY WITH N&P:
    N&P compute Ce for ALL governments at ALL elections, regardless of whether
    the incumbent won or lost. This paper follows the same logic for PM parties:
    Ce is measured for every PM party that faces a re-election bid, including
    those that subsequently lost power.

    Prior design (winner-only) only observed PM parties at elections they WON,
    creating an upward selection bias: losers (who face more negative Ce) were
    excluded, pulling the estimated cost artificially upward (less negative).

  TERM COUNTER (CT):
    CT = 1: Party just came to power (not an incumbency observation — excluded).
    CT = 2: First re-election bid. Has served one full term.
    CT ≥ 3: Subsequent re-election bids.
    Only CT ≥ 2 observations enter the analytical sample.

HYPOTHESES:
  H1: The PM incumbency cost is negative (Ce < 0 on average).
  H2: The PM incumbency cost is more negative than the N&P all-party benchmark
      of −2.25 pp (PM parties face a LARGER cost than the average government party).
  H3: The negative effect is geographically consistent (majority of countries).

DATA: Döring, H., & Manow, P. (2024). ParlGov 2024 Release.
      https://doi.org/10.7910/DVN/2VZ5ZC
"""

# ════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────
# UPDATE THIS PATH before running:
DATA_PATH = "/Users/nkf/Library/CloudStorage/OneDrive-UniversitaetSt.Gallen/HSG assessment/ATP/dataverse_files"

ANALYSIS_START = 1990
ANALYSIS_END   = 2020

OECD_COUNTRIES = [
    "Australia", "Austria", "Belgium", "Canada", "Denmark",
    "Finland", "France", "Germany", "Greece", "Iceland",
    "Ireland", "Italy", "Japan", "Luxembourg", "Netherlands",
    "New Zealand", "Norway", "Portugal", "Spain", "Sweden",
    "United Kingdom",
]

# Nannestad & Paldam (2002) cross-party benchmark
BENCHMARK = -2.25

B = 10_000
np.random.seed(42)
ALPHA = 0.05


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return "n.s."

def p_one_less(t, p2):
    """One-sided p-value for H0: μ ≥ x  vs  HA: μ < x  (cost is negative / more negative)."""
    return p2 / 2 if t < 0 else 1 - p2 / 2

def bootstrap_ci(data, n_boot=B):
    data = np.asarray(data)
    means = [np.random.choice(data, size=len(data), replace=True).mean()
             for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def get_vote_share(country, party_id, edate, votes_df):
    """Return party vote share at a given election date."""
    m = votes_df[
        (votes_df["country_name"] == country) &
        (votes_df["party_id"]     == party_id) &
        (votes_df["election_date"] == edate)
    ]
    return m["vote_share"].iloc[0] if len(m) > 0 else np.nan


# ════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════

elections = pd.read_csv(os.path.join(DATA_PATH, "view_election.csv"))
cabinets  = pd.read_csv(os.path.join(DATA_PATH, "view_cabinet.csv"))

elections["election_date"] = pd.to_datetime(elections["election_date"])
elections["election_year"] = elections["election_date"].dt.year
cabinets["election_date"]  = pd.to_datetime(cabinets["election_date"])


# ════════════════════════════════════════════════════════════════
# STEP 1: BUILD FULL ELECTORAL HISTORY (1945–2020)
# ════════════════════════════════════════════════════════════════
# History back to 1945 is required to:
# (a) compute Ce for first-period elections (need Le-1 from before 1990), and
# (b) correctly initialise the CT counter.

df_all = elections[
    (elections["country_name"].isin(OECD_COUNTRIES)) &
    (elections["election_year"] >= 1945) &
    (elections["election_year"] <= ANALYSIS_END) &
    (elections["election_type"] == "parliament")
].copy()


# ── Germany CDU+CSU fix ──────────────────────────────────────────────────────
# ParlGov cabinet data records the PM party for CDU/CSU governments as the
# unified bloc CDU+CSU (party_id = 1727). In the election table the two
# parties appear separately (CDU = 808, CSU = 1180). This fix combines their
# vote shares into a synthetic CDU+CSU row (party_id = 1727), matching the
# cabinet table's identifier and producing the standard Bundestag combined share.

de_cdu_csu = (
    df_all[
        (df_all["country_name"] == "Germany") &
        (df_all["party_name_short"].isin(["CDU", "CSU"]))
    ]
    .groupby(["country_name", "election_date", "election_year"],
             as_index=False)["vote_share"]
    .sum()
    .assign(
        party_id=1727,
        party_name_short="CDU+CSU",
        party_name="CDU/CSU Union",
        party_name_english="CDU/CSU Union",
        election_type="parliament",
    )
)
df_all = pd.concat([df_all, de_cdu_csu], ignore_index=True)
# ─────────────────────────────────────────────────────────────────────────────

# All vote records (used for both Ce and Le-1 lookups)
all_votes_df = df_all[["country_name", "election_date", "party_id",
                        "vote_share"]].drop_duplicates()

# All parliamentary election dates by country (for sequencing)
all_edates = (
    df_all[["country_name", "election_date"]]
    .drop_duplicates()
    .sort_values(["country_name", "election_date"])
)


# ════════════════════════════════════════════════════════════════
# STEP 2: IDENTIFY PM WINNER AT EACH ELECTION
# ════════════════════════════════════════════════════════════════
# pm_winner[e] = party_id of the PM after election e
# (i.e., the party that FORMED THE NEXT GOVERNMENT after election e)

pm_winner = (
    cabinets[cabinets["prime_minister"] == 1]
    [["country_name", "election_date", "party_id", "party_name_english"]]
    .drop_duplicates(subset=["country_name", "election_date"])
    .rename(columns={"party_id":           "winner_pm_id",
                     "party_name_english":  "winner_pm_name"})
)
pm_winner["election_date"] = pd.to_datetime(pm_winner["election_date"])

# Germany fix: map CDU (808) and CSU (1180) cabinet entries → CDU+CSU (1727)
de_mask = (
    (pm_winner["country_name"] == "Germany") &
    (pm_winner["winner_pm_id"].isin([808, 1180]))
)
pm_winner.loc[de_mask, "winner_pm_id"] = 1727


# ════════════════════════════════════════════════════════════════
# STEP 3: BUILD N&P-COMPARABLE PM INCUMBENCY DATASET
# ════════════════════════════════════════════════════════════════
# For each election e:
#   INCUMBENT = the party that won the PM role at election e-1
#   Ce = Ve - Le-1 (change in incumbent PM party's vote share)
#   Include regardless of whether the incumbent WON or LOST
#
# CT counter tracks consecutive terms held by the incumbent party.
# CT=1 (first-time winner) observations are excluded — these are not
# incumbency elections; they are entry-to-power observations.

rows = []

for country in OECD_COUNTRIES:
    c_edates = (
        all_edates[all_edates["country_name"] == country]["election_date"]
        .sort_values().tolist()
    )
    c_winners = (
        pm_winner[pm_winner["country_name"] == country]
        .set_index("election_date")
    )

    prev_pm_id   = None
    ct           = 0            # consecutive terms for the current PM party

    for i, edate in enumerate(c_edates):
        if i == 0:
            # Very first election in history: record winner, start chain
            w = (c_winners.loc[edate, "winner_pm_id"]
                 if edate in c_winners.index else np.nan)
            prev_pm_id = w
            ct = 1
            continue

        prev_edate     = c_edates[i - 1]
        incumbent_pm_id = prev_pm_id

        if pd.isna(incumbent_pm_id):
            # No incumbent identified — skip and reset
            w = (c_winners.loc[edate, "winner_pm_id"]
                 if edate in c_winners.index else np.nan)
            prev_pm_id = w
            ct = 1
            continue

        # Ce = Ve − Le-1
        Ve  = get_vote_share(country, incumbent_pm_id, edate,      all_votes_df)
        Le1 = get_vote_share(country, incumbent_pm_id, prev_edate, all_votes_df)

        # Determine whether incumbent retained PM after this election
        new_winner_id = (c_winners.loc[edate, "winner_pm_id"]
                         if edate in c_winners.index else np.nan)
        # Apply Germany CDU+CSU fix for the new winner lookup
        if country == "Germany" and not pd.isna(new_winner_id) and new_winner_id in [808, 1180]:
            new_winner_id = 1727

        won = (not pd.isna(new_winner_id)) and (new_winner_id == incumbent_pm_id)

        # Record if both vote shares are available
        if not (pd.isna(Ve) or pd.isna(Le1)):
            rows.append({
                "country_name":    country,
                "election_date":   edate,
                "election_year":   edate.year,
                "incumbent_pm_id": incumbent_pm_id,
                "ct":              ct,           # ≥2 means true incumbency bid
                "Ve":              Ve,
                "Le1":             Le1,
                "Ce":              Ve - Le1,
                "won":             won,
            })

        # Update chain for next election
        if won:
            ct += 1
            prev_pm_id = incumbent_pm_id
        else:
            # New party took power; reset
            prev_pm_id = new_winner_id if not pd.isna(new_winner_id) else np.nan
            ct = 1

full_df = pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# STEP 4: FILTER — ANALYSIS PERIOD + INCUMBENCY ONLY (CT ≥ 2)
# ════════════════════════════════════════════════════════════════
# CT=1 observations (party just came to power) are excluded.
# They are entry-to-power events, not incumbency elections.

analysis = full_df[
    (full_df["election_year"] >= ANALYSIS_START) &
    (full_df["election_year"] <= ANALYSIS_END) &
    (full_df["ct"] >= 2)
].copy()

# Also keep CT=1 in analysis window for reference counts
ct1_in_window = full_df[
    (full_df["election_year"] >= ANALYSIS_START) &
    (full_df["election_year"] <= ANALYSIS_END) &
    (full_df["ct"] == 1)
].copy()

# Convenience subsets
all_inc  = analysis["Ce"]
won_df   = analysis[analysis["won"]]
lost_df  = analysis[~analysis["won"]]
t1_df    = analysis[analysis["ct"] == 2]
t2p_df   = analysis[analysis["ct"] >= 3]

won_ce   = won_df["Ce"]
lost_ce  = lost_df["Ce"]
t1_ce    = t1_df["Ce"]
t2p_ce   = t2p_df["Ce"]


# ════════════════════════════════════════════════════════════════
# DESCRIPTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════

print("\n" + "█" * 70)
print("  PM INCUMBENCY COST — N&P-COMPARABLE DESIGN")
print(f"  All PM Re-Election Bids (CT≥2), {ANALYSIS_START}–{ANALYSIS_END}")
print("█" * 70)

print(f"""
  SAMPLE:
  ────────────────────────────────────────────────────────────────
    Total PM incumbency bids (CT≥2):    {len(analysis):>5}
      Winners (retained PM):            {len(won_df):>5}
      Losers  (lost PM):                {len(lost_df):>5}
    ─────────────────────────────────────────────
      Term 1 (CT=2, 1st re-election):   {len(t1_df):>5}
      Term 2+ (CT≥3, long tenure):      {len(t2p_df):>5}
    Countries with incumbency obs:      {analysis['country_name'].nunique():>5}
    CT=1 (entry-to-power, excluded):    {len(ct1_in_window):>5}
    Year range:                         {analysis['election_year'].min()}–{analysis['election_year'].max()}

  [N&P-comparable design: Ce measured at ALL re-election bids, including
   elections where the PM party subsequently lost power. CT=1 (first-time
   winners) are excluded as non-incumbency observations.]
""")

print("  Descriptive Statistics:")
print(f"  {'Group':<35} {'N':>5} {'Mean':>8} {'Median':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
print("  " + "─" * 80)
for label, data in [
    ("All PM bids (CT≥2)",           all_inc),
    ("  Term 1 (1st re-elec, CT=2)", t1_ce),
    ("  Term 2+ (long tenure, CT≥3)", t2p_ce),
    ("  Winners (retained PM)",       won_ce),
    ("  Losers  (lost PM)",           lost_ce),
]:
    print(f"  {label:<35} {len(data):>5} {data.mean():>+8.3f} {data.median():>+8.3f} "
          f"{data.std():>8.3f} {data.min():>+8.3f} {data.max():>+8.3f}")
print(f"\n  Nannestad & Paldam (2002) benchmark: {BENCHMARK:+.2f} pp")
print(f"  PM mean vs. benchmark:               {all_inc.mean() - BENCHMARK:+.3f} pp")

# Country-level summary
print("\n  Country contributions:")
print(f"  {'Country':<22} {'CT=1(excl)':>10} {'N_all':>6} {'N_won':>6} {'N_lost':>6} "
      f"{'MeanCe':>8} {'Mean(won)':>10}")
print("  " + "─" * 72)

for country in sorted(OECD_COUNTRIES):
    n_ct1 = len(ct1_in_window[ct1_in_window["country_name"] == country])
    sub   = analysis[analysis["country_name"] == country]
    sub_w = sub[sub["won"]]
    sub_l = sub[~sub["won"]]
    if len(sub) == 0:
        print(f"  {country:<22} {n_ct1:>10} {'0':>6} {'0':>6} {'0':>6} {'n/a':>8}")
        continue
    mw = f"{sub_w['Ce'].mean():+.2f}" if len(sub_w) > 0 else "n/a"
    print(f"  {country:<22} {n_ct1:>10} {len(sub):>6} {len(sub_w):>6} {len(sub_l):>6} "
          f"{sub['Ce'].mean():>+8.2f} {mw:>10}")


# ════════════════════════════════════════════════════════════════
#  ██  H1: IS THE PM INCUMBENCY COST NEGATIVE?
# ════════════════════════════════════════════════════════════════
section("H1: IS THE PM INCUMBENCY COST NEGATIVE? (vs. zero)")

t_h1, p_h1_2s = stats.ttest_1samp(all_inc, 0)
p_h1_1s = p_one_less(t_h1, p_h1_2s)
se_h1   = all_inc.std() / np.sqrt(len(all_inc))
ci_h1   = stats.t.interval(0.95, df=len(all_inc)-1, loc=all_inc.mean(), scale=se_h1)
ci_lo_boot, ci_hi_boot = bootstrap_ci(all_inc.values)

print(f"""
  H1: Mean Ce for PM parties at re-election < 0
  ──────────────────────────────────────────────────
  N (incumbency elections):  {len(all_inc)}
  Mean Ce:                   {all_inc.mean():+.3f} pp
  Median Ce:                 {all_inc.median():+.3f} pp
  SD:                        {all_inc.std():.3f} pp
  SE:                        {se_h1:.3f}

  One-sample t-test (H0: μ = 0):
    t = {t_h1:.3f},  df = {len(all_inc)-1}
    p (2-sided)       = {p_h1_2s:.5f}
    p (1-sided, less) = {p_h1_1s:.5f}  {stars(p_h1_1s)}
  95% CI (t-based):   [{ci_h1[0]:+.3f}, {ci_h1[1]:+.3f}]
  95% CI (bootstrap): [{ci_lo_boot:+.3f}, {ci_hi_boot:+.3f}]

  → H1 {'CONFIRMED ✓' if p_h1_1s < ALPHA else 'NOT CONFIRMED at α=5%'}
    PM parties lose an average of {all_inc.mean():.2f} pp per re-election bid.
""")

# Term-specific
print("  Term-specific breakdown:")
for label, d in [("Term 1 (CT=2, first re-elec)", t1_ce),
                  ("Term 2+ (CT≥3, long tenure)", t2p_ce)]:
    tt, pp = stats.ttest_1samp(d, 0)
    pp1 = p_one_less(tt, pp)
    print(f"    {label}: mean = {d.mean():+.3f} pp,  t = {tt:.3f},  "
          f"p(1s) = {pp1:.5f}  {stars(pp1)}")


# ════════════════════════════════════════════════════════════════
#  ██  H2: IS PM COST MORE NEGATIVE THAN BENCHMARK (−2.25 pp)?
# ════════════════════════════════════════════════════════════════
section(f"H2: IS PM INCUMBENCY COST MORE NEGATIVE THAN BENCHMARK ({BENCHMARK:+.2f} pp)?")
print(f"  H0: μ(PM Ce) = {BENCHMARK}  vs.  HA: μ(PM Ce) < {BENCHMARK}")
print(f"  [PM parties face a LARGER cost than the all-party average government]")
print()
print(f"  Design note: N&P compute Ce for all governments at all elections,")
print(f"  regardless of outcome. This paper follows the same design for PM parties,")
print(f"  including re-election bids that were subsequently lost. This makes the")
print(f"  comparison fully symmetric.")

t_h2, p_h2_2s = stats.ttest_1samp(all_inc, BENCHMARK)
p_h2_1s_less    = p_one_less(t_h2, p_h2_2s)         # PM more negative than benchmark
se_h2           = all_inc.std() / np.sqrt(len(all_inc))
ci_h2           = stats.t.interval(0.95, df=len(all_inc)-1, loc=all_inc.mean(), scale=se_h2)
diff_from_bench = all_inc.mean() - BENCHMARK

print(f"""
  ─── One-sample t-test vs. benchmark ({BENCHMARK:+.2f} pp) ───
  N:                        {len(all_inc)}
  Mean Ce:                  {all_inc.mean():+.3f} pp
  Benchmark:                {BENCHMARK:+.2f} pp
  Difference (PM − bench):  {diff_from_bench:+.3f} pp

  t = {t_h2:.3f},  df = {len(all_inc)-1}
  p (2-sided)            = {p_h2_2s:.5f}
  p (1-sided, PM < bench) = {p_h2_1s_less:.5f}  {stars(p_h2_1s_less)}
  95% CI: [{ci_h2[0]:+.3f}, {ci_h2[1]:+.3f}]

  → H2 {'CONFIRMED ✓' if p_h2_1s_less < ALPHA else 'NOT CONFIRMED'}
    PM parties bear a {abs(diff_from_bench):.2f} pp {'larger (more negative)' if diff_from_bench < 0 else 'smaller (less negative)'}
    cost than the all-party benchmark.
""")



# ════════════════════════════════════════════════════════════════
#  ██  H3: GEOGRAPHIC CONSISTENCY
# ════════════════════════════════════════════════════════════════
section("H3: GEOGRAPHIC CONSISTENCY OF PM INCUMBENCY COST")
print("  H0: Proportion of countries with mean Ce < 0 = 0.5")
print("  HA: Proportion > 0.5  [majority of countries show a cost]")

country_inc = (
    analysis.groupby("country_name")["Ce"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "mean_ce", "count": "n"})
    .reset_index()
)

n_negative            = (country_inc["mean_ce"] < 0).sum()
n_countries_with_data = len(country_inc)

try:
    bp = stats.binomtest(n_negative, n_countries_with_data, 0.5, alternative="greater").pvalue
except AttributeError:
    bp = stats.binom_test(n_negative, n_countries_with_data, 0.5, alternative="greater")

print(f"""
  Countries with incumbency observations:   {n_countries_with_data}
  Countries with negative mean Ce:          {n_negative}  ({100*n_negative/n_countries_with_data:.0f}%)
  Countries with non-negative mean Ce:      {n_countries_with_data - n_negative}  ({100*(n_countries_with_data-n_negative)/n_countries_with_data:.0f}%)

  Binomial test: p = {bp:.5f}  {stars(bp)}
  → H3 {'CONFIRMED ✓' if bp < ALPHA else 'NOT CONFIRMED'}
""")

print(f"  {'Country':<22} {'N':>4} {'Mean Ce':>9} {'Direction':>10}")
print("  " + "─" * 48)
for _, row in country_inc.sort_values("mean_ce").iterrows():
    direction = "✓ (neg)" if row["mean_ce"] < 0 else "✗ (pos)"
    print(f"  {row['country_name']:<22} {int(row['n']):>4} "
          f"{row['mean_ce']:>+9.2f} {direction:>10}")


# ════════════════════════════════════════════════════════════════
#  ██  WINNER/LOSER DECOMPOSITION
# ════════════════════════════════════════════════════════════════
section("WINNER/LOSER DECOMPOSITION — PM INSULATION ANALYSIS")
print("""
  The N&P-comparable sample includes PM parties that subsequently LOST power.
  Decomposing Ce by election outcome reveals the structure of the PM cost.
""")

for label, d in [
    ("Winners (retained PM)", won_ce),
    ("Losers  (lost PM)",     lost_ce),
    ("All PM bids",           all_inc),
]:
    if len(d) < 2:
        continue
    t_d, p_d = stats.ttest_1samp(d, 0)
    p1_d = p_one_less(t_d, p_d)
    t_b, p_b = stats.ttest_1samp(d, BENCHMARK)
    p1_b = p_one_less(t_b, p_b)
    print(f"  {label} (N={len(d):>3}): mean={d.mean():+.3f} pp, "
          f"vs 0: p={p1_d:.4f}{stars(p1_d)}, vs bench: p={p1_b:.4f}{stars(p1_b)}")

print()
print(f"  Winner/Loser mean difference: {won_ce.mean()-lost_ce.mean():+.3f} pp")
t_wl, p_wl = stats.ttest_ind(won_ce, lost_ce)
print(f"  Two-sample t-test (won vs lost): t={t_wl:.3f}, p(2s)={p_wl:.5f}  {stars(p_wl)}")





# ════════════════════════════════════════════════════════════════
#  ██  FINAL VERDICT
# ════════════════════════════════════════════════════════════════
print("\n\n" + "█" * 70)
print("  FINAL VERDICT — N&P-COMPARABLE DESIGN")
print("█" * 70)

print(f"""
  H1 — PM INCUMBENCY COST IS NEGATIVE:
    Mean Ce = {all_inc.mean():+.3f} pp  (N = {len(all_inc)})
    t = {t_h1:.3f},  p(1-sided) = {p_h1_1s:.5f}  {stars(p_h1_1s)}
    95% CI: [{ci_h1[0]:+.3f}, {ci_h1[1]:+.3f}]
    → {'CONFIRMED ✓' if p_h1_1s < ALPHA else 'NOT CONFIRMED at α=5%'}

  H2 — PM COST MORE NEGATIVE THAN BENCHMARK ({BENCHMARK:+.2f} pp):
    Difference (PM − benchmark) = {diff_from_bench:+.3f} pp
    p(1-sided, PM < benchmark) = {p_h2_1s_less:.5f}  {stars(p_h2_1s_less)}
 
  H3 — GEOGRAPHIC CONSISTENCY:
    {n_negative}/{n_countries_with_data} countries ({100*n_negative/n_countries_with_data:.0f}%) show negative mean PM incumbency effect
    Binomial: p = {bp:.5f}  {stars(bp)}
    → {'CONFIRMED ✓' if bp < ALPHA else 'NOT CONFIRMED'}



 

    Geographical consistency: {n_negative}/{n_countries_with_data} countries ({100*n_negative/n_countries_with_data:.0f}%)

""")

print("═" * 70)
print("  Analysis complete.")
print("═" * 70)
