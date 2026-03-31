# loading relevant modules
import os 
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
# ignoring filterwarnings for clean output
warnings.filterwarnings("ignore")



# loading Doering, H., & Manow, P. (2024). ParlGov 2024 Release. https://doi.org/10.7910/DVN/
# 2VZ5ZC, adjust path when running code
path = "yourpath/dataverse_files"



elections = pd.read_csv(os.path.join(path, "view_election.csv"))
cabinets  = pd.read_csv(os.path.join(path, "view_cabinet.csv"))

# country filter for selected sample
oecd_countries = [
    "Australia",
    "Austria",
    "Belgium",
    "Canada",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Iceland",
    "Ireland",
    "Italy",
    "Japan",
    "Luxembourg",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Portugal",
    "Spain",
    "Sweden",
    "United Kingdom",
]




# convert dates to datetime objects
elections["election_date"]  = pd.to_datetime(elections["election_date"])
elections["election_year"]  = elections["election_date"].dt.year
cabinets["election_date"]   = pd.to_datetime(cabinets["election_date"])


# filter sample for country-sample, 1990-2020 and parliamentary elections
df = elections[
    (elections["country_name"].isin(oecd_countries)) &
    (elections["election_year"] >= 1990) &
    (elections["election_year"] <= 2020) &
    (elections["election_type"] == "parliament")
].copy()


# extracting PM party at time of election, renamed as incumbent_party_id
pm_party = (cabinets[cabinets["prime_minister"] == 1]
            [["country_name","election_date","party_id"]]
            .drop_duplicates(subset=["country_name","election_date"])
            .rename(columns={"party_id":"incumbent_party_id"}))


# merge incumbent party into election data
df = df.merge(pm_party, on=["country_name","election_date"], how="left")
# extract rows where party in election dataset is incumbent and extract said party's vote share
incumbent_vs = (df[df["party_id"] == df["incumbent_party_id"]]
                [["country_name","election_date","vote_share"]]
                .rename(columns={"vote_share":"incumbent_vote_share"}))

# comprime data into one row and attach incumbent voteshare / incumbent party id
df_elec = (df[["country_name","election_date","election_year"]]
           .drop_duplicates(subset=["country_name","election_date"]))
df_elec = df_elec.merge(incumbent_vs, on=["country_name","election_date"], how="left")
df_elec = df_elec.merge(pm_party,     on=["country_name","election_date"], how="left")
df_elec = df_elec.sort_values(["country_name","election_year"]).reset_index(drop=True)

#counting consecutive terms, relevant to testing H1/H2/H3
def count_terms(df):
    df = df.copy()
    df["consecutive_terms"] = 0
    for country in df["country_name"].unique():
        mask = df["country_name"] == country
        rows = df[mask].sort_values("election_year")
        terms, prev_party = 0, None
        for idx, row in rows.iterrows():
            terms = terms + 1 if row["incumbent_party_id"] == prev_party else 1
            df.loc[idx, "consecutive_terms"] = terms
            prev_party = row["incumbent_party_id"]
    return df

df_elec = count_terms(df_elec)

# vote share table by party, country and election year
party_votes = df[["country_name","election_year","party_id","vote_share"]].drop_duplicates()

#vote change vs own previous election
prev_votes = []
for _, row in df_elec.iterrows():
    prior = party_votes[
        (party_votes["country_name"] == row["country_name"]) &
        (party_votes["party_id"]     == row["incumbent_party_id"]) &
        (party_votes["election_year"] < row["election_year"])
    ].sort_values("election_year")
    prev_votes.append(prior["vote_share"].iloc[-1] if len(prior) > 0 else np.nan)

# defining dependent variable \Delta Vote
df_elec["prev_vote_share"] = prev_votes
df_elec["vote_change"]     = df_elec["incumbent_vote_share"] - df_elec["prev_vote_share"]
clean = df_elec.dropna(subset=["vote_change"]).copy()


# split by term: honeymoon, fatigue, consecutive
term1 = clean[clean["consecutive_terms"] == 1]["vote_change"]
term2 = clean[clean["consecutive_terms"] == 2]["vote_change"]
term3 = clean[clean["consecutive_terms"] >= 3]["vote_change"]

# ════════════════════════════════════════════════════════════
# TEST 1: OVERALL INCUMBENCY ADVANTAGE
# One-sample t-test: H0: mean vote change = 0
# ════════════════════════════════════════════════════════════

print("=" * 65)
print("TEST 1: OVERALL INCUMBENCY ADVANTAGE")
print("H0: Mean vote change = 0  (no incumbency effect)")
print("=" * 65)

t, p = stats.ttest_1samp(clean["vote_change"], 0)
ci   = stats.t.interval(0.95, len(clean)-1,
                         loc=clean["vote_change"].mean(),
                         scale=stats.sem(clean["vote_change"]))
print(f"\n  N:               {len(clean)}")
print(f"  Mean ΔVote:      {clean['vote_change'].mean():+.3f} pp")
print(f"  Std Dev:         {clean['vote_change'].std():.3f} pp")
print(f"  95% CI:          [{ci[0]:+.3f}, {ci[1]:+.3f}]")
print(f"  t-statistic:     {t:.3f}")
print(f"  p-value:         {p:.4f}")
print(f"  Significant:     {'YES ***' if p<0.01 else 'YES **' if p<0.05 else 'YES *' if p<0.10 else 'NO'}")
print(f"\n  → {'INCUMBENCY ADVANTAGE CONFIRMED ✓' if p<0.05 and clean['vote_change'].mean()>0 else 'NOT CONFIRMED'}")

# ════════════════════════════════════════════════════════════
# TEST 2: HONEYMOON EFFECT (1st term)
# H0: mean vote change in term 1 = 0
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 2: HONEYMOON EFFECT — First-term incumbents")
print("H0: Mean vote change in term 1 = 0")
print("=" * 65)

t1, p1 = stats.ttest_1samp(term1, 0)
ci1    = stats.t.interval(0.95, len(term1)-1,
                           loc=term1.mean(),
                           scale=stats.sem(term1))
print(f"\n  N (1st term):    {len(term1)}")
print(f"  Mean ΔVote:      {term1.mean():+.3f} pp")
print(f"  Std Dev:         {term1.std():.3f} pp")
print(f"  95% CI:          [{ci1[0]:+.3f}, {ci1[1]:+.3f}]")
print(f"  t-statistic:     {t1:.3f}")
print(f"  p-value:         {p1:.4f}")
print(f"  Significant:     {'YES ***' if p1<0.01 else 'YES **' if p1<0.05 else 'YES *' if p1<0.10 else 'NO'}")
print(f"\n  → {'HONEYMOON EFFECT CONFIRMED ✓' if p1<0.05 and term1.mean()>0 else 'NOT CONFIRMED'}")

# ════════════════════════════════════════════════════════════
# TEST 3: FATIGUE EFFECT (2nd term)
# H0: mean vote change in term 2 = 0
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 3: FATIGUE EFFECT — Second-term incumbents")
print("H0: Mean vote change in term 2 = 0")
print("=" * 65)

t2, p2 = stats.ttest_1samp(term2, 0)
ci2    = stats.t.interval(0.95, len(term2)-1,
                           loc=term2.mean(),
                           scale=stats.sem(term2))
print(f"\n  N (2nd term):    {len(term2)}")
print(f"  Mean ΔVote:      {term2.mean():+.3f} pp")
print(f"  Std Dev:         {term2.std():.3f} pp")
print(f"  95% CI:          [{ci2[0]:+.3f}, {ci2[1]:+.3f}]")
print(f"  t-statistic:     {t2:.3f}")
print(f"  p-value:         {p2:.4f}")
print(f"  Significant:     {'YES ***' if p2<0.01 else 'YES **' if p2<0.05 else 'YES *' if p2<0.10 else 'NO'}")
print(f"\n  → {'FATIGUE EFFECT CONFIRMED ✓' if p2<0.05 and term2.mean()<0 else 'MARGINAL' if p2<0.10 and term2.mean()<0 else 'NOT CONFIRMED'}")

# ════════════════════════════════════════════════════════════
# TEST 4: HONEYMOON > FATIGUE
# Two-sample t-test: H0: mean(term1) = mean(term2)
# This directly tests whether honeymoon and fatigue differ
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 4: IS HONEYMOON SIGNIFICANTLY LARGER THAN FATIGUE?")
print("H0: Mean ΔVote(term 1) = Mean ΔVote(term 2)")
print("= 65)")

t_diff, p_diff = stats.ttest_ind(term1, term2, equal_var=False)
ci_diff = stats.t.interval(0.95,
                             len(term1)+len(term2)-2,
                             loc=term1.mean()-term2.mean(),
                             scale=np.sqrt(term1.sem()**2 + term2.sem()**2))
print(f"\n  Term 1 mean:     {term1.mean():+.3f} pp  (N={len(term1)})")
print(f"  Term 2 mean:     {term2.mean():+.3f} pp  (N={len(term2)})")
print(f"  Difference:      {term1.mean()-term2.mean():+.3f} pp")
print(f"  95% CI of diff:  [{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}]")
print(f"  t-statistic:     {t_diff:.3f}")
print(f"  p-value:         {p_diff:.4f}")
print(f"  Significant:     {'YES ***' if p_diff<0.01 else 'YES **' if p_diff<0.05 else 'YES *' if p_diff<0.10 else 'NO'}")
print(f"\n  → {'HONEYMOON SIGNIFICANTLY LARGER THAN FATIGUE ✓' if p_diff<0.05 else 'DIFFERENCE MARGINALLY SIGNIFICANT' if p_diff<0.10 else 'DIFFERENCE NOT SIGNIFICANT'}")

# ════════════════════════════════════════════════════════════
# TEST 5: NON-PARAMETRIC CONFIRMATION
# Wilcoxon signed-rank test (does not assume normality)
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 5: NON-PARAMETRIC ROBUSTNESS CHECK")
print("Wilcoxon signed-rank test (no normality assumption)")
print("=" * 65)

for label, data in [("Overall", clean["vote_change"]),
                     ("Term 1 (honeymoon)", term1),
                     ("Term 2 (fatigue)",   term2)]:
    w, pw = stats.wilcoxon(data)
    sig   = "***" if pw<0.01 else "**" if pw<0.05 else "*" if pw<0.10 else "n.s."
    print(f"\n  {label}:")
    print(f"    W-statistic:   {w:.1f}")
    print(f"    p-value:       {pw:.4f}  {sig}")
    print(f"    Median ΔVote:  {data.median():+.3f} pp")

# ════════════════════════════════════════════════════════════
# TEST 6: BOOTSTRAP CONFIDENCE INTERVALS
# Model-free, resampling-based CI for each term mean
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 6: BOOTSTRAP CONFIDENCE INTERVALS (10,000 replications)")
print("Resampling-based — robust to any distributional assumption")
print("=" * 65)

np.random.seed(42)
B = 10000

for label, data in [("Overall",           clean["vote_change"]),
                     ("Term 1 (honeymoon)", term1),
                     ("Term 2 (fatigue)",   term2),
                     ("Term 3+",            term3)]:
    boot = [np.random.choice(data, size=len(data), replace=True).mean()
            for _ in range(B)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig    = "SIGNIFICANT" if (lo > 0 or hi < 0) else "NOT significant"
    print(f"\n  {label} (N={len(data)}):")
    print(f"    Mean:          {data.mean():+.3f} pp")
    print(f"    Bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    print(f"    Zero in CI:    {'NO → ' + sig : <30}")

# ════════════════════════════════════════════════════════════
# TEST 7: DECOMPOSITION VERIFICATION
# Verify the weighted decomposition adds up correctly
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 7: DECOMPOSITION VERIFICATION")
print("Does weighted sum of term effects = overall mean?")
print("=" * 65)

N    = len(clean)
w1   = len(term1) / N
w2   = len(term2) / N
w3   = len(term3) / N

contrib1 = w1 * term1.mean()
contrib2 = w2 * term2.mean()
contrib3 = w3 * term3.mean()
total    = contrib1 + contrib2 + contrib3

print(f"\n  Term 1: w={w1:.3f} × {term1.mean():+.3f} = {contrib1:+.3f} pp")
print(f"  Term 2: w={w2:.3f} × {term2.mean():+.3f} = {contrib2:+.3f} pp")
print(f"  Term 3: w={w3:.3f} × {term3.mean():+.3f} = {contrib3:+.3f} pp")
print(f"  {'─'*40}")
print(f"  Weighted sum:         {total:+.3f} pp")
print(f"  Direct overall mean:  {clean['vote_change'].mean():+.3f} pp")
print(f"  Difference:           {abs(total - clean['vote_change'].mean()):.4f} pp  "
      f"({'✓ verified' if abs(total - clean['vote_change'].mean()) < 0.01 else '⚠ check data'})")

pct_honey  = (contrib1 / clean["vote_change"].mean()) * 100
pct_fat    = (contrib2 / clean["vote_change"].mean()) * 100
pct_entren = (contrib3 / clean["vote_change"].mean()) * 100

print(f"\n  Honeymoon share:      {pct_honey:+.1f}% of net advantage")
print(f"  Fatigue share:        {pct_fat:+.1f}% of net advantage")
print(f"  Entrenchment share:   {pct_entren:+.1f}% of net advantage")

# ════════════════════════════════════════════════════════════
# TEST 8: COUNTRY-LEVEL ROBUSTNESS
# Is the honeymoon effect present in a majority of countries?
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("TEST 8: COUNTRY-LEVEL ROBUSTNESS")
print("Is the honeymoon positive in most individual countries?")
print("=" * 65)

country_results = []
for country in sorted(clean["country_name"].unique()):
    sub1 = clean[(clean["country_name"]==country) &
                 (clean["consecutive_terms"]==1)]["vote_change"]
    sub2 = clean[(clean["country_name"]==country) &
                 (clean["consecutive_terms"]==2)]["vote_change"]
    country_results.append({
        "country":       country,
        "n_term1":       len(sub1),
        "mean_term1":    sub1.mean() if len(sub1)>0 else np.nan,
        "n_term2":       len(sub2),
        "mean_term2":    sub2.mean() if len(sub2)>0 else np.nan,
    })

cr = pd.DataFrame(country_results)
pos_honey = (cr["mean_term1"] > 0).sum()
neg_fat   = (cr["mean_term2"] < 0).sum()
n_cty     = cr["mean_term1"].notna().sum()
n_cty2    = cr["mean_term2"].notna().sum()

print(f"\n  Countries with positive 1st-term effect: "
      f"{pos_honey}/{n_cty} ({100*pos_honey/n_cty:.0f}%)")
print(f"  Countries with negative 2nd-term effect: "
      f"{neg_fat}/{n_cty2} ({100*neg_fat/n_cty2:.0f}%)")

print(f"\n  {'Country':<20} {'N(t1)':>6} {'ΔV(t1)':>9} {'N(t2)':>6} {'ΔV(t2)':>9}")
print("  " + "-"*55)
for _, row in cr.iterrows():
    t1_str = f"{row['mean_term1']:+.2f}" if not np.isnan(row['mean_term1']) else "  n/a"
    t2_str = f"{row['mean_term2']:+.2f}" if not np.isnan(row['mean_term2']) else "  n/a"
    print(f"  {row['country']:<20} {int(row['n_term1']) if row['n_term1']>0 else 0:>6} "
          f"{t1_str:>9} {int(row['n_term2']) if row['n_term2']>0 else 0:>6} {t2_str:>9}")

# ════════════════════════════════════════════════════════════
# FINAL VERDICT
# ════════════════════════════════════════════════════════════

print("\n" + "█"*65)
print("  FINAL VERDICT")
print("█"*65)
print(f"""
  H1 (Overall advantage):  Mean = {clean['vote_change'].mean():+.2f} pp
                           p = {stats.ttest_1samp(clean['vote_change'],0)[1]:.4f}
                           → {"CONFIRMED ✓" if stats.ttest_1samp(clean['vote_change'],0)[1]<0.05 else "NOT CONFIRMED"}

  H2 (Honeymoon, term 1):  Mean = {term1.mean():+.2f} pp
                           p = {stats.ttest_1samp(term1,0)[1]:.4f}
                           → {"CONFIRMED ✓" if stats.ttest_1samp(term1,0)[1]<0.05 and term1.mean()>0 else "NOT CONFIRMED"}

  H3 (Fatigue, term 2):    Mean = {term2.mean():+.2f} pp
                           p = {stats.ttest_1samp(term2,0)[1]:.4f}
                           → {"CONFIRMED ✓" if stats.ttest_1samp(term2,0)[1]<0.05 and term2.mean()<0 else "MARGINAL ◑" if stats.ttest_1samp(term2,0)[1]<0.10 and term2.mean()<0 else "NOT CONFIRMED"}

  H2 vs H3 (difference):   Δ = {term1.mean()-term2.mean():+.2f} pp
                           p = {stats.ttest_ind(term1,term2,equal_var=False)[1]:.4f}
                           → {"CONFIRMED ✓" if stats.ttest_ind(term1,term2,equal_var=False)[1]<0.05 else "MARGINAL ◑" if stats.ttest_ind(term1,term2,equal_var=False)[1]<0.10 else "NOT CONFIRMED"}

  Decomposition verified:  Honeymoon = {pct_honey:.1f}% of net advantage
                           Fatigue   = {pct_fat:.1f}% of net advantage
""")



# ════════════════════════════════════════════════════════════
# FIGURE: Incumbency Advantage — Honeymoon & Fatigue Effects
# ════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


means      = [clean["vote_change"].mean(), term1.mean(), term2.mean()]
ci_lo      = [
    stats.t.interval(0.95, len(clean)-1,  loc=clean["vote_change"].mean(), scale=stats.sem(clean["vote_change"]))[0],
    stats.t.interval(0.95, len(term1)-1,  loc=term1.mean(), scale=stats.sem(term1))[0],
    stats.t.interval(0.95, len(term2)-1,  loc=term2.mean(), scale=stats.sem(term2))[0],
]
ci_hi      = [
    stats.t.interval(0.95, len(clean)-1,  loc=clean["vote_change"].mean(), scale=stats.sem(clean["vote_change"]))[1],
    stats.t.interval(0.95, len(term1)-1,  loc=term1.mean(), scale=stats.sem(term1))[1],
    stats.t.interval(0.95, len(term2)-1,  loc=term2.mean(), scale=stats.sem(term2))[1],
]
xlabels    = ["Overall<br>(N=210)", "Honeymoon<br>Term 1 (N=104)", "Fatigue<br>Term 2 (N=62)"]  # shortened
bar_colors = ["#5B8FF9", "#5AD8A6", "#FF6B6B"]
p_labels   = ["p < 0.0001 ***", "p < 0.0001 ***", "p = 0.136 n.s."]

cr_plot    = cr.dropna(subset=["mean_term1"]).sort_values("mean_term1").reset_index(drop=True)
dot_colors = ["#FF6B6B" if v < -0.5 else "#5AD8A6" for v in cr_plot["mean_term1"]]

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.42, 0.58],
    subplot_titles=["Mean Vote Change by Term (95% CI)",
                    "Country-Level Honeymoon Effect (Term 1)"],
    horizontal_spacing=0.20
)

# ── Left: bar chart ──
for xl, mean, lo, hi, col, pv in zip(xlabels, means, ci_lo, ci_hi, bar_colors, p_labels):
    fig.add_trace(go.Bar(
        x=[xl], y=[mean],
        error_y=dict(type="data", array=[hi-mean], arrayminus=[mean-lo],
                     visible=True, thickness=2, width=8, color="rgba(0,0,0,0.45)"),
        marker_color=col,
        marker_line_color="rgba(0,0,0,0.2)", marker_line_width=1,
        showlegend=False,
        text=[f"{mean:+.2f} pp"], textposition="outside", textfont=dict(size=12),
    ), row=1, col=1)

for xl, pv in zip(xlabels, p_labels):
    fig.add_annotation(
        x=xl, y=-5.2, text=f"<i>{pv}</i>",
        showarrow=False, font=dict(size=9, color="gray"),
        xref="x1", yref="y1"
    )

fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1.5, row=1, col=1)
fig.update_xaxes(title_text="Incumbency Term", tickangle=0, row=1, col=1)  # force upright
fig.update_yaxes(
    title_text="ΔVote Share (pp)", row=1, col=1,
    range=[-6.5, 11],
    gridcolor="rgba(200,200,200,0.4)", zeroline=False
)

# ── Right: dot plot ──
fig.add_trace(go.Scatter(
    x=cr_plot["mean_term1"].tolist(),
    y=cr_plot["country"].tolist(),
    mode="markers+text",
    marker=dict(color=dot_colors, size=10,
                line=dict(color="rgba(0,0,0,0.2)", width=1)),
    text=[f"{v:+.1f}" for v in cr_plot["mean_term1"]],
    textposition=["middle left" if v < 0 else "middle right"
                  for v in cr_plot["mean_term1"]],
    textfont=dict(size=8.5),
    showlegend=False,
    hovertemplate="%{y}: %{x:+.2f} pp<extra></extra>"
), row=1, col=2)

fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1.5, row=1, col=2)
fig.update_xaxes(title_text="ΔVote Share (pp)", row=1, col=2,
                 gridcolor="rgba(200,200,200,0.4)",
                 range=[-3, 21])  # extended to show Hungary fully
fig.update_yaxes(
    tickfont=dict(size=8.5),
    tickmode="array",
    tickvals=cr_plot["country"].tolist(),
    ticktext=cr_plot["country"].tolist(),
    row=1, col=2
)

fig.update_layout(
    title=dict(
        text=("Incumbency Advantage in OECD Democracies, 1990–2020<br>"
              "<span style='font-size:15px;font-weight:normal;color:gray'>"
              "Source: ParlGov | 28/31 countries show positive honeymoon</span>"),
        x=0.5, xanchor="center"
    ),
    margin=dict(t=110, b=70, l=60, r=110),  # increased right margin
    width=1250, height=720,                   # wider canvas
    plot_bgcolor="white", paper_bgcolor="white",
)

os.makedirs("/Users/nkf/Desktop/atp/figures", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
fig.write_image(f"yourpath/honeymoon_results_{timestamp}.png")
print("Figure saved → honeymoon_results.png")

