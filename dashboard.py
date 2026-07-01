import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------ CONFIG
st.set_page_config(page_title="Data Quality — Road Safety 2024",
                   page_icon="🚦", layout="wide")

BLUE, ORANGE, GREEN, RED, GREY = "#118DFF", "#E66C37", "#0F9D58", "#D64550", "#6B6B6B"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.kpi {background:#F7F9FC;border-left:6px solid %s;padding:14px 16px;border-radius:8px;margin-bottom:6px;}
.kpi .lbl {font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.6px;}
.kpi .val {font-size:27px;font-weight:700;color:#1b1b1b;line-height:1.1;}
.kpi .sub {font-size:11px;color:#999;}
h2 {border-bottom:2px solid #EEE;padding-bottom:4px;}
</style>
""" % BLUE, unsafe_allow_html=True)

FILES = {"caract": "caract", "lieux": "lieux", "vehicules": "vehicules", "usagers": "usagers"}
PK = {"caract": "Num_Acc", "vehicules": "id_vehicule", "usagers": "id_usager", "lieux": None}


@st.cache_data
def load():
    dfs = {}
    for name, prefix in FILES.items():                    # works for 2024, 2025, ...
        dfs[name] = pd.read_csv(glob.glob(prefix + "*.csv")[0], sep=";", decimal=",",
                                na_values=["-1", " -1", "N/A"], low_memory=False)
    year = int(dfs["caract"]["an"].mode()[0])             # year inferred from the data
    dfs["usagers"]["age"] = year - dfs["usagers"]["an_nais"]
    return dfs, year


dfs, YEAR = load()
car, veh, usa, lieux = dfs["caract"], dfs["vehicules"], dfs["usagers"], dfs["lieux"]


def kpi(col, label, value, sub="", color=BLUE):
    col.markdown(f'<div class="kpi" style="border-color:{color}">'
                 f'<div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def gauge(value, title, color=BLUE):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value, 1), number={"suffix": " %", "font": {"size": 22}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1},
               "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, 75], "color": "#FCE8E6"},
                         {"range": [75, 90], "color": "#FEF7E0"},
                         {"range": [90, 100], "color": "#E6F4EA"}]}))
    fig.update_layout(height=170, margin=dict(l=15, r=15, t=35, b=5),
                      title={"text": title, "font": {"size": 14}, "x": 0.5})
    return fig


# ------------------------------------------------------------------ SCORES (5 dimensions from the course)
tot_cells = sum(df.drop(columns=["age"], errors="ignore").size for df in dfs.values())
tot_missing = sum(df.drop(columns=["age"], errors="ignore").isnull().sum().sum() for df in dfs.values())
tot_rows = sum(len(df) for df in dfs.values())
tot_dups = sum(df.duplicated().sum() for df in dfs.values())

# --- WEIGHTED completeness: a missing value on an important column costs more ---
HIGH_WEIGHT = {"Num_Acc", "grav", "lat", "long", "an", "mois", "jour", "hrmn", "an_nais",
              "catu", "sexe", "catv", "catr", "vma", "agg", "lum", "atm", "col", "dep", "com",
              "id_usager", "id_vehicule"}
LOW_WEIGHT = {"etatp", "secu3", "secu2", "locp", "actp", "larrout", "lartpc", "pr", "pr1",
                "v1", "v2", "voie", "occutc", "adr", "num_veh"}


def weight(col):
    return 3.0 if col in HIGH_WEIGHT else (0.3 if col in LOW_WEIGHT else 1.0)


num = den = 0.0
for df in dfs.values():
    for col in df.columns:
        if col == "age":
            continue
        w = weight(col)
        den += w * len(df)
        num += w * df[col].notna().sum()
completeness = num / den * 100

# --- COMBINED uniqueness: half unique primary key + half absence of duplicates ---
pk_ok = [0 if key is None else (100 if dfs[name][key].duplicated().sum() == 0 else 0)
         for name, key in PK.items()]
uniqueness_tables = []
for name, df in dfs.items():
    key = PK[name]
    pk_score = 0 if key is None else (100 if df[key].duplicated().sum() == 0 else 0)
    dup_score = (1 - df.duplicated().mean()) * 100
    uniqueness_tables.append(0.5 * pk_score + 0.5 * dup_score)
uniqueness = float(np.mean(uniqueness_tables))

# French geographic zones (mainland + overseas) : lat_min, lat_max, lon_min, lon_max
BOXES = [(41.3, 51.1, -5.3, 9.6), (15.7, 16.55, -61.85, -61.0), (14.35, 14.9, -61.25, -60.8),
         (2.0, 5.8, -54.7, -51.5), (-21.4, -20.8, 55.2, 55.9), (-13.1, -12.4, 45.0, 45.35),
         (46.7, 47.2, -56.5, -56.1), (17.8, 18.2, -63.2, -62.75), (-28, -7, -155, -134),
         (-22.9, -19.5, 163.5, 168.2), (-14.4, -13.1, -178.3, -176.0)]


def coords_outside_fr(lat, lon):
    ok = pd.Series(False, index=lat.index)
    for a, b, c, d in BOXES:
        ok |= lat.between(a, b) & lon.between(c, d)
    return int(((lat.notna() & lon.notna()) & ~ok).sum())


bad_geo = coords_outside_fr(car["lat"], car["long"])   # coords outside FR territory (swapped lat/long)
bad_age = int(((usa["age"] < 0) | (usa["age"] > 110)).sum())
bad_vma = int((lieux["vma"] > 130).sum())              # >130 km/h is impossible in France

# Validity = share of variables (checked columns) WITHOUT any invalid value
rng = lambda *a: set(range(*a))
RULES = {
 "caract":    {"jour": dict(lo=1, hi=31), "mois": dict(lo=1, hi=12), "an": dict(allowed={YEAR}),
               "lum": dict(allowed={1, 2, 3, 4, 5}), "agg": dict(allowed={1, 2}),
               "int": dict(allowed=rng(1, 10)), "atm": dict(allowed=rng(1, 10)), "col": dict(allowed=rng(1, 8))},
 "lieux":     {"catr": dict(allowed={1, 2, 3, 4, 5, 6, 7, 9}), "circ": dict(allowed={1, 2, 3, 4}),
               "vosp": dict(allowed={0, 1, 2, 3}), "prof": dict(allowed={1, 2, 3, 4}),
               "plan": dict(allowed={1, 2, 3, 4}), "surf": dict(allowed=rng(1, 10)),
               "infra": dict(allowed=rng(0, 10)), "situ": dict(allowed={0, 1, 2, 3, 4, 5, 6, 8}),
               "vma": dict(lo=5, hi=130), "nbv": dict(lo=0, hi=20)},
 "usagers":   {"catu": dict(allowed={1, 2, 3}), "grav": dict(allowed={1, 2, 3, 4}), "sexe": dict(allowed={1, 2}),
               "an_nais": dict(lo=1900, hi=YEAR), "trajet": dict(allowed={0, 1, 2, 3, 4, 5, 9}),
               "secu1": dict(allowed=rng(0, 10)), "secu2": dict(allowed=rng(0, 10)), "secu3": dict(allowed=rng(0, 10))},
 "vehicules": {"senc": dict(allowed={0, 1, 2, 3}), "obs": dict(allowed=rng(0, 18)),
               "obsm": dict(allowed={0, 1, 2, 4, 5, 6, 9}), "choc": dict(allowed=rng(0, 10)),
               "manv": dict(allowed=rng(0, 27)), "motor": dict(allowed=rng(0, 7)), "occutc": dict(lo=0, hi=200)},
}


def n_invalid(s, allowed=None, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce"); ok = s.notna()
    if allowed is not None:
        return int((ok & ~s.isin(list(allowed))).sum())
    return int((ok & ~s.between(lo, hi)).sum())


val_rows = [("caract", "lat/long", bad_geo)]
for tbl, cols in RULES.items():
    for col, kw in cols.items():
        val_rows.append((tbl, col, n_invalid(dfs[tbl][col], **kw)))
val_df = pd.DataFrame(val_rows, columns=["table", "column", "n_invalid"])
n_var = len(val_df)
n_var_bad = int((val_df.n_invalid > 0).sum())
validity = (1 - n_var_bad / n_var) * 100

freshness = float((car["an"] == YEAR).mean() * 100)    # share of rows on the dataset's year
overall_score = np.mean([completeness, validity, uniqueness, freshness])

# ------------------------------------------------------------------ HEADER
st.title(f"🚦 Data Quality — Road Accidents (BAAC {YEAR})")
st.caption("Data profiling and quality report · 4 files linked by Num_Acc · "
           "source: data.gouv.fr")

# ------------------------------------------------------------------ KPI ROW (Dataset statistics)
st.subheader("Overview")
c = st.columns(5)
kpi(c[0], "Tables", "4", "caract · lieux · vehicles · users")
kpi(c[1], "Variables", str(sum(df.shape[1] for df in dfs.values()) - 1), "columns in total")
kpi(c[2], "Observations", f"{tot_rows:,}".replace(",", " "), "rows across all files")
kpi(c[3], "Missing cells", f"{tot_missing/tot_cells*100:.1f} %",
    f"{tot_missing:,}".replace(",", " ") + " cells", ORANGE)
kpi(c[4], "Duplicate rows", f"{tot_dups}", "across all files", RED if tot_dups else GREEN)
st.markdown(
    "<div style='font-size:12.5px;color:#555;line-height:1.9'>"
    "• <b>Tables</b> — number of files in the dataset.<br>"
    "• <b>Variables</b> — total number of columns (all tables combined).<br>"
    "• <b>Observations</b> — total number of rows.<br>"
    "• <b>Missing cells</b> — % of empty fields / « N/A » / -1.<br>"
    "• <b>Duplicate rows</b> — strictly identical rows."
    "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ GAUGES (5 dimensions)
st.subheader("Quality by dimension (course: completeness · validity · uniqueness · freshness)")
g = st.columns(5)
g[0].plotly_chart(gauge(overall_score, "Overall score",
                        GREEN if overall_score > 90 else ORANGE), use_container_width=True)
g[1].plotly_chart(gauge(completeness, "Completeness", ORANGE), use_container_width=True)
g[2].plotly_chart(gauge(validity, "Validity",
                        GREEN if validity >= 95 else ORANGE), use_container_width=True)
g[3].plotly_chart(gauge(uniqueness, "Uniqueness",
                        GREEN if uniqueness == 100 else ORANGE), use_container_width=True)
g[4].plotly_chart(gauge(freshness, "Freshness", GREEN), use_container_width=True)
st.markdown(
    "<div style='font-size:12.5px;color:#555;line-height:1.9'>"
    "• <b>Overall score</b> — average of the 4 dimensions below.<br>"
    "• <b>Completeness</b> — % of filled-in cells, <b>weighted by column importance</b> "
    "(a missing value on grav/coordinates counts more than one on etatp/secu3).<br>"
    f"• <b>Validity</b> — share of variables with no invalid value at all: "
    f"{n_var - n_var_bad}/{n_var} (swapped lat/long and abnormal vma bring the score down).<br>"
    f"• <b>Uniqueness</b> — 50% « unique primary key » ({sum(1 for v in pk_ok if v)}/4 tables, "
    "lieux doesn't have one) + 50% « no duplicate rows ».<br>"
    "• <b>Freshness</b> — share of the data that falls on the dataset's year."
    "</div>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------------ PRIMARY KEYS / INTEGRITY
st.subheader("Primary keys and referential integrity")
col_pk, col_note = st.columns([2, 1])
rows = []
acc = set(car["Num_Acc"])
for name, df in dfs.items():
    key = PK[name]
    if key is None:
        rows.append([name, "— (none)", "❌ no primary key", f"{len(df):,}".replace(",", " ")])
    else:
        dup = df[key].duplicated().sum()
        rows.append([name, key, "✅ unique" if dup == 0 else f"❌ {dup} duplicates",
                     f"{len(df):,}".replace(",", " ")])
pk_df = pd.DataFrame(rows, columns=["Table", "Primary key", "Uniqueness", "Rows"])
col_pk.dataframe(pk_df, use_container_width=True, hide_index=True)
col_note.info("**lieux has no unique identifier**: its Num_Acc repeats "
              f"({len(lieux):,} rows for {lieux['Num_Acc'].nunique():,} accidents) because an accident at "
              "an intersection describes several roads. → 1:N, risk of double counting when joining."
              .replace(",", " "))
orph = {n: int((~dfs[n]["Num_Acc"].isin(acc)).sum()) for n in ["lieux", "vehicules", "usagers"]}
st.caption("Referential integrity (rows with no Num_Acc in caract): " +
           " · ".join(f"{k} = {v}" for k, v in orph.items()) + "  →  no orphan rows.")

st.divider()

# ------------------------------------------------------------------ DISTRIBUTIONS
st.subheader("Key distributions")
d1, d2, d3 = st.columns(3)
labels = {1: "Unharmed", 2: "Killed", 3: "Hospitalized", 4: "Slightly injured"}
grav = usa["grav"].map(labels).value_counts()
with d1:
    st.markdown("**Severity of users**")
    fig = px.bar(x=grav.index, y=grav.values,
                 color=grav.index, color_discrete_sequence=[GREEN, RED, ORANGE, "#F2C14E"])
    fig.update_layout(height=280, showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis_title="", yaxis_title="users")
    st.plotly_chart(fig, use_container_width=True)
with d2:
    st.markdown("**Age of users**")
    fig = px.histogram(usa["age"].dropna(), nbins=40, color_discrete_sequence=[BLUE])
    fig.update_layout(height=280, showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis_title="age", yaxis_title="users")
    st.plotly_chart(fig, use_container_width=True)
with d3:
    st.markdown("**Accidents per month**")
    pm = car["mois"].value_counts().sort_index()
    fig = px.line(x=pm.index, y=pm.values, markers=True, color_discrete_sequence=[ORANGE])
    fig.update_layout(height=280, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis_title="month", yaxis_title="accidents")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------------ VALIDITY CHECKS
st.subheader("Validity checks (ranges & anomalies)")
checks = pd.DataFrame({
    "Check": ["Coordinates outside FR territory (swapped lat/long)",
                 "Ages out of range (< 0 or > 110)",
                 "Max speed > 130 km/h (impossible)",
                 "grav ∈ {1,2,3,4}", "sexe ∈ {1,2}", "Years present"],
    "Result": [f"{bad_geo} cases (Mayotte / Haute-Corse swapped → in the sea)",
                 f"{bad_age} cases",
                 f"{bad_vma} cases (up to {int(lieux['vma'].max())} km/h)",
                 "OK" if set(usa['grav'].dropna().unique()) <= {1, 2, 3, 4} else "anomaly",
                 "OK" if set(usa['sexe'].dropna().unique()) <= {1, 2} else "anomaly",
                 ", ".join(map(str, sorted(car['an'].dropna().unique())))],
    "Status": ["✅" if bad_geo == 0 else "⚠️", "✅" if bad_age == 0 else "⚠️",
               "✅" if bad_vma == 0 else "⚠️", "✅", "✅", "✅"],
})
st.dataframe(checks, use_container_width=True, hide_index=True)

# -------- Detail: all problematic rows + their issue --------
st.markdown("#### Problematic rows (detail)")

issues = []

# 1) coordinates outside French territory (swapped lat/long)
ok = pd.Series(False, index=car.index)
for a, b, c, d in BOXES:
    ok = ok | (car["lat"].between(a, b) & car["long"].between(c, d))
mask_geo = (car["lat"].notna() & car["long"].notna()) & ~ok
for _, r in car[mask_geo].iterrows():
    issues.append({"table": "caract", "id": r["Num_Acc"], "column": "lat / long",
                   "value": f"lat={r['lat']:.2f}, long={r['long']:.2f}",
                   "issue": "Coordinates outside France (swapped lat/long)"})

# 2) max speed outside the legal range (5–130 km/h)
for _, r in lieux[(lieux["vma"] > 130) | (lieux["vma"] < 5)].iterrows():
    v = r["vma"]
    prob = "Impossible max speed (> 130 km/h)" if v > 130 else "Max speed too low (< 5 km/h)"
    issues.append({"table": "lieux", "id": r["Num_Acc"], "column": "vma",
                   "value": f"{v:.0f} km/h", "issue": prob})

# 3) strictly duplicated rows
for _, r in lieux[lieux.duplicated(keep=False)].iterrows():
    issues.append({"table": "lieux", "id": r["Num_Acc"], "column": "(whole row)",
                   "value": "—", "issue": "Strictly duplicated row"})

issues = pd.DataFrame(issues)
st.write(f"**{len(issues)} problematic rows** detected in total.")
st.dataframe(issues, use_container_width=True, hide_index=True, height=320)

st.caption("Missing values = empty fields, « N/A », or code -1 (not filled in). "
           "Profiling aligned with the course (5 dimensions) and the assignment (inventory, "
           "missing values, ranges, anomalies, duplicates, keys).")
