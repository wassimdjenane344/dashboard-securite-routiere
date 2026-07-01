import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------ CONFIG
st.set_page_config(page_title="Data Quality — Sécurité Routière 2024",
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
    for name, prefix in FILES.items():                    # marche pour 2024, 2025, ...
        dfs[name] = pd.read_csv(glob.glob(prefix + "*.csv")[0], sep=";", decimal=",",
                                na_values=["-1", " -1", "N/A"], low_memory=False)
    year = int(dfs["caract"]["an"].mode()[0])             # année déduite des données
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


# ------------------------------------------------------------------ SCORES (5 dimensions du cours)
tot_cells = sum(df.drop(columns=["age"], errors="ignore").size for df in dfs.values())
tot_missing = sum(df.drop(columns=["age"], errors="ignore").isnull().sum().sum() for df in dfs.values())
tot_rows = sum(len(df) for df in dfs.values())
tot_dups = sum(df.duplicated().sum() for df in dfs.values())

# --- Complétude PONDÉRÉE : un manquant sur une colonne importante coûte plus cher ---
POIDS_FORT = {"Num_Acc", "grav", "lat", "long", "an", "mois", "jour", "hrmn", "an_nais",
              "catu", "sexe", "catv", "catr", "vma", "agg", "lum", "atm", "col", "dep", "com",
              "id_usager", "id_vehicule"}
POIDS_FAIBLE = {"etatp", "secu3", "secu2", "locp", "actp", "larrout", "lartpc", "pr", "pr1",
                "v1", "v2", "voie", "occutc", "adr", "num_veh"}


def poids(col):
    return 3.0 if col in POIDS_FORT else (0.3 if col in POIDS_FAIBLE else 1.0)


num = den = 0.0
for df in dfs.values():
    for col in df.columns:
        if col == "age":
            continue
        w = poids(col)
        den += w * len(df)
        num += w * df[col].notna().sum()
completude = num / den * 100

# --- Unicité COMBINÉE : moitié clé primaire unique + moitié absence de doublons ---
pk_ok = [0 if key is None else (100 if dfs[name][key].duplicated().sum() == 0 else 0)
         for name, key in PK.items()]
uni_tables = []
for name, df in dfs.items():
    key = PK[name]
    pk_score = 0 if key is None else (100 if df[key].duplicated().sum() == 0 else 0)
    dup_score = (1 - df.duplicated().mean()) * 100
    uni_tables.append(0.5 * pk_score + 0.5 * dup_score)
unicite = float(np.mean(uni_tables))

# Zones géographiques françaises (métropole + outre-mer) : lat_min,lat_max,lon_min,lon_max
BOXES = [(41.3, 51.1, -5.3, 9.6), (15.7, 16.55, -61.85, -61.0), (14.35, 14.9, -61.25, -60.8),
         (2.0, 5.8, -54.7, -51.5), (-21.4, -20.8, 55.2, 55.9), (-13.1, -12.4, 45.0, 45.35),
         (46.7, 47.2, -56.5, -56.1), (17.8, 18.2, -63.2, -62.75), (-28, -7, -155, -134),
         (-22.9, -19.5, 163.5, 168.2), (-14.4, -13.1, -178.3, -176.0)]


def coords_hors_fr(lat, lon):
    ok = pd.Series(False, index=lat.index)
    for a, b, c, d in BOXES:
        ok |= lat.between(a, b) & lon.between(c, d)
    return int(((lat.notna() & lon.notna()) & ~ok).sum())


bad_geo = coords_hors_fr(car["lat"], car["long"])     # coords hors territoire FR (lat/long inversés)
bad_age = int(((usa["age"] < 0) | (usa["age"] > 110)).sum())
bad_vma = int((lieux["vma"] > 130).sum())             # >130 km/h impossible en France

# Validité = part des variables (colonnes contrôlées) SANS aucune valeur invalide
rng = lambda *a: set(range(*a))
REGLES = {
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


def n_invalides(s, allowed=None, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce"); ok = s.notna()
    if allowed is not None:
        return int((ok & ~s.isin(list(allowed))).sum())
    return int((ok & ~s.between(lo, hi)).sum())


val_rows = [("caract", "lat/long", bad_geo)]
for tbl, cols in REGLES.items():
    for col, kw in cols.items():
        val_rows.append((tbl, col, n_invalides(dfs[tbl][col], **kw)))
val_df = pd.DataFrame(val_rows, columns=["table", "colonne", "n_invalide"])
n_var = len(val_df)
n_var_bad = int((val_df.n_invalide > 0).sum())
validite = (1 - n_var_bad / n_var) * 100

fraicheur = float((car["an"] == YEAR).mean() * 100)     # toutes les lignes sur l'année du jeu
score_global = np.mean([completude, validite, unicite, fraicheur])

# ------------------------------------------------------------------ HEADER
st.title(f"🚦 Data Quality — Accidents de la route (BAAC {YEAR})")
st.caption("Rapport de profilage et qualité des données · 4 fichiers reliés par Num_Acc · "
           "source : data.gouv.fr")

# ------------------------------------------------------------------ KPI ROW (Dataset statistics)
st.subheader("Vue d'ensemble")
c = st.columns(5)
kpi(c[0], "Tables", "4", "caract · lieux · véhicules · usagers")
kpi(c[1], "Variables", str(sum(df.shape[1] for df in dfs.values()) - 1), "colonnes au total")
kpi(c[2], "Observations", f"{tot_rows:,}".replace(",", " "), "lignes tous fichiers")
kpi(c[3], "Cellules manquantes", f"{tot_missing/tot_cells*100:.1f} %",
    f"{tot_missing:,}".replace(",", " ") + " cellules", ORANGE)
kpi(c[4], "Lignes dupliquées", f"{tot_dups}", "sur tous les fichiers", RED if tot_dups else GREEN)
st.markdown(
    "<div style='font-size:12.5px;color:#555;line-height:1.9'>"
    "• <b>Tables</b> — nombre de fichiers du jeu de données.<br>"
    "• <b>Variables</b> — nombre total de colonnes (toutes tables confondues).<br>"
    "• <b>Observations</b> — nombre total de lignes.<br>"
    "• <b>Cellules manquantes</b> — % de champs vides / « N/A » / -1.<br>"
    "• <b>Lignes dupliquées</b> — lignes strictement identiques."
    "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ GAUGES (5 dimensions)
st.subheader("Qualité par dimension (cours : complétude · validité · unicité · fraîcheur)")
g = st.columns(5)
g[0].plotly_chart(gauge(score_global, "Score global",
                        GREEN if score_global > 90 else ORANGE), use_container_width=True)
g[1].plotly_chart(gauge(completude, "Complétude", ORANGE), use_container_width=True)
g[2].plotly_chart(gauge(validite, "Validité",
                        GREEN if validite >= 95 else ORANGE), use_container_width=True)
g[3].plotly_chart(gauge(unicite, "Unicité",
                        GREEN if unicite == 100 else ORANGE), use_container_width=True)
g[4].plotly_chart(gauge(fraicheur, "Fraîcheur", GREEN), use_container_width=True)
st.markdown(
    "<div style='font-size:12.5px;color:#555;line-height:1.9'>"
    "• <b>Score global</b> — moyenne des 4 dimensions ci-dessous.<br>"
    "• <b>Complétude</b> — % de cellules renseignées, <b>pondéré par l'importance</b> des colonnes "
    "(un manquant sur grav/coordonnées compte plus qu'un manquant sur etatp/secu3).<br>"
    f"• <b>Validité</b> — part des variables sans aucune valeur invalide : "
    f"{n_var - n_var_bad}/{n_var} (lat/long inversés et vma aberrants font baisser le score).<br>"
    f"• <b>Unicité</b> — 50 % « clé primaire unique » ({sum(1 for v in pk_ok if v)}/4 tables, "
    "lieux n'en a pas) + 50 % « absence de lignes dupliquées ».<br>"
    "• <b>Fraîcheur</b> — part des données situées sur l'année du jeu."
    "</div>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------------ CLÉS PRIMAIRES / INTÉGRITÉ
st.subheader("Clés primaires et intégrité référentielle")
col_pk, col_note = st.columns([2, 1])
rows = []
acc = set(car["Num_Acc"])
for name, df in dfs.items():
    key = PK[name]
    if key is None:
        rows.append([name, "— (aucune)", "❌ pas de clé primaire", f"{len(df):,}".replace(",", " ")])
    else:
        dup = df[key].duplicated().sum()
        rows.append([name, key, "✅ unique" if dup == 0 else f"❌ {dup} doublons",
                     f"{len(df):,}".replace(",", " ")])
pk_df = pd.DataFrame(rows, columns=["Table", "Clé primaire", "Unicité", "Lignes"])
col_pk.dataframe(pk_df, use_container_width=True, hide_index=True)
col_note.info("**lieux n'a aucun identifiant unique** : son Num_Acc se répète "
              f"({len(lieux):,} lignes pour {lieux['Num_Acc'].nunique():,} accidents) car un accident à "
              "une intersection décrit plusieurs voies. → 1:N, risque de double comptage en jointure."
              .replace(",", " "))
orph = {n: int((~dfs[n]["Num_Acc"].isin(acc)).sum()) for n in ["lieux", "vehicules", "usagers"]}
st.caption("Intégrité référentielle (lignes sans Num_Acc dans caract) : " +
           " · ".join(f"{k} = {v}" for k, v in orph.items()) + "  →  aucune ligne orpheline.")

st.divider()

# ------------------------------------------------------------------ DISTRIBUTIONS
st.subheader("Distributions clés")
d1, d2, d3 = st.columns(3)
labels = {1: "Indemne", 2: "Tué", 3: "Blessé hosp.", 4: "Blessé léger"}
grav = usa["grav"].map(labels).value_counts()
with d1:
    st.markdown("**Gravité des usagers**")
    fig = px.bar(x=grav.index, y=grav.values,
                 color=grav.index, color_discrete_sequence=[GREEN, RED, ORANGE, "#F2C14E"])
    fig.update_layout(height=280, showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis_title="", yaxis_title="usagers")
    st.plotly_chart(fig, use_container_width=True)
with d2:
    st.markdown("**Âge des usagers**")
    fig = px.histogram(usa["age"].dropna(), nbins=40, color_discrete_sequence=[BLUE])
    fig.update_layout(height=280, showlegend=False, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis_title="âge", yaxis_title="usagers")
    st.plotly_chart(fig, use_container_width=True)
with d3:
    st.markdown("**Accidents par mois**")
    pm = car["mois"].value_counts().sort_index()
    fig = px.line(x=pm.index, y=pm.values, markers=True, color_discrete_sequence=[ORANGE])
    fig.update_layout(height=280, margin=dict(l=5, r=5, t=5, b=5),
                      xaxis_title="mois", yaxis_title="accidents")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------------ CONTRÔLES DE VALIDITÉ
st.subheader("Contrôles de validité (plages & anomalies)")
checks = pd.DataFrame({
    "Contrôle": ["Coordonnées hors territoire FR (lat/long inversés)",
                 "Âges hors plage (< 0 ou > 110)",
                 "Vitesse max > 130 km/h (impossible)",
                 "grav ∈ {1,2,3,4}", "sexe ∈ {1,2}", "Années présentes"],
    "Résultat": [f"{bad_geo} cas (Mayotte / Haute-Corse permutés → en mer)",
                 f"{bad_age} cas",
                 f"{bad_vma} cas (jusqu'à {int(lieux['vma'].max())} km/h)",
                 "OK" if set(usa['grav'].dropna().unique()) <= {1, 2, 3, 4} else "anomalie",
                 "OK" if set(usa['sexe'].dropna().unique()) <= {1, 2} else "anomalie",
                 ", ".join(map(str, sorted(car['an'].dropna().unique())))],
    "Statut": ["✅" if bad_geo == 0 else "⚠️", "✅" if bad_age == 0 else "⚠️",
               "✅" if bad_vma == 0 else "⚠️", "✅", "✅", "✅"],
})
st.dataframe(checks, use_container_width=True, hide_index=True)

# -------- Détail : toutes les lignes problématiques + leur problème --------
st.markdown("#### Lignes problématiques (détail)")

problemes = []

# 1) coordonnées hors territoire français (lat/long inversés)
ok = pd.Series(False, index=car.index)
for a, b, c, d in BOXES:
    ok = ok | (car["lat"].between(a, b) & car["long"].between(c, d))
mask_geo = (car["lat"].notna() & car["long"].notna()) & ~ok
for _, r in car[mask_geo].iterrows():
    problemes.append({"table": "caract", "identifiant": r["Num_Acc"], "colonne": "lat / long",
                      "valeur": f"lat={r['lat']:.2f}, long={r['long']:.2f}",
                      "problème": "Coordonnées hors de France (lat/long inversés)"})

# 2) vitesse maximale hors plage légale (5–130 km/h)
for _, r in lieux[(lieux["vma"] > 130) | (lieux["vma"] < 5)].iterrows():
    v = r["vma"]
    prob = "Vitesse max impossible (> 130 km/h)" if v > 130 else "Vitesse max trop basse (< 5 km/h)"
    problemes.append({"table": "lieux", "identifiant": r["Num_Acc"], "colonne": "vma",
                      "valeur": f"{v:.0f} km/h", "problème": prob})

# 3) lignes strictement dupliquées
for _, r in lieux[lieux.duplicated(keep=False)].iterrows():
    problemes.append({"table": "lieux", "identifiant": r["Num_Acc"], "colonne": "(ligne entière)",
                      "valeur": "—", "problème": "Ligne strictement dupliquée"})

problemes = pd.DataFrame(problemes)
st.write(f"**{len(problemes)} lignes problématiques** détectées au total.")
st.dataframe(problemes, use_container_width=True, hide_index=True, height=320)

st.caption("Valeurs manquantes = champs vides, « N/A » ou code -1 (non renseigné). "
           "Profiling aligné sur le cours (5 dimensions) et le TP (inventaire, manquants, "
           "plages, anomalies, doublons, clés).")
