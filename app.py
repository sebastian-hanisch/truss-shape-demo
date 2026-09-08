"""
Formoptimierung von Stabtragwerken – interaktive Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweite Stufe der Strukturoptimierungs-Hierarchie (Sizing -> Form ->
Topologie), direkte Fortsetzung von truss-sizing-demo: dort waren Knoten
und Stäbe fest vorgegeben und nur die Querschnitte wurden optimiert. Hier
ist die TOPOLOGIE weiterhin fest, aber einzelne Knoten dürfen sich entlang
einer vorgegebenen Linie ("Schnitt") verschieben - z. B. die Höhe eines
Brückenfachwerk-Obergurts oder die Breite eines Mastes je Segment. Für
JEDE geprüfte Form wird intern die volle Sizing-Optimierung (Vollspannungs-
Entwurf) gelöst - das macht dies zu einer echten bilevel-Optimierung
(äußere Form, innere Querschnitte).

Kernthema dieser Demo: Eine Koordinate nach der anderen zu optimieren
(Koordinatensuche) übersieht Kopplungseffekte zwischen Knoten, die bei
statisch unbestimmten Tragwerken real sind - eine echte gemeinsame
Optimierung (SLSQP oder eine Metaheuristik) findet dagegen automatisch
klassische Ergebnisse der Formoptimierung: bei einer Brücke folgt die
optimale Tragwerkstiefe dem Biegemomentenverlauf (am tiefsten in
Feldmitte), bei einem Mast verjüngt sich die optimale Form nach oben.

Code-Struktur: Modell, Sizing-Teilproblem, Formoptimierungs-Solver,
PDF-Export und Visualisierung liegen in den Modulen ts_*.py neben dieser
Datei.
"""

import numpy as np
import pandas as pd
import streamlit as st

from ts_constants import A_MAX, A_MIN, DEFAULT_MATERIAL, MATERIALS
from ts_model import constraint_verletzungen, gesamtmasse
from ts_networks import VORLAGEN, VORLAGEN_NAMEN
from ts_pdf_export import generate_shape_report_pdf
from ts_presets import (
    MATERIAL_NAMEN,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ts_sizing import vollspannungs_entwurf
from ts_solver import loese_alle
from ts_visualization import konvergenz_figure, struktur_figure, vergleich_balken_figure

METHODEN_REIHENFOLGE = ["Koordinatensuche", "Kontinuierliche Optimierung", "Metaheuristik (Differential Evolution)"]
METHODEN_TAB_LABEL = {
    "Koordinatensuche": "📍 Koordinatensuche",
    "Kontinuierliche Optimierung": "📈 Kontinuierliche Optimierung",
    "Metaheuristik (Differential Evolution)": "🧬 Metaheuristik",
}


def _sizing_details(vorlage, material, lastfaktor, t_werte):
    struktur = vorlage.baue_struktur(t_werte, lastfaktor)
    flaechen = vollspannungs_entwurf(struktur, material, A_MIN, A_MAX)
    spannungsausl, knickausl, _ = constraint_verletzungen(struktur, flaechen, material)
    auslastung = np.maximum(spannungsausl, knickausl)
    return struktur, flaechen, auslastung


@st.cache_data(show_spinner=False)
def _compute(vorlage_name, material_name, lastfaktor, seed, cache_key):
    vorlage = VORLAGEN[vorlage_name]
    material = MATERIALS[material_name]
    ergebnisse = loese_alle(vorlage, material, lastfaktor, seed)
    return vorlage, material, ergebnisse


st.set_page_config(page_title="Formoptimierung von Stabtragwerken – Sebastian Hanisch", layout="wide")

st.title("📐 Formoptimierung von Stabtragwerken")
st.markdown(
    """
Interaktive Demo zur **Formoptimierung von Fachwerken**: Die Topologie (welche Knoten mit welchen
Stäben verbunden sind) bleibt fest, aber einzelne Knoten dürfen sich entlang einer vorgegebenen
Linie ("**Schnitt**") verschieben - z. B. die Höhe eines Brückenobergurts oder die Breite eines
Mastsegments. Für jede geprüfte Form wird intern die **vollständige Querschnittsoptimierung** gelöst
(dasselbe Sizing-Problem wie in truss-sizing-demo) - gesucht ist die Form, die danach am leichtesten
ist. Details im Expander "Wie funktioniert diese Demo?" unten sowie formal hergeleitet im Expander
"📐 Mathematische Formulierung".
"""
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_col1, preset_col2, preset_col3 = st.columns(3)
with preset_col1:
    st.button(
        "🌉 Pratt-Brücke (Höhenprofil)", use_container_width=True,
        on_click=apply_preset, args=("Pratt-Brückenfachwerk (Höhenprofil)", DEFAULT_MATERIAL, 1.0, 0),
        help="Klassisches Ergebnis: die optimale Tragwerkstiefe folgt dem Biegemomentenverlauf - am tiefsten in Feldmitte.",
    )
with preset_col2:
    st.button(
        "🗼 Antennenmast (Verjüngung)", use_container_width=True,
        on_click=apply_preset, args=("Turmfachwerk (Verjüngungsprofil)", DEFAULT_MATERIAL, 1.0, 0),
        help="Der optimierte Mast verjüngt sich nach oben - wie ein echter Gittermast.",
    )
with preset_col3:
    st.button(
        "⚠️ Statisch unbestimmt (10-Stab)", use_container_width=True,
        on_click=apply_preset, args=("10-Stab-Kragarm (Höhenprofil)", DEFAULT_MATERIAL, 1.0, 0),
        help="Hier zeigt sich der Unterschied zwischen schrittweiser Koordinatensuche und echter gemeinsamer Optimierung am deutlichsten.",
    )

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    vorlage_name = st.selectbox("Tragwerk", options=VORLAGEN_NAMEN, key="vorlage_select")
    material_name = st.selectbox("Werkstoff", options=MATERIAL_NAMEN, key="material_select")
    lastfaktor = st.slider(
        "Lastfaktor", *bounds("lastfaktor_slider"), step=0.1, key="lastfaktor_slider",
        help="Skaliert alle Lasten - höhere Lasten machen die optimale Form meist noch ausgeprägter.",
    )

    seed_lo, seed_hi = bounds("seed_input")
    seed = st.number_input("Zufalls-Seed (Metaheuristik)", min_value=seed_lo, max_value=seed_hi, step=1, key="seed_input")
    st.button(
        "🎲 Neuen Zufalls-Seed für die Metaheuristik", use_container_width=True, on_click=randomize_seed,
        help="Differential Evolution ist stochastisch - ein neuer Seed führt zu einer neuen Suchtrajektorie.",
    )

sync_query_params(vorlage_name, material_name, lastfaktor, int(seed))

cache_key = (vorlage_name, material_name, lastfaktor, int(seed))
with st.spinner("Optimiere Form (jede Zielfunktionsauswertung löst intern das Sizing-Problem)..."):
    vorlage, material, ergebnisse = _compute(vorlage_name, material_name, lastfaktor, int(seed), cache_key)

bester_name = min(ergebnisse, key=lambda k: ergebnisse[k].masse)
bestes = ergebnisse[bester_name]
struktur_default, flaechen_default, _ = _sizing_details(vorlage, material, lastfaktor, vorlage.t_defaults())
struktur_opt, flaechen_opt, auslastung_opt = _sizing_details(vorlage, material, lastfaktor, bestes.t_werte)
masse_default = gesamtmasse(struktur_default, flaechen_default, material)

st.markdown(f"## 🎯 Ergebnis: {bester_name}")

m1, m2, m3 = st.columns(3)
m1.metric("Gesamtmasse (optimierte Form)", f"{bestes.masse:.0f} kg")
m2.metric("Gesamtmasse (Standardform)", f"{masse_default:.0f} kg", delta=f"{bestes.masse - masse_default:.0f} kg", delta_color="inverse")
m3.metric("Anzahl freie Knoten", f"{vorlage.anzahl_frei()}")

st.plotly_chart(
    struktur_figure(struktur_opt, flaechen_opt, auslastung_opt, f"{vorlage.name} – optimierte Form (gestrichelt: Standardform)", referenz=struktur_default),
    use_container_width=True, key="struktur_main",
)

pdf_bytes = generate_shape_report_pdf(vorlage, material, bestes, flaechen_opt, auslastung_opt)
st.download_button(
    "📄 Formoptimierungs-Bericht als PDF herunterladen", data=pdf_bytes,
    file_name="formoptimierung_bericht.pdf", mime="application/pdf", key="primary_pdf_download",
)

st.caption(
    "Ermittelt mit dem besten von drei eigenen Lösungsverfahren für dieses Szenario - jede "
    "Formauswertung löst intern die vollständige Querschnittsoptimierung. Details unten im "
    "vollständigen Methodenvergleich."
)

st.markdown("---")
st.subheader("📐 Wann übersieht eine schrittweise Suche Kopplungseffekte?")
st.markdown(
    """
Die **Koordinatensuche** optimiert einen Knoten nach dem anderen und hält die übrigen dabei fest.
Das findet das Optimum genau dann, wenn die Knoten sich gegenseitig kaum beeinflussen. Bei statisch
**unbestimmten** Tragwerken (mehrere Lastpfade) verschiebt eine Änderung an einem Knoten aber die
Kraftverteilung im ganzen Tragwerk - der wirklich beste Kompromiss kann dann nur gefunden werden,
wenn mehrere Knoten GEMEINSAM verändert werden.
"""
)

koord = ergebnisse["Koordinatensuche"]
andere = [v for k, v in ergebnisse.items() if k != "Koordinatensuche"]
beste_gemeinsame = min(andere, key=lambda v: v.masse)
gap_pct = (koord.masse - beste_gemeinsame.masse) / beste_gemeinsame.masse * 100

c1, c2 = st.columns(2)
c1.metric("Koordinatensuche", f"{koord.masse:.0f} kg")
c2.metric(
    "Beste gemeinsame Optimierung", f"{beste_gemeinsame.masse:.0f} kg",
    delta=f"{-gap_pct:.1f} %", delta_color="inverse",
)
if gap_pct > 1.5:
    st.warning(
        f"⚠️ Bei **{vorlage.name}** verschenkt die Koordinatensuche **{gap_pct:.1f}%** Gewicht "
        "gegenüber gemeinsamer Optimierung - ein Hinweis auf Kopplung zwischen den Knoten."
    )
else:
    st.success(
        f"✅ Bei **{vorlage.name}** liegt die Koordinatensuche nur **{max(gap_pct, 0):.1f}%** über dem "
        "besten gefundenen Ergebnis - hier beeinflussen sich die Knoten kaum gegenseitig."
    )

st.markdown("---")

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich", expanded=False):
    st.plotly_chart(vergleich_balken_figure(ergebnisse), use_container_width=True, key="vergleich_balken")
    tabelle = pd.DataFrame(
        [
            {"Methode": name, "Masse (kg)": ergebnisse[name].masse, "Rechenzeit (s)": ergebnisse[name].rechenzeit}
            for name in METHODEN_REIHENFOLGE
        ]
    )
    st.dataframe(tabelle, use_container_width=True, hide_index=True)

    tab_labels = [METHODEN_TAB_LABEL[m] for m in METHODEN_REIHENFOLGE] + ["📊 Konvergenz"]
    tabs = st.tabs(tab_labels)
    beschreibungen = {
        "Koordinatensuche": "Optimiert eine Koordinate nach der anderen per Rastersuche - schnell, aber blind für Kopplungseffekte zwischen Knoten.",
        "Kontinuierliche Optimierung": "SciPy SLSQP optimiert alle Schnitt-Parameter gemeinsam - findet Kopplungseffekte, aber als lokales Verfahren ohne Garantie auf das globale Optimum.",
        "Metaheuristik (Differential Evolution)": "Globale, ableitungsfreie Suche über alle Schnitt-Parameter gleichzeitig - robuster gegen lokale Optima, aber langsamer.",
    }
    for tab, name in zip(tabs[:-1], METHODEN_REIHENFOLGE):
        with tab:
            ergebnis = ergebnisse[name]
            st.caption(beschreibungen[name])
            s, f_opt, a_opt = _sizing_details(vorlage, material, lastfaktor, ergebnis.t_werte)
            tm1, tm2 = st.columns(2)
            tm1.metric("Masse", f"{ergebnis.masse:.0f} kg")
            tm2.metric("Rechenzeit", f"{ergebnis.rechenzeit:.2f} s")
            st.plotly_chart(
                struktur_figure(s, f_opt, a_opt, name, referenz=struktur_default),
                use_container_width=True, key=f"struktur_{name}",
            )

    with tabs[-1]:
        meta_verlauf = ergebnisse["Metaheuristik (Differential Evolution)"].verlauf
        st.caption("Beste bisher gefundene Masse je Generation der Differential-Evolution-Suche.")
        st.plotly_chart(konvergenz_figure(meta_verlauf), use_container_width=True, key="konvergenz")

st.markdown("---")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        r"""
**Die Problemstellung:** Formoptimierung ist die zweite Stufe der klassischen Dreier-Hierarchie der
Strukturoptimierung (**Sizing -> Form -> Topologie**, siehe truss-sizing-demo für Stufe 1). Die
Topologie (Knoten, Stäbe, Lagerung) bleibt fest, aber einzelne Knoten dürfen sich entlang einer
vorgegebenen Linie - einem "**Schnitt**" - verschieben, z. B. eine Höhenlinie (Brücke, Kragarm,
Dreieck) oder eine Breitenlinie (Mast). Gesucht ist die Position auf jedem Schnitt, die - nach
erneuter Optimierung der Querschnitte - das leichteste Tragwerk ergibt.

**Warum das eine bilevel-Optimierung ist:** Jede geprüfte Form braucht ihre EIGENEN optimalen
Querschnitte, um fair verglichen zu werden - eine tiefere Brücke etwa hat andere (meist kleinere)
Stabkräfte als eine flache, braucht aber längere Diagonalen. Die äußere Formoptimierung ruft daher
bei jeder Auswertung das vollständige innere Sizing-Problem auf (Vollspannungs-Entwurf, derselbe
Fixpunkt-Solver wie in truss-sizing-demo, hier bewusst als einziges Sizing-Verfahren, weil das
Sizing selbst nicht mehr das Thema ist).

**Drei Lösungsverfahren im Vergleich:**
- **Koordinatensuche**: optimiert einen Knoten nach dem anderen per Rastersuche, hält die übrigen
  fest. Findet das Optimum exakt dann, wenn sich die Knoten kaum gegenseitig beeinflussen - bei
  echter Kopplung (statisch unbestimmte Tragwerke) bleibt sie stecken, weil sie nie zwei Koordinaten
  gleichzeitig verändert.
- **Kontinuierliche Optimierung** (SciPy SLSQP): optimiert alle Schnitt-Parameter gemeinsam -
  erfasst Kopplungseffekte, ist aber ein lokales Verfahren.
- **Metaheuristik** (SciPy Differential Evolution): globale, ableitungsfreie Suche über alle
  Parameter gleichzeitig - robuster, aber langsamer.

**Zwei klassische Ergebnisse der Formoptimierung, hier live nachrechenbar:**
- **Brücken-Höhenprofil folgt dem Biegemomentenverlauf**: bei einer einfach gelagerten Brücke unter
  Last ist die Biegebeanspruchung an den Auflagern null und in Feldmitte maximal - die optimierte
  Form wird entsprechend an den Auflagern flach und in der Mitte tief (Bogen-/Fischbauchträger-Prinzip).
- **Türme verjüngen sich nach oben**: ein oben belasteter, schlanker Mast ist unten den größten
  Biegemomenten ausgesetzt - die optimierte Form wird unten breit und oben schmal, wie ein echter
  Gittermast.

**Ausblick:** Die dritte Stufe - Topologieoptimierung (welche Stäbe überhaupt existieren, z. B. über
die Ground-Structure-Method) - baut auf genau diesem Formoptimierungs-Baustein auf: jede geprüfte
Topologie braucht wiederum eine Form- UND Querschnittsoptimierung, um fair bewertet zu werden.
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
Gegeben eine feste Topologie (Knoten $j$, Stäbe $i$, Lagerung, Lasten $\mathbf{f}$) und für eine
Teilmenge der Knoten je eine Bewegungslinie (Schnitt) $j \mapsto \mathbf{p}_j(t_j) = \mathbf{p}_j^0 +
t_j \, \mathbf{d}_j$ mit Richtungsvektor $\mathbf{d}_j$ und Grenzen $t_j \in [t_j^{\min}, t_j^{\max}]$.
Gesucht sind die Parameter $t_j$, die - nach optimaler Querschnittswahl - die Gesamtmasse minimieren:
"""
    )
    st.latex(
        r"\min_{t_j \in [t_j^{\min}, t_j^{\max}]} \;\; \underbrace{\min_{A_i \in [A_{\min}, A_{\max}]} "
        r"\sum_i \rho \, A_i \, L_i(\mathbf{t})}_{\text{inneres Sizing-Problem, siehe truss-sizing-demo}}"
    )
    st.markdown(
        r"""
mit denselben Spannungs- und Knick-Nebenbedingungen wie in truss-sizing-demo, ausgewertet für die
durch $\mathbf{t}$ bestimmte Geometrie (Stablängen $L_i(\mathbf{t})$ und Richtungscosinus hängen
direkt von den Knotenpositionen ab).

**Warum die innere Optimierung IN der äußeren steckt (nicht daneben):** Für zwei verschiedene Formen
$\mathbf{t}_1 \neq \mathbf{t}_2$ sind auch die optimalen Querschnitte i. A. verschieden - ein
direkter Massenvergleich zweier Formen ist nur fair, wenn beide mit ihren JEWEILS optimalen
Querschnitten verglichen werden. Die äußere Zielfunktion ist deshalb selbst ein Optimierungsproblem
(`ts_sizing.masse_bei_geometrie()` in `ts_solver.py`) - eine echte **bilevel-Optimierung**.

**Warum Koordinatensuche bei Kopplung scheitert:** Die Koordinatensuche optimiert
$t_j \mapsto \text{Masse}(t_1, \ldots, t_j, \ldots, t_n)$ nacheinander für jedes $j$ bei festen
übrigen Werten - das findet nur dann das globale Optimum, wenn die Zielfunktion **separabel** ist
(keine Wechselwirkung zwischen den $t_j$). Bei statisch unbestimmten Tragwerken ist das i. A. NICHT
der Fall: die optimale Steifigkeitsverteilung (und damit der beste Wert von $t_j$) hängt von der
Position der NACHBARKNOTEN ab, weil sich die Kraftverteilung mit der Geometrie mitverändert
(dieselbe Kraftumlagerung, die in truss-sizing-demo das Sizing-Problem nichtlinear macht, macht hier
zusätzlich die Formoptimierung nicht-separabel).

**Bezug zum Code:** `ts_model.py` implementiert die Fachwerk-FEM und die Schnitt-Parametrisierung
(`Schnitt.position()`), `ts_sizing.py` das innere Sizing-Problem (Vollspannungs-Entwurf),
`ts_solver.py` alle drei äußeren Formoptimierungs-Verfahren.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
