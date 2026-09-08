# 📐 Formoptimierung von Stabtragwerken

Interaktive Demo zur Formoptimierung von Fachwerken: Die Topologie bleibt fest, aber einzelne Knoten dürfen sich entlang einer vorgegebenen Linie ("Schnitt") verschieben — z. B. die Höhe eines Brückenobergurts oder die Breite eines Mastsegments.

## Worum geht's?

Zweite Stufe der klassischen Dreier-Hierarchie der Strukturoptimierung (**Sizing → Form → Topologie**), direkte Fortsetzung von [truss-sizing-demo](https://github.com/sebastian-hanisch/truss-sizing-demo): dort waren Knoten und Stäbe fest vorgegeben und nur die Querschnitte wurden optimiert. Hier bleibt die Topologie weiterhin fest, aber Knoten dürfen sich auf vorgegebenen Bewegungslinien verschieben. Für **jede** geprüfte Form wird intern die vollständige Sizing-Optimierung gelöst — das macht dies zu einer echten **bilevel-Optimierung** (äußere Form, innere Querschnitte).

Kernthema der Demo: Eine Koordinate nach der anderen zu optimieren (Koordinatensuche) übersieht Kopplungseffekte zwischen Knoten, die bei statisch unbestimmten Tragwerken real sind — eine echte gemeinsame Optimierung findet dagegen automatisch zwei klassische Ergebnisse der Formoptimierung: bei einer Brücke folgt die optimale Tragwerkstiefe dem Biegemomentenverlauf (Bogenträger-/Fischbauch-Prinzip, am tiefsten in Feldmitte), bei einem Mast verjüngt sich die optimale Form nach oben (Battered-Tower-Prinzip).

## Methodik

- Vier Formvorlagen: dieselben drei Tragwerks-Geometrien wie in truss-sizing-demo (Höhenprofil des 10-Stab-Kragarms und der Pratt-Brücke, Verjüngungsprofil des Antennenmasts), plus ein minimales Test-Dreieck mit nur einem freien Knoten
- Bilevel-Struktur: die äußere Formoptimierung ruft bei jeder Auswertung das vollständige innere Sizing-Problem auf (Vollspannungs-Entwurf, derselbe Solver wie in truss-sizing-demo)
- Drei Lösungsverfahren für die äußere Form: naive Koordinatensuche (eine Achse nach der anderen per Rastersuche), kontinuierliche Optimierung (SciPy SLSQP über alle Schnitt-Parameter gemeinsam), Metaheuristik (SciPy Differential Evolution)
- Empirisch verifiziert (Parametersweeps + globale DE-Referenzläufe) für alle vier Vorlagen, bevor die App gebaut wurde — inklusive einer Fixpunkt-Konvergenzbeschleunigung für den doppelt statisch unbestimmten Kragarm, der das innere Sizing sonst bei jeder Auswertung ausbremste
- PDF-Export, Permalink
- Mathematische Herleitung der bilevel-Formulierung im Expander „Mathematische Formulierung“

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `pytest tests/ -v`

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von [Sebastian Hanisch](https://sebastianhanisch.net) — Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
