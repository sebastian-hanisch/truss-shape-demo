"""Drei Lösungsverfahren für dieselbe (äußere) Formoptimierung: gegeben eine
feste Topologie und Bewegungslinien ("Schnitte") je Knoten, welche
Position auf jedem Schnitt minimiert die Gesamtmasse? Jede Zielfunktions-
auswertung löst dabei intern das vollständige Sizing-Problem
(`ts_sizing.masse_bei_geometrie`) - das macht dies zu einer echten
bilevel-Optimierung (äußere Form, innere Querschnitte).
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from ts_constants import A_MAX, A_MIN, Material
from ts_model import Formvorlage, InstabilesTragwerk
from ts_sizing import masse_bei_geometrie

STRAF_MASSE = 1e7  # Ersatzwert für instabile (kinematisch labile) Geometrien


@dataclass
class Formergebnis:
    methode: str
    t_werte: np.ndarray
    masse: float
    rechenzeit: float
    verlauf: list[float] = field(default_factory=list)


_SUCH_MAX_ITER = 25  # schnelle (evtl. nicht ganz konvergierte) innere Lösung während der Suche
_FINAL_MAX_ITER = 150  # präzise innere Lösung für die abschließend berichtete Masse


def _ziel(vorlage: Formvorlage, material: Material, lastfaktor: float, max_iter: int = _SUCH_MAX_ITER):
    def f(t: np.ndarray) -> float:
        try:
            struktur = vorlage.baue_struktur(np.asarray(t), lastfaktor)
            return masse_bei_geometrie(struktur, material, A_MIN, A_MAX, max_iter=max_iter)
        except InstabilesTragwerk:
            return STRAF_MASSE

    return f


def _endgueltige_masse(vorlage: Formvorlage, material: Material, lastfaktor: float, t: np.ndarray) -> float:
    return _ziel(vorlage, material, lastfaktor, max_iter=_FINAL_MAX_ITER)(t)


def koordinatensuche(
    vorlage: Formvorlage, material: Material, lastfaktor: float, punkte_je_achse: int = 25, durchlaeufe: int = 4,
) -> Formergebnis:
    """Naive Baseline: optimiert eine Koordinate nach der anderen per
    Rastersuche, hält die übrigen fest - findet KEINE Kopplungseffekte
    zwischen Knoten (z. B. dass ein tieferer Mittelknoten einen flacheren
    Nachbarknoten begünstigt), weil nie zwei Koordinaten gleichzeitig
    verändert werden."""
    start = time.perf_counter()
    f = _ziel(vorlage, material, lastfaktor)
    lo, hi = vorlage.t_bounds()
    t = vorlage.t_defaults().copy()
    verlauf = [f(t)]

    for _ in range(durchlaeufe):
        verbessert = False
        for i in range(len(t)):
            kandidaten = np.linspace(lo[i], hi[i], punkte_je_achse)
            werte = []
            for kandidat in kandidaten:
                t_test = t.copy()
                t_test[i] = kandidat
                werte.append(f(t_test))
            bester_idx = int(np.argmin(werte))
            if werte[bester_idx] < verlauf[-1] - 1e-9:
                t[i] = kandidaten[bester_idx]
                verlauf.append(werte[bester_idx])
                verbessert = True
        if not verbessert:
            break

    masse = _endgueltige_masse(vorlage, material, lastfaktor, t)
    return Formergebnis("Koordinatensuche", t, masse, time.perf_counter() - start, verlauf)


def kontinuierliche_optimierung(vorlage: Formvorlage, material: Material, lastfaktor: float) -> Formergebnis:
    start = time.perf_counter()
    f = _ziel(vorlage, material, lastfaktor)
    lo, hi = vorlage.t_bounds()
    x0 = vorlage.t_defaults()
    res = minimize(f, x0, method="SLSQP", bounds=list(zip(lo, hi)), options={"maxiter": 200, "ftol": 1e-10})
    t = np.clip(res.x, lo, hi)
    masse = _endgueltige_masse(vorlage, material, lastfaktor, t)
    return Formergebnis("Kontinuierliche Optimierung", t, masse, time.perf_counter() - start)


def metaheuristik(vorlage: Formvorlage, material: Material, lastfaktor: float, seed: int) -> Formergebnis:
    start = time.perf_counter()
    f = _ziel(vorlage, material, lastfaktor)
    lo, hi = vorlage.t_bounds()
    verlauf = []

    def callback(intermediate_result):
        verlauf.append(float(intermediate_result.fun))

    res = differential_evolution(
        f, bounds=list(zip(lo, hi)), seed=seed, maxiter=50, popsize=10, tol=1e-9,
        polish=True, callback=callback,
    )
    t = np.clip(res.x, lo, hi)
    masse = _endgueltige_masse(vorlage, material, lastfaktor, t)
    return Formergebnis("Metaheuristik (Differential Evolution)", t, masse, time.perf_counter() - start, verlauf)


def loese_alle(vorlage: Formvorlage, material: Material, lastfaktor: float, seed: int) -> dict[str, Formergebnis]:
    return {
        "Koordinatensuche": koordinatensuche(vorlage, material, lastfaktor),
        "Kontinuierliche Optimierung": kontinuierliche_optimierung(vorlage, material, lastfaktor),
        "Metaheuristik (Differential Evolution)": metaheuristik(vorlage, material, lastfaktor, seed),
    }
