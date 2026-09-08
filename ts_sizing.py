"""Inneres Optimierungsproblem (Sizing): für eine GEGEBENE Geometrie die
leichtesten Querschnitte finden, die Spannungs- und Knick-Nebenbedingungen
einhalten. Dieselbe Vollspannungs-Entwurf-Fixpunktiteration wie in
truss-sizing-demo (dort einer von drei verglichenen Solvern - hier bewusst
der EINZIGE, weil das Sizing-Verfahren selbst nicht mehr das Thema ist,
sondern nur ein schneller innerer Baustein für die äußere Formoptimierung).
Kontinuierliche Flächen (kein Querschnittskatalog - das war die
Sizing-Demo)."""

import numpy as np

from ts_constants import Material
from ts_model import TrussStructure, gesamtmasse, loese_tragwerk


def _erforderliche_flaeche_je_stab(struktur: TrussStructure, material: Material, flaechen: np.ndarray) -> np.ndarray:
    ergebnis = loese_tragwerk(struktur, flaechen, material)
    kraefte = np.abs(ergebnis.stabkraefte)
    laengen = struktur.stablaengen()

    a_spannung = kraefte / material.sigma_zul
    druck = ergebnis.stabkraefte < 0
    a_knick = np.zeros_like(flaechen)
    a_knick[druck] = np.sqrt(4.0 * kraefte[druck] * laengen[druck] ** 2 / (np.pi * material.e_modul))
    return np.maximum(a_spannung, a_knick)


def vollspannungs_entwurf(
    struktur: TrussStructure, material: Material, a_min: float, a_max: float,
    max_iter: int = 150, tol: float = 1e-8,
) -> np.ndarray:
    """Bei manchen (statisch unbestimmten) Formen oszilliert die Fixpunkt-
    iteration zwischen zwei Zuständen statt zu konvergieren (bekannt aus
    truss-sizing-demo) - dort schöpfte jeder Aufruf sonst das volle
    Iterationslimit aus. Ein 2er-Zyklus-Check bricht so einen Fall früh ab
    und meldet den Mittelwert der beiden oszillierenden Zustände zurück -
    das ist sowohl schneller als auch eine ehrlichere Schätzung als ein
    beliebiger Schnappschuss bei Erreichen von `max_iter`."""
    n = len(struktur.staebe)
    flaechen = np.full(n, a_max)
    vorletzte = None
    for _ in range(max_iter):
        erforderlich = _erforderliche_flaeche_je_stab(struktur, material, flaechen)
        neu = np.clip(erforderlich, a_min, a_max)
        if np.max(np.abs(neu - flaechen)) / a_max < tol:
            return neu
        if vorletzte is not None and np.max(np.abs(neu - vorletzte)) / a_max < tol:
            return (neu + flaechen) / 2.0  # stabiler 2er-Zyklus erkannt
        vorletzte = flaechen
        flaechen = neu
    return flaechen


def masse_bei_geometrie(
    struktur: TrussStructure, material: Material, a_min: float, a_max: float, max_iter: int = 150,
) -> float:
    """Löst das innere Sizing-Problem und gibt nur die resultierende Masse
    zurück - das ist die (äußere) Zielfunktion der Formoptimierung.

    `max_iter` wird während der äußeren Suche bewusst NIEDRIG gehalten
    (siehe `ts_solver.py`): bei den doppelt statisch unbestimmten Kragarm-
    Formen (X-Aussteifung, dieselbe Struktur wie in truss-sizing-demo)
    oszilliert die Vollspannungs-Entwurf-Fixpunktiteration für viele
    Formen, statt sauber zu konvergieren, und schöpft sonst bei JEDER
    Zielfunktionsauswertung während Hunderter Suchschritte das volle
    Iterationslimit aus (live gemessen: 25s statt <3s Solve-Zeit). Für die
    abschließende Massen-Angabe wird stattdessen mit dem hohen Standardwert
    (150) nachgerechnet."""
    flaechen = vollspannungs_entwurf(struktur, material, a_min, a_max, max_iter=max_iter)
    return gesamtmasse(struktur, flaechen, material)
