"""Physikalische Konstanten und Werkstoffe - dieselbe Modellierung wie in
truss-sizing-demo (Vollkreis-Querschnitt, Euler-Knicken), hier aber nur mit
kontinuierlichen Querschnittsflächen (keine Katalogrundung - das war das
Thema der Sizing-Demo, hier liegt der Fokus auf der Geometrie)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    e_modul: float  # Elastizitätsmodul [Pa]
    dichte: float  # Dichte [kg/m^3]
    sigma_zul: float  # zulässige Spannung (Zug & Druck, ohne Knicken) [Pa]


MATERIALS: dict[str, Material] = {
    "Baustahl S235": Material("Baustahl S235", 210e9, 7850.0, 160e6),
    "Aluminium EN AW-6060": Material("Aluminium EN AW-6060", 70e9, 2700.0, 110e6),
}
DEFAULT_MATERIAL = "Baustahl S235"

KNICK_BEIWERT_K = 1.0


def flaechenmoment(area: float) -> float:
    """Flächenträgheitsmoment I(A) für einen Vollkreisquerschnitt der Fläche A."""
    import numpy as np

    return area ** 2 / (4.0 * np.pi)


A_MIN = 1.0e-4  # 1 cm^2
A_MAX = 2.0e-2  # 200 cm^2
