"""Stabtragwerk mit VARIABLER Geometrie: dieselbe Fachwerk-FEM (direkte
Steifigkeitsmethode) wie in truss-sizing-demo, hier aber mit Knoten, deren
Position sich entlang einer vorgegebenen Linie ("Schnitt") verschieben lässt
- das ist die Formoptimierung (Stufe 2 der Sizing->Form->Topologie-Hierarchie).
"""

from dataclasses import dataclass, field

import numpy as np

from ts_constants import KNICK_BEIWERT_K, Material, flaechenmoment


@dataclass(frozen=True)
class Knoten:
    id: int
    x: float
    y: float
    fest_x: bool = False
    fest_y: bool = False


@dataclass(frozen=True)
class Stab:
    id: int
    knoten1: int
    knoten2: int


@dataclass(frozen=True)
class Last:
    knoten: int
    fx: float
    fy: float


@dataclass(frozen=True)
class TrussStructure:
    name: str
    knoten: list[Knoten]
    staebe: list[Stab]
    lasten: list[Last] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.knoten)
        for s in self.staebe:
            if not (0 <= s.knoten1 < n and 0 <= s.knoten2 < n):
                raise ValueError(f"Stab {s.id} referenziert unbekannten Knoten")

    def stablaenge(self, stab: Stab) -> float:
        k1, k2 = self.knoten[stab.knoten1], self.knoten[stab.knoten2]
        return float(np.hypot(k2.x - k1.x, k2.y - k1.y))

    def stablaengen(self) -> np.ndarray:
        return np.array([self.stablaenge(s) for s in self.staebe])

    def freiheitsgrade(self) -> np.ndarray:
        frei = []
        for k in self.knoten:
            if not k.fest_x:
                frei.append(2 * k.id)
            if not k.fest_y:
                frei.append(2 * k.id + 1)
        return np.array(frei, dtype=int)

    def lastvektor(self) -> np.ndarray:
        n = len(self.knoten)
        f = np.zeros(2 * n)
        for last in self.lasten:
            f[2 * last.knoten] += last.fx
            f[2 * last.knoten + 1] += last.fy
        return f


@dataclass
class Tragwerksergebnis:
    stabkraefte: np.ndarray
    verschiebungen: np.ndarray
    max_verschiebung: float


class InstabilesTragwerk(Exception):
    pass


def _element_steifigkeit(e_modul: float, area: float, laenge: float, c: float, s: float) -> np.ndarray:
    k = e_modul * area / laenge
    t = np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ]
    )
    return k * t


def loese_tragwerk(struktur: TrussStructure, flaechen: np.ndarray, material: Material) -> Tragwerksergebnis:
    n_knoten = len(struktur.knoten)
    n_dof = 2 * n_knoten
    k_global = np.zeros((n_dof, n_dof))
    richtungen = []
    for stab, area in zip(struktur.staebe, flaechen):
        k1, k2 = struktur.knoten[stab.knoten1], struktur.knoten[stab.knoten2]
        laenge = struktur.stablaenge(stab)
        c = (k2.x - k1.x) / laenge
        s = (k2.y - k1.y) / laenge
        richtungen.append((c, s, laenge))
        k_e = _element_steifigkeit(material.e_modul, float(area), laenge, c, s)
        dofs = [2 * stab.knoten1, 2 * stab.knoten1 + 1, 2 * stab.knoten2, 2 * stab.knoten2 + 1]
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                k_global[i_global, j_global] += k_e[i_local, j_local]

    frei = struktur.freiheitsgrade()
    f = struktur.lastvektor()
    k_ff = k_global[np.ix_(frei, frei)]
    f_f = f[frei]

    if k_ff.size == 0 or np.linalg.cond(k_ff) > 1e12:
        raise InstabilesTragwerk("Tragwerk ist kinematisch instabil (Mechanismus) bei dieser Geometrie.")

    u = np.zeros(n_dof)
    u[frei] = np.linalg.solve(k_ff, f_f)

    stabkraefte = np.zeros(len(struktur.staebe))
    for idx, (stab, area) in enumerate(zip(struktur.staebe, flaechen)):
        c, s, laenge = richtungen[idx]
        dofs = [2 * stab.knoten1, 2 * stab.knoten1 + 1, 2 * stab.knoten2, 2 * stab.knoten2 + 1]
        u_e = u[dofs]
        dehnung = (-c, -s, c, s) @ u_e / laenge
        stabkraefte[idx] = material.e_modul * float(area) * dehnung

    return Tragwerksergebnis(stabkraefte, u, float(np.max(np.abs(u))))


def knick_lasten(struktur: TrussStructure, flaechen: np.ndarray, material: Material) -> np.ndarray:
    laengen = struktur.stablaengen()
    i_flaeche = np.array([flaechenmoment(a) for a in flaechen])
    return np.pi ** 2 * material.e_modul * i_flaeche / (KNICK_BEIWERT_K * laengen) ** 2


def constraint_verletzungen(struktur: TrussStructure, flaechen: np.ndarray, material: Material):
    ergebnis = loese_tragwerk(struktur, flaechen, material)
    spannungen = ergebnis.stabkraefte / flaechen
    spannungs_auslastung = np.abs(spannungen) / material.sigma_zul

    p_kr = knick_lasten(struktur, flaechen, material)
    druck = ergebnis.stabkraefte < 0
    knick_auslastung = np.zeros_like(flaechen)
    knick_auslastung[druck] = np.abs(ergebnis.stabkraefte[druck]) / p_kr[druck]

    tol = 1e-6
    erfuellt = bool(np.all(spannungs_auslastung <= 1 + tol) and np.all(knick_auslastung <= 1 + tol))
    return spannungs_auslastung, knick_auslastung, erfuellt


def gesamtmasse(struktur: TrussStructure, flaechen: np.ndarray, material: Material) -> float:
    return float(np.sum(material.dichte * flaechen * struktur.stablaengen()))


@dataclass(frozen=True)
class Schnitt:
    """Eine Bewegungslinie ("Schnitt"), auf der ein Knoten verschoben werden
    darf: Position(t) = (basis_x + t*richtung_x, basis_y + t*richtung_y)."""

    knoten_id: int
    basis_x: float
    basis_y: float
    richtung_x: float
    richtung_y: float
    t_min: float
    t_max: float
    t_default: float

    def position(self, t: float) -> tuple[float, float]:
        return (self.basis_x + t * self.richtung_x, self.basis_y + t * self.richtung_y)


@dataclass(frozen=True)
class Formvorlage:
    """Feste Topologie (Stäbe, Lasten, Lagerung) + welche Knoten sich auf
    welchem Schnitt bewegen dürfen. Alle nicht in `schnitte` genannten Knoten
    bleiben an ihrer festen Basis-Position."""

    name: str
    basis_knoten: list[Knoten]
    staebe: list[Stab]
    lasten: list[Last]
    schnitte: list[Schnitt]

    def anzahl_frei(self) -> int:
        return len(self.schnitte)

    def t_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([s.t_min for s in self.schnitte])
        hi = np.array([s.t_max for s in self.schnitte])
        return lo, hi

    def t_defaults(self) -> np.ndarray:
        return np.array([s.t_default for s in self.schnitte])

    def baue_struktur(self, t_werte: np.ndarray, lastfaktor: float = 1.0) -> TrussStructure:
        positionen = {s.knoten_id: s.position(t) for s, t in zip(self.schnitte, t_werte)}
        knoten = [
            Knoten(k.id, *positionen.get(k.id, (k.x, k.y)), fest_x=k.fest_x, fest_y=k.fest_y)
            for k in self.basis_knoten
        ]
        lasten = [Last(l.knoten, l.fx * lastfaktor, l.fy * lastfaktor) for l in self.lasten]
        return TrussStructure(self.name, knoten, self.staebe, lasten)
