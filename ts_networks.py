"""Formvorlagen - dieselben Grundtopologien wie in truss-sizing-demo, hier
aber mit definierten Bewegungslinien ("Schnitten") für einzelne Knoten:
Höhenprofil bei Brücke/Kragarm/Dreieck, Breitenprofil (Verjüngung) beim Mast.
"""

from dataclasses import dataclass

from ts_model import Formvorlage, Knoten, Last, Schnitt, Stab


def _kragarm_10_stab() -> Formvorlage:
    basis_knoten = [
        Knoten(0, 0.0, 9.0, fest_x=True, fest_y=True),  # oben links: Lager, NICHT auf Schnitt
        Knoten(1, 0.0, 0.0, fest_x=True, fest_y=True),  # unten links: Lager
        Knoten(2, 9.0, 9.0),   # oben mitte: Schnitt (Höhe)
        Knoten(3, 9.0, 0.0),   # unten mitte: fest
        Knoten(4, 18.0, 9.0),  # oben rechts: Schnitt (Höhe)
        Knoten(5, 18.0, 0.0),  # unten rechts: fest, Last
    ]
    paare = [(0, 2), (2, 4), (1, 3), (3, 5), (2, 3), (4, 5), (0, 3), (1, 2), (2, 5), (3, 4)]
    staebe = [Stab(i, a, b) for i, (a, b) in enumerate(paare)]
    lasten = [Last(3, 0.0, -60_000.0), Last(5, 0.0, -60_000.0)]
    schnitte = [
        Schnitt(2, 9.0, 0.0, 0.0, 1.0, t_min=2.0, t_max=12.0, t_default=9.0),
        Schnitt(4, 18.0, 0.0, 0.0, 1.0, t_min=2.0, t_max=12.0, t_default=9.0),
    ]
    return Formvorlage("10-Stab-Kragarm (Höhenprofil)", basis_knoten, staebe, lasten, schnitte)


def _bruecke_pratt() -> Formvorlage:
    n_felder = 4
    feldlaenge = 4.0
    hoehe0 = 3.0
    basis_knoten = []
    for i in range(n_felder + 1):
        fest_x = i == 0
        fest_y = i in (0, n_felder)
        basis_knoten.append(Knoten(2 * i, i * feldlaenge, 0.0, fest_x=fest_x, fest_y=fest_y))
        basis_knoten.append(Knoten(2 * i + 1, i * feldlaenge, hoehe0))
    staebe = []
    sid = 0
    for i in range(n_felder):
        staebe.append(Stab(sid, 2 * i, 2 * (i + 1))); sid += 1
        staebe.append(Stab(sid, 2 * i + 1, 2 * (i + 1) + 1)); sid += 1
    for i in range(n_felder + 1):
        staebe.append(Stab(sid, 2 * i, 2 * i + 1)); sid += 1
    for i in range(n_felder):
        staebe.append(Stab(sid, 2 * i, 2 * (i + 1) + 1)); sid += 1
    lasten = [Last(2 * i, 0.0, -80_000.0) for i in range(1, n_felder)]
    schnitte = [
        Schnitt(2 * i + 1, i * feldlaenge, 0.0, 0.0, 1.0, t_min=1.0, t_max=7.0, t_default=hoehe0)
        for i in range(n_felder + 1)
    ]
    return Formvorlage("Pratt-Brückenfachwerk (Höhenprofil)", basis_knoten, staebe, lasten, schnitte)


def _turm_mast() -> Formvorlage:
    n_seg = 5
    seg_hoehe = 3.0
    breite0 = 2.0
    basis_knoten = []
    for i in range(n_seg + 1):
        fest = i == 0
        basis_knoten.append(Knoten(2 * i, 0.0, i * seg_hoehe, fest_x=fest, fest_y=fest))
        basis_knoten.append(Knoten(2 * i + 1, breite0, i * seg_hoehe, fest_x=fest, fest_y=fest))
    staebe = []
    sid = 0
    for i in range(n_seg):
        staebe.append(Stab(sid, 2 * i, 2 * (i + 1))); sid += 1
        staebe.append(Stab(sid, 2 * i + 1, 2 * (i + 1) + 1)); sid += 1
        staebe.append(Stab(sid, 2 * i, 2 * (i + 1) + 1)); sid += 1
    for i in range(n_seg + 1):
        staebe.append(Stab(sid, 2 * i, 2 * i + 1)); sid += 1
    top_l, top_r = 2 * n_seg, 2 * n_seg + 1
    lasten = [Last(top_l, 60_000.0, 0.0), Last(top_r, 60_000.0, 0.0)]
    # Rechter Gurt (i=1..n_seg) darf sich horizontal verjüngen; linker Gurt und
    # Fußpunkt (i=0) sind fest - das Fundament bleibt an seiner Position.
    schnitte = [
        Schnitt(2 * i + 1, 0.0, i * seg_hoehe, 1.0, 0.0, t_min=0.5, t_max=3.0, t_default=breite0)
        for i in range(1, n_seg + 1)
    ]
    return Formvorlage("Turmfachwerk (Verjüngungsprofil)", basis_knoten, staebe, lasten, schnitte)


def _dreieck_kompakt() -> Formvorlage:
    basis_knoten = [
        Knoten(0, 0.0, 0.0, fest_x=True, fest_y=True),
        Knoten(1, 4.0, 0.0, fest_y=True),
        Knoten(2, 2.0, 3.0),
    ]
    staebe = [Stab(0, 0, 1), Stab(1, 0, 2), Stab(2, 1, 2)]
    lasten = [Last(2, 0.0, -50_000.0)]
    schnitte = [Schnitt(2, 2.0, 0.0, 0.0, 1.0, t_min=1.0, t_max=6.0, t_default=3.0)]
    return Formvorlage("Kompaktes Dreieck (Höhe)", basis_knoten, staebe, lasten, schnitte)


VORLAGEN: dict[str, Formvorlage] = {
    v.name: v for v in [_kragarm_10_stab(), _bruecke_pratt(), _turm_mast(), _dreieck_kompakt()]
}
VORLAGEN_NAMEN = list(VORLAGEN.keys())
