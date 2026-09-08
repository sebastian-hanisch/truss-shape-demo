import numpy as np
import pytest

from ts_constants import DEFAULT_MATERIAL, MATERIALS
from ts_networks import VORLAGEN, VORLAGEN_NAMEN
from ts_solver import (
    _endgueltige_masse,
    koordinatensuche,
    kontinuierliche_optimierung,
    loese_alle,
    metaheuristik,
)

MAT = MATERIALS[DEFAULT_MATERIAL]


@pytest.mark.parametrize("name", VORLAGEN_NAMEN)
def test_t_werte_liegen_innerhalb_der_schnitt_grenzen(name):
    vorlage = VORLAGEN[name]
    lo, hi = vorlage.t_bounds()
    for ergebnis in loese_alle(vorlage, MAT, lastfaktor=1.0, seed=0).values():
        assert np.all(ergebnis.t_werte >= lo - 1e-6)
        assert np.all(ergebnis.t_werte <= hi + 1e-6)


@pytest.mark.parametrize("name", VORLAGEN_NAMEN)
def test_optimierte_masse_ist_nie_schlechter_als_standardform(name):
    vorlage = VORLAGEN[name]
    default_masse = _endgueltige_masse(vorlage, MAT, 1.0, vorlage.t_defaults())
    for methode_name, ergebnis in loese_alle(vorlage, MAT, lastfaktor=1.0, seed=0).items():
        assert ergebnis.masse <= default_masse + 1e-6, f"{name}/{methode_name} ist schlechter als die Standardform"


@pytest.mark.parametrize("name", VORLAGEN_NAMEN)
def test_slsqp_und_metaheuristik_finden_aehnliche_masse(name):
    vorlage = VORLAGEN[name]
    slsqp = kontinuierliche_optimierung(vorlage, MAT, 1.0)
    meta = metaheuristik(vorlage, MAT, 1.0, seed=0)
    # beide sollten sich auf < 3% annähern - sie lösen dasselbe Problem mit
    # unterschiedlichen globalen/lokalen Strategien.
    assert abs(slsqp.masse - meta.masse) / slsqp.masse < 0.03


def test_koordinatensuche_verfehlt_kopplung_beim_kragarm():
    """Kernbehauptung der Demo: beim doppelt statisch unbestimmten Kragarm
    (Kopplung zwischen den beiden Höhen-Parametern) verpasst die
    Koordinatensuche einen Teil der Verbesserung, die SLSQP/DE finden,
    weil sie nie beide Koordinaten gleichzeitig verändert."""
    vorlage = VORLAGEN["10-Stab-Kragarm (Höhenprofil)"]
    koord = koordinatensuche(vorlage, MAT, 1.0)
    slsqp = kontinuierliche_optimierung(vorlage, MAT, 1.0)
    assert koord.masse > slsqp.masse * 1.02


def test_dreieck_hat_keine_kopplung_alle_methoden_gleich():
    """Gegenprobe: bei nur einem freien Knoten (1D) kann es per Definition
    keine Kopplung geben - alle drei Methoden sollten dasselbe Optimum finden."""
    vorlage = VORLAGEN["Kompaktes Dreieck (Höhe)"]
    ergebnisse = loese_alle(vorlage, MAT, lastfaktor=1.0, seed=0)
    massen = [r.masse for r in ergebnisse.values()]
    assert max(massen) - min(massen) < 0.5


def test_bruecke_hoehenprofil_ist_in_der_mitte_am_tiefsten():
    """Klassisches Ergebnis der Formoptimierung: bei einer einfach gelagerten
    Brücke unter symmetrischer Last folgt die optimale Tragwerkstiefe dem
    Biegemomentenverlauf - am größten in Feldmitte, am kleinsten über den
    Auflagern (Bogen-/Fischbauchträger-Prinzip)."""
    vorlage = VORLAGEN["Pratt-Brückenfachwerk (Höhenprofil)"]
    ergebnis = kontinuierliche_optimierung(vorlage, MAT, 1.0)
    # Schnitte sind in Feldreihenfolge: [Auflager, Viertel, Mitte, Viertel, Auflager]
    t = ergebnis.t_werte
    mitte = t[len(t) // 2]
    assert mitte > t[0]
    assert mitte > t[-1]


def test_turm_verjuengt_sich_nach_oben():
    """Klassisches Ergebnis: ein schlanker, oben belasteter Mast wird zur
    Spitze hin schmaler (Battered-Tower-Prinzip), nicht breiter."""
    vorlage = VORLAGEN["Turmfachwerk (Verjüngungsprofil)"]
    ergebnis = metaheuristik(vorlage, MAT, 1.0, seed=0)
    t = ergebnis.t_werte
    assert t[0] > t[-1]
