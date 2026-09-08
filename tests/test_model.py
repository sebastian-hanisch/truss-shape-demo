import numpy as np
import pytest

from ts_constants import DEFAULT_MATERIAL, MATERIALS
from ts_model import (
    InstabilesTragwerk,
    Knoten,
    Last,
    Stab,
    TrussStructure,
    gesamtmasse,
    loese_tragwerk,
)
from ts_networks import VORLAGEN, VORLAGEN_NAMEN

MAT = MATERIALS[DEFAULT_MATERIAL]


def test_dreieck_stabkraefte_stimmen_mit_handrechnung_ueberein():
    knoten = [
        Knoten(0, 0.0, 0.0, fest_x=True, fest_y=True),
        Knoten(1, 4.0, 0.0, fest_y=True),
        Knoten(2, 2.0, 3.0),
    ]
    staebe = [Stab(0, 0, 1), Stab(1, 0, 2), Stab(2, 1, 2)]
    lasten = [Last(2, 0.0, -50_000.0)]
    struktur = TrussStructure("Dreieck", knoten, staebe, lasten)
    erg = loese_tragwerk(struktur, np.full(3, 1e-3), MAT)
    erwartet = np.array([16666.6667, -30046.2606, -30046.2606])
    np.testing.assert_allclose(erg.stabkraefte, erwartet, rtol=1e-4)


def test_instabiles_tragwerk_wird_erkannt():
    knoten = [Knoten(0, 0.0, 0.0, fest_x=True, fest_y=True), Knoten(1, 1.0, 0.0)]
    staebe = [Stab(0, 0, 1)]
    lasten = [Last(1, 0.0, -1000.0)]
    struktur = TrussStructure("Instabil", knoten, staebe, lasten)
    with pytest.raises(InstabilesTragwerk):
        loese_tragwerk(struktur, np.full(1, 1e-3), MAT)


@pytest.mark.parametrize("name", VORLAGEN_NAMEN)
def test_vorlage_ist_stabil_am_standardwert(name):
    vorlage = VORLAGEN[name]
    struktur = vorlage.baue_struktur(vorlage.t_defaults())
    erg = loese_tragwerk(struktur, np.full(len(struktur.staebe), 5e-3), MAT)
    assert np.isfinite(erg.max_verschiebung)


@pytest.mark.parametrize("name", VORLAGEN_NAMEN)
def test_schnitt_position_stimmt_mit_t_wert_ueberein(name):
    vorlage = VORLAGEN[name]
    t = vorlage.t_defaults()
    struktur = vorlage.baue_struktur(t)
    for schnitt, t_wert in zip(vorlage.schnitte, t):
        knoten = struktur.knoten[schnitt.knoten_id]
        erwartete_position = schnitt.position(t_wert)
        assert (knoten.x, knoten.y) == pytest.approx(erwartete_position)


@pytest.mark.parametrize("name", VORLAGEN_NAMEN)
def test_lastfaktor_skaliert_lasten_linear(name):
    vorlage = VORLAGEN[name]
    t = vorlage.t_defaults()
    struktur1 = vorlage.baue_struktur(t, lastfaktor=1.0)
    struktur2 = vorlage.baue_struktur(t, lastfaktor=2.0)
    for l1, l2 in zip(struktur1.lasten, struktur2.lasten):
        assert l2.fx == pytest.approx(2 * l1.fx)
        assert l2.fy == pytest.approx(2 * l1.fy)


def test_masse_ist_positiv_und_skaliert_mit_flaeche():
    v = VORLAGEN["Kompaktes Dreieck (Höhe)"]
    struktur = v.baue_struktur(v.t_defaults())
    m1 = gesamtmasse(struktur, np.full(len(struktur.staebe), 1e-3), MAT)
    m2 = gesamtmasse(struktur, np.full(len(struktur.staebe), 2e-3), MAT)
    assert m1 > 0
    assert m2 == pytest.approx(2 * m1)
