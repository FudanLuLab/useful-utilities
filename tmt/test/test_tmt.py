from operator import attrgetter

import numpy as np
import pytest
from attrs import define

import tmt


@define
class TestClass:
    a: float
    b: int


@pytest.mark.parametrize(
    "x, a, key, expected",
    [
        (4, [1, 2, 3, 7, 8], None, 3),
        (0, [1, 2, 3, 7, 8], None, 1),
        (10, [1, 2, 3, 7, 8], None, 8),
        (3, [1, 2, 3, 7, 8], None, 3),
        (4, [TestClass(1, 0), TestClass(2, 0), TestClass(5, 0)],
         attrgetter('a'), TestClass(5, 0)),
    ]
)
def test_most_closed(x, a, key, expected):
    result = tmt.most_closed(x, a, key=key)
    assert result == expected


@pytest.mark.parametrize(
    "x, a, key, expected",
    [
        (4, [1, 2, 3, 7, 8], None, 2),
        (0, [1, 2, 3, 7, 8], None, 0),
        (10, [1, 2, 3, 7, 8], None, 4),
        (3, [1, 2, 3, 7, 8], None, 2),
        (4, [TestClass(1, 0), TestClass(2, 0), TestClass(5, 0)], attrgetter('a'), 2),
    ]
)
def test_index_most_closed(x, a, key, expected):
    result = tmt.index_most_closed(x, a, key=key)
    assert result == expected


def test_ppm():
    ppm = tmt.Ppm(10)
    result = ppm(1000)
    expected = 0.01
    assert result == expected


class TestDDAList:

    @pytest.fixture
    def dda_list(self):
        data = [
            tmt.Ion(1000, 3),
            tmt.Ion(2000, 3),
            tmt.Ion(3000, 4)
        ]
        return tmt.DDAList(data)

    def test_from_skyline_csv(self, dda_list):
        filepath = "data/test_dda_list.csv"
        result = tmt.DDAList.from_skyline_csv(filepath)
        expected = dda_list
        assert result == expected

    @pytest.mark.parametrize(
        "mz, tol, charge, expected",
        [
            (1000.5, 1, 3, tmt.Ion(1000, 3)),
            (1000.5, 1, 4, None),
            (3000, 1, 4, tmt.Ion(3000, 4)),
            (1000.5, tmt.Ppm(1000), 3, tmt.Ion(1000, 3))
        ]
    )
    def test_find(self, dda_list, mz, tol, charge, expected):
        result = dda_list.find(mz, tol, charge)
        assert result == expected


class TestSpectrum:

    @pytest.fixture
    def spec(self):
        return tmt.Spectrum(1000, 3, np.array([100, 200, 300]), np.array([1e6, 2e6, 3e6]))

    @pytest.mark.parametrize(
        "mz, tol, expected",
        [
            (500, 1, 0),
            (100.5, 1, 1e6),
            (201, 5, 2e6),
            (100.5, tmt.Ppm(10), 0),
            (100.00001, tmt.Ppm(10), 1e6)
        ]
    )
    def test_get_intensity(self, spec, mz, tol, expected):
        result = spec.get_intensity(mz, tol)
        assert result == expected
