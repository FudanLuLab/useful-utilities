import bisect
import csv
import re
from operator import attrgetter
from pathlib import Path
from typing import TypeVar, Sequence, Callable, Optional, Iterable, Collection

import click
from attrs import define, frozen, field
from numpy.typing import NDArray
from pyteomics import mgf

REPORTERS = [
    126.127726,
    127.124761,
    127.131081,
    128.128116,
    128.134436,
    129.131471,
    129.137790,
    130.134825,
    130.141145,
    131.138180,
]


@click.command()
@click.argument("file1")
@click.argument("file2")
@click.option("--ms1_ppm", default=20)
@click.option("--ms2_ppm", default=20)
def app(file1, file2, ms1_ppm, ms2_ppm):
    mgf_file, list_file = distin_files(file1, file2)
    result_file = mgf_file.with_name(f"result_{mgf_file.stem}.csv")
    search(mgf_file, list_file, result_file, ms1_ppm, ms2_ppm)


def distin_files(file1, file2) -> tuple[Path, Path]:
    file1, file2 = Path(file1), Path(file2)
    if file1.suffix == '.mgf' and file2.suffix == '.csv':
        return file1, file2
    elif file1.suffix == '.csv' and file2.suffix == '.mgf':
        return file2, file1
    else:
        raise ValueError("Wrong file types.")


def search(
    mgf_file: Path,
    list_file: Path,
    result_file: Path,
    ms1_ppm: float,
    ms2_ppm: float,
):
    dda_list = DDAList.from_skyline_csv(list_file)
    searcher = ReporterSearcher(dda_list, ms1_tol=Ppm(ms1_ppm), ms2_tol=Ppm(ms2_ppm))
    mgf_reader = MGFReader(mgf_file)
    with CSVExporter(result_file, SearchRecord.fields()) as exporter:
        for record in searcher.run(mgf_reader):
            exporter.feed(record)
    print(f"Result saved in {result_file}.")


def index_most_closed(x, a: Sequence, *, key: Optional[Callable] = None) -> int:
    """
    Returns the index of the element in ``a`` most closed to ``x``.

    Args:
        x: The element.
        a: A Sequence of element.
        key: Optional. If provided, ``key`` will be called on each element in ``a``
            before compared to x.

    """
    bisect_i = bisect.bisect(a, x, key=key)
    if bisect_i == 0:
        return 0
    if bisect_i == len(a):
        return len(a) - 1
    if key is None:
        min_i = min([bisect_i, bisect_i - 1], key=lambda i: abs(x - a[i]))
    else:
        min_i = min([bisect_i, bisect_i - 1], key=lambda i: abs(x - key(a[i])))
    return min_i


T = TypeVar("T")


def most_closed(x, a: Sequence[T], *, key: Optional[Callable] = None) -> T:
    """
    Returns the element in ``a`` most closed to ``x``.

    Args:
        x: The element.
        a: A Sequence of element.
        key: Optional. If provided, ``key`` will be called on each element in ``a``
            before compared to x.

    """
    return a[index_most_closed(x, a, key=key)]


@define
class Ppm:
    """
    Examples:
        >>> ppm = Ppm(10)
        >>> ppm(1000)
        0.01

    """

    _ppm: float

    def __call__(self, mz: float) -> float:
        return self._ppm * mz / 1e6

    def __repr__(self):
        return f"Ppm({self._ppm})"


@frozen
class Ion:
    mz: float = field(converter=float)
    charge: int = field(converter=int)


@define
class DDAList:
    _data: list[Ion] = field(
        factory=list, converter=lambda l: sorted(l, key=attrgetter("mz"))
    )

    @classmethod
    def from_skyline_csv(cls, filepath: str | Path):
        with open(filepath, encoding="utf8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            data = [Ion(mz, charge) for mz, charge, *_ in reader]
        return cls(data)

    def find(self, mz: float, tol: float | Ppm, charge: int) -> Ion | None:
        """
        Find ion in the DDA List with similar m/z and the same charge.

        Args:
            mz: The m/z value.
            tol: The tolerance of the m/z value.
            charge: The charge.

        Returns:
            ion: The found Ion.

        """
        candidate = most_closed(mz, self._data, key=attrgetter("mz"))
        tol = tol(mz) if isinstance(tol, Ppm) else tol
        if abs(candidate.mz - mz) <= tol and candidate.charge == charge:
            return candidate
        return None


@frozen
class Spectrum:
    scan: int
    precursor_mz: float
    precursor_charge: int
    mz_a: NDArray
    intensity_a: NDArray

    @classmethod
    def from_pyteomics_spec(cls, spec: dict):
        def extract_scan(title: str) -> int:
            m = re.search(r"scan=([0-9]+)", title)
            return int(m.group(1))

        return cls(
            scan=extract_scan(spec["params"]["title"]),
            precursor_mz=spec["params"]["pepmass"][0],
            precursor_charge=spec["params"]["charge"][0],
            mz_a=spec["m/z array"],
            intensity_a=spec["intensity array"],
        )

    def get_intensity(self, mz: float, tol: float | Ppm) -> float:
        """
        Returns the fragment peak near ``mz``.

        Args:
            mz: The m/z value.
            tol: The m/z tolerance.

        Returns:
            The intensity of the peak found. If no peak is found, 0 will be returned.

        Raises:
            ValueError: No peak is found.

        """
        tol = tol(mz) if isinstance(tol, Ppm) else tol
        peak_i = index_most_closed(mz, self.mz_a)
        if abs(self.mz_a[peak_i] - mz) < tol:
            return self.intensity_a[peak_i]
        return 0


@define
class MGFReader:
    _filepath: str | Path

    def __iter__(self):
        with mgf.read(str(self._filepath)) as reader:
            for spec in reader:
                yield Spectrum.from_pyteomics_spec(spec)


@frozen
class SearchRecord:
    scan: int
    precursor_mz: float
    reporters: dict

    def as_dict(self) -> dict:
        return {
            "precursor": self.precursor_mz,
            "scan": self.scan,
            **{str(k): v for k, v in self.reporters.items()},
        }

    @classmethod
    def fields(cls) -> tuple[str]:
        return ("scan", "precursor") + tuple(str(mz) for mz in REPORTERS)


@define
class ReporterSearcher:
    _dda_list: DDAList
    ms1_tol: float | Ppm
    ms2_tol: float | Ppm = field()

    @ms2_tol.validator
    def _check_ms2_tol(self, attribute, value):
        def min_diff(a: Iterable) -> float:
            a = sorted(a)
            min_ = float("Inf")
            for i in range(1, len(a) - 1):
                if diff := a[i] - a[i - 1] < min_:
                    min_ = diff
            return min_

        reporter_min_diff = min_diff(REPORTERS)
        tol = value(max(REPORTERS)) if isinstance(value, Ppm) else value
        if tol > reporter_min_diff:
            raise ValueError(f"MS2 tolerance too large: {value}.")

    def run(self, mgf_reader: MGFReader):
        for spec in mgf_reader:
            if precursor := self._dda_list.find(
                spec.precursor_mz, self.ms1_tol, spec.precursor_charge
            ):
                record = SearchRecord(
                    spec.scan,
                    precursor.mz,
                    {mz: spec.get_intensity(mz, tol=self.ms2_tol) for mz in REPORTERS},
                )
                yield record


@define(slots=False)
class CSVExporter:
    _filepath: str | Path
    fieldnames: Collection[str]

    def __enter__(self):
        self._file = open(self._filepath, "w", encoding="utf8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

    def feed(self, record: SearchRecord):
        self._writer.writerow(record.as_dict())
