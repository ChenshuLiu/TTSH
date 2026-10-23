"""Patient clinical covariates, with training-patient-only preprocessing.

Only explicitly supported numeric, non-outcome variables enter the model. IDs
remain join keys; HbA1c, diagnosis, glucose summaries, and sex codes are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd


CLINICAL_FEATURES = ("age", "bmi", "height_cm", "weight_kg", "sbp", "dbp")
CLINICAL_ALIASES = {
    "age": ("age", "patient.age", "age_years"),
    "bmi": ("bmi", "patient.bmi", "body_mass_index"),
    "height_cm": ("height_cm", "height", "patient.height", "patient.height_cm"),
    "weight_kg": ("weight_kg", "weight", "patient.weight", "patient.weight_kg"),
    "sbp": ("sbp", "patient.sbp", "systolic_bp", "systolic_blood_pressure"),
    "dbp": ("dbp", "patient.dbp", "diastolic_bp", "diastolic_blood_pressure"),
}
SUBJECT_ALIASES = ("subject_id", "subjectId", "patient_id", "participant_id", "userID", "subject", "id")


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _subject(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _numeric(values: pd.Series, column: str) -> pd.Series:
    blank = values.isna() | values.astype(str).str.strip().eq("")
    numbers = pd.to_numeric(values.where(~blank), errors="coerce")
    invalid = ~blank & numbers.isna()
    if invalid.any():
        raise ValueError(f"Clinical column {column!r} must be numeric; contains nonnumeric values. "
                         "Categorical codes require a confirmed encoding and are not supported here.")
    return numbers.astype(float).where(np.isfinite(numbers), np.nan)


def _forbidden_source(column: str) -> bool:
    key = _key(column)
    return (key in {_key(c) for c in SUBJECT_ALIASES} or key.endswith("id") or
            any(token in key for token in ("hba1c", "a1c", "diagnos", "diabetes", "t2d", "prediabet",
                                            "glucos", "glucotyp", "cgm", "label", "target", "class",
                                            "fbg", "ogtt", "insulin", "gender", "sex")) or key in {"fg", "patientfg"})


def _matches(frame: pd.DataFrame, aliases: Sequence[str]) -> list[str]:
    keys = {_key(alias) for alias in aliases}
    return [c for c in frame.columns if _key(c) in keys]


def has_clinical_columns(frame: pd.DataFrame, mapping: dict[str, str] | None = None) -> bool:
    """Whether a table has recognized covariates, without touching its outcomes."""
    mapping = mapping or {}
    return any(_matches(frame, aliases) or (mapping.get(feature) in frame)
               for feature, aliases in CLINICAL_ALIASES.items())


def canonicalize_clinical(frame: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Return one row per subject and only present, safe clinical columns.

    Repeated patients/events may repeat the same covariates or supply a missing
    value. Conflicting finite values and conflicting ID aliases fail explicitly.
    Height is interpreted as cm and weight as kg; no units are guessed.
    """
    mapping = {k: v for k, v in (mapping or {}).items() if v}
    unknown = set(mapping) - {"subject", *CLINICAL_FEATURES}
    if unknown:
        raise ValueError(f"Unsupported clinical feature mapping: {sorted(unknown)}")
    if frame.columns.duplicated().any():
        raise ValueError("Clinical table contains duplicate column names")
    if mapping.get("subject"):
        subjects = [mapping["subject"]]
        if subjects[0] not in frame:
            raise ValueError(f"Mapped subject column {subjects[0]!r} is absent")
    else:
        subjects = _matches(frame, SUBJECT_ALIASES)
    if not subjects:
        raise ValueError("Clinical metadata requires a patient ID column")
    subject = "subject_id" if "subject_id" in subjects else subjects[0]
    ids = frame[subject].map(_subject)
    if any(not ids.equals(frame[c].map(_subject)) for c in subjects):
        raise ValueError(f"Ambiguous clinical subject columns {subjects}: values disagree; provide an explicit mapping")
    if ids.eq("").any():
        raise ValueError("Clinical metadata contains an empty subject ID")
    out = pd.DataFrame({"subject_id": ids}, index=frame.index)
    for feature, aliases in CLINICAL_ALIASES.items():
        if feature in mapping:
            source = mapping[feature]
            if _forbidden_source(source):
                raise ValueError(f"Forbidden clinical source {source!r}: identifiers, targets, diagnosis, and glucose variables cannot be model features")
            if source not in frame:
                raise ValueError(f"Mapped clinical column {source!r} is absent")
            columns = [source]
        else:
            columns = _matches(frame, aliases)
        if not columns:
            continue
        values = pd.concat([_numeric(frame[c], str(c)).rename(str(c)) for c in columns], axis=1)
        if values.nunique(axis=1, dropna=True).gt(1).any():
            raise ValueError(f"Conflicting clinical aliases for {feature}: {columns}")
        out[feature] = values.bfill(axis=1).iloc[:, 0]
    features = [c for c in CLINICAL_FEATURES if c in out]
    if out.empty:
        return out.reset_index(drop=True)
    if features:
        conflicts = out.groupby("subject_id", sort=False)[features].nunique(dropna=True)
        if conflicts.gt(1).any().any():
            conflicting_features = conflicts.columns[conflicts.gt(1).any()].tolist()
            raise ValueError(f"Conflicting clinical metadata for repeated patients: {conflicting_features}")
        # first() coalesces missing values but never averages inconsistent rows.
        return out.groupby("subject_id", sort=False, as_index=False)[features].first().reset_index(drop=True)
    return out.drop_duplicates("subject_id").reset_index(drop=True)


@dataclass
class MetadataPreprocessor:
    features: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    fitted_subject_ids: list[str]

    @property
    def dim(self) -> int:
        return 2 * len(self.features)

    @property
    def feature_names(self) -> list[str]:
        return [f"{f}_z" for f in self.features] + [f"{f}_missing" for f in self.features]

    @property
    def feature_groups(self) -> dict[str, list[int]]:
        return {f: [i, i + len(self.features)] for i, f in enumerate(self.features)}

    @property
    def baseline_vector(self) -> np.ndarray:
        return np.concatenate([(self.medians - self.means) / self.scales,
                               np.zeros(len(self.features))]).astype(np.float32)

    @staticmethod
    def _validate_features(features: Sequence[str]) -> list[str]:
        features = list(features)
        if not features or len(set(features)) != len(features):
            raise ValueError("Select at least one unique clinical feature")
        unsupported = set(features) - set(CLINICAL_FEATURES)
        if unsupported:
            raise ValueError(f"Unsupported or forbidden clinical features: {sorted(unsupported)}")
        return features

    @staticmethod
    def _values(frame: pd.DataFrame, features: list[str], subject_ids: Sequence[str]) -> np.ndarray:
        canonical = canonicalize_clinical(frame)
        absent = set(features) - set(canonical.columns)
        if absent:
            raise ValueError(f"Selected clinical columns are absent: {sorted(absent)}; provide them or retrain with available features")
        return canonical.set_index("subject_id").reindex([_subject(s) for s in subject_ids])[features].to_numpy(float)

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: list[str], train_ids: Sequence[str]) -> "MetadataPreprocessor":
        features = cls._validate_features(features)
        ids = list(dict.fromkeys(_subject(s) for s in train_ids))
        if not ids or "" in ids:
            raise ValueError("Clinical preprocessing requires nonempty training subject IDs")
        values = cls._values(frame, features, ids)
        missing = np.isnan(values)
        absent = [f for i, f in enumerate(features) if missing[:, i].all()]
        if absent:
            raise ValueError(f"Clinical columns have no finite values in training patients: {absent}")
        medians = np.nanmedian(values, axis=0)
        imputed = np.where(missing, medians, values)
        means, scales = imputed.mean(axis=0), imputed.std(axis=0)
        scales = np.where(scales > 1e-8, scales, 1.0)
        return cls(features, medians, means, scales, ids)

    def transform(self, frame: pd.DataFrame, subject_ids: Sequence[str]) -> np.ndarray:
        values = self._values(frame, self.features, subject_ids)
        missing = np.isnan(values)
        imputed = np.where(missing, self.medians, values)
        return np.concatenate([(imputed - self.means) / self.scales,
                               missing.astype(float)], axis=1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "features": list(self.features), "medians": self.medians.tolist(),
                "means": self.means.tolist(), "scales": self.scales.tolist(),
                "fitted_subject_ids": list(self.fitted_subject_ids),
                "feature_names": self.feature_names,
                "units": {f: {"age": "years", "bmi": "kg/m2", "height_cm": "cm", "weight_kg": "kg",
                              "sbp": "mmHg", "dbp": "mmHg"}[f] for f in self.features}}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "MetadataPreprocessor":
        if state.get("version") != 1:
            raise ValueError("Unsupported clinical preprocessor version")
        features = cls._validate_features(state["features"])
        arrays = [np.asarray(state[k], dtype=float) for k in ("medians", "means", "scales")]
        if any(a.shape != (len(features),) or not np.isfinite(a).all() for a in arrays) or (arrays[2] <= 0).any():
            raise ValueError("Invalid clinical preprocessor statistics")
        ids = [_subject(s) for s in state["fitted_subject_ids"]]
        if not ids or "" in ids or len(set(ids)) != len(ids):
            raise ValueError("Invalid fitted training subject IDs")
        return cls(features, *arrays, ids)
