"""Fail-closed adapters for public out-of-distribution PII datasets.

Normalized records retain source text solely in the external normalized dataset;
all annotation and manifest data are offset/label-only.  No entity values are
written outside the source-text record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scripts.evaluation_checkpoint import canonical_sha256
from scripts.gliner_evaluation import GlinerLabelMap

OOD_NORMALIZATION_SCHEMA = "privacy-gateway/ood-pii-normalized/v1"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repository: str
    revision: str
    source_url: str
    license: str
    has_language: bool


DATASETS = {
    "ai4privacy": DatasetSpec(
        "ai4privacy", "ai4privacy/pii-masking-300k", "c8c77895a005822682b66ab547fc0422579bc1d3",
        "https://huggingface.co/datasets/ai4privacy/pii-masking-300k", "license.md", True,
    ),
    "gretel": DatasetSpec(
        "gretel", "gretelai/gretel-pii-masking-en-v1", "e06eb1499ca8d54470f085021cd8e54f9efac7fd",
        "https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1", "Apache-2.0", True,
    ),
    "argilla": DatasetSpec(
        "argilla", "argilla/textcat-tokencat-pii-per-domain", "683954a23877151a1d05e7515d5e6cf87203ffd2",
        "https://huggingface.co/datasets/argilla/textcat-tokencat-pii-per-domain", "not specified", True,
    ),
}

# Source label -> GLiNER's pinned Nemotron prompt label. Missing and inherently
# ambiguous labels deliberately remain UNMAPPED. This supports a native score only
# for the subset that has an unambiguous common label space.
_COMMON = {
    "EMAIL": "email", "USERNAME": "user_name", "TEL": "phone_number", "PHONENUMBER": "phone_number",
    "FIRSTNAME": "first_name", "GIVENNAME1": "first_name", "GIVENNAME2": "first_name",
    "LASTNAME": "last_name", "LASTNAME1": "last_name", "LASTNAME2": "last_name", "DATE": "date",
    "TIME": "time", "STREET": "street_address", "ZIPCODE": "postcode", "CITY": "city",
    "STATE": "state", "COUNTRY": "country", "COMPANYNAME": "company_name", "ACCOUNTNUMBER": "account_number",
    "CREDITCARDNUMBER": "credit_debit_card", "CREDITCARDCVV": "cvv", "IPV4": "ipv4", "IPV6": "ipv6",
    "PASSWORD": "password", "PIN": "pin", "URL": "url", "SSN": "ssn", "DOB": "date_of_birth",
    "BIC": "swift_bic", "MAC": "mac_address", "VEHICLEVIN": "vehicle_identifier",
}
AI4PRIVACY_LABELS = _COMMON
ARGILLA_LABELS = _COMMON
GRETEL_LABELS = {
    "email": "email", "first_name": "first_name", "last_name": "last_name", "date": "date",
    "date_of_birth": "date_of_birth", "phone_number": "phone_number", "street_address": "street_address",
    "company_name": "company_name", "account_number": "account_number", "bank_routing_number": "bank_routing_number",
    "credit_card_number": "credit_debit_card", "customer_id": "customer_id", "employee_id": "employee_id",
    "ipv4": "ipv4", "ipv6": "ipv6", "medical_record_number": "medical_record_number",
    "ssn": "ssn", "user_name": "user_name", "unique_identifier": "unique_id", "url": "url",
    "postcode": "postcode", "city": "city", "country": "country", "state": "state", "time": "time",
    "date_time": "date_time", "device_identifier": "device_identifier", "password": "password",
    "pin": "pin", "api_key": "api_key", "vehicle_identifier": "vehicle_identifier",
}


class OodNormalizationError(ValueError):
    """An OOD annotation cannot be reproduced as a safe exact character span."""


def _offset(value: object, text: str, start: object, end: object) -> tuple[int, int]:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise OodNormalizationError("source annotation offsets are not integers")
    if start < 0 or end <= start or end > len(text):
        raise OodNormalizationError("source annotation offsets are outside text")
    if value is not None and (not isinstance(value, str) or text[start:end] != value):
        raise OodNormalizationError("source annotation value does not match its offsets")
    return start, end


def _unique_offset(text: str, value: object) -> tuple[int, int]:
    if not isinstance(value, str) or not value:
        raise OodNormalizationError("source entity value is missing")
    first = text.find(value)
    if first < 0 or text.find(value, first + 1) >= 0:
        raise OodNormalizationError("source entity cannot be uniquely reconstructed")
    return first, first + len(value)


def _entity(source_label: object, start: int, end: int, labels: Mapping[str, str], model: GlinerLabelMap) -> dict[str, Any]:
    if not isinstance(source_label, str) or not source_label:
        raise OodNormalizationError("source annotation label is missing")
    native = labels.get(source_label.upper(), labels.get(source_label, "UNMAPPED"))
    return {
        "type": model.gateway_type(native) if native != "UNMAPPED" else "UNMAPPED",
        "start": start,
        "end": end,
        "native_label": native,
        "source_label": source_label,
    }


def _record(index: int, text: object, entities: list[dict[str, Any]], spec: DatasetSpec, split: str, language: object) -> dict[str, Any]:
    if not isinstance(text, str):
        raise OodNormalizationError("source text is missing")
    if language is not None and not isinstance(language, str):
        raise OodNormalizationError("language metadata is invalid")
    return {
        "example_index": index,
        "text": text,
        "entities": [{key: item[key] for key in ("type", "start", "end")} for item in entities],
        "native_entities": [{key: item[key] for key in ("native_label", "start", "end")} for item in entities],
        "metadata": {"dataset": spec.repository, "revision": spec.revision, "split": split, "language": language},
    }


def normalize_ai4privacy(record: Mapping[str, Any], index: int, split: str, model: GlinerLabelMap) -> dict[str, Any]:
    text, spans = record.get("source_text"), record.get("privacy_mask")
    if not isinstance(text, str) or not isinstance(spans, list):
        raise OodNormalizationError("AI4Privacy requires source_text and privacy_mask")
    entities = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise OodNormalizationError("AI4Privacy privacy_mask item is invalid")
        start, end = _offset(span.get("value"), text, span.get("start"), span.get("end"))
        entities.append(_entity(span.get("label"), start, end, AI4PRIVACY_LABELS, model))
    return _record(index, text, entities, DATASETS["ai4privacy"], split, record.get("language"))


def normalize_gretel(record: Mapping[str, Any], index: int, split: str, model: GlinerLabelMap) -> dict[str, Any]:
    text, items = record.get("text"), record.get("entities")
    if not isinstance(text, str) or not isinstance(items, list):
        raise OodNormalizationError("Gretel requires text and entities")
    entities = []
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("types"), list):
            raise OodNormalizationError("Gretel entity item is invalid")
        start, end = _unique_offset(text, item.get("entity"))
        types = item["types"]
        label: object = types[0] if len(types) == 1 else "UNMAPPED"
        entities.append(_entity(label, start, end, GRETEL_LABELS, model))
    return _record(index, text, entities, DATASETS["gretel"], split, "en")


def normalize_argilla(record: Mapping[str, Any], index: int, split: str, model: GlinerLabelMap) -> dict[str, Any]:
    text, spans = record.get("source-text"), record.get("pii.suggestion")
    if not isinstance(text, str) or not isinstance(spans, list):
        raise OodNormalizationError("Argilla requires source-text and pii.suggestion")
    entities = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise OodNormalizationError("Argilla pii.suggestion item is invalid")
        start, end = _offset(None, text, span.get("start"), span.get("end"))
        entities.append(_entity(span.get("label"), start, end, ARGILLA_LABELS, model))
    return _record(index, text, entities, DATASETS["argilla"], split, record.get("language"))


def normalize(dataset: str, record: Mapping[str, Any], index: int, split: str, model: GlinerLabelMap) -> dict[str, Any]:
    adapters = {"ai4privacy": normalize_ai4privacy, "gretel": normalize_gretel, "argilla": normalize_argilla}
    try:
        return adapters[dataset](record, index, split, model)
    except KeyError as error:
        raise OodNormalizationError("unknown OOD dataset") from error


def manifest(spec: DatasetSpec, split: str, row_count: int, output_sha256: str, ontology: Mapping[str, str]) -> dict[str, Any]:
    return {"dataset_repository": spec.repository, "dataset_revision": spec.revision, "source_url": spec.source_url,
            "license": spec.license, "split": split, "row_count": row_count, "normalized_sha256": output_sha256,
            "normalization_schema": OOD_NORMALIZATION_SCHEMA, "ontology_sha256": canonical_sha256(dict(ontology))}
