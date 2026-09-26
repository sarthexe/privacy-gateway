"""Typed domain models for privacy transformations."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityType(StrEnum):
    """PII entity types supported by the initial detector."""

    PERSON = "PERSON"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_TIME = "DATE_TIME"
    LOCATION = "LOCATION"


class EntitySpan(BaseModel):
    """A detected entity and its half-open character offsets in source text."""

    model_config = ConfigDict(frozen=True)

    entity_type: EntityType
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    value: str

    @model_validator(mode="after")
    def validate_span(self) -> "EntitySpan":
        if self.end <= self.start or len(self.value) != self.end - self.start:
            raise ValueError("Entity span offsets must match the entity value length")
        return self


class TokenMapping(BaseModel):
    """In-memory reversible mapping for one distinct entity value."""

    model_config = ConfigDict(frozen=True)

    token: str
    entity_type: EntityType
    value: str


class TokenizationResult(BaseModel):
    """Tokenized text and mappings required to reconstruct the source."""

    text: str
    mappings: dict[str, TokenMapping]
