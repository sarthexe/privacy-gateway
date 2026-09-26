"""Safe restoration of known privacy tokens in generated text."""

import re
from collections.abc import Mapping

from app.models.privacy import EntityType, TokenMapping

_TOKEN_PATTERN = re.compile(r"<PII_([A-Z_]+):[0-9a-f]{32}>")


def find_tokens(text: str) -> list[str]:
    """Return distinct syntactically valid tokens in order of first appearance."""
    return list(dict.fromkeys(match.group(0) for match in _TOKEN_PATTERN.finditer(text)))


def token_entity_type(token: str) -> EntityType | None:
    """Return the entity type named by a well-formed token, or ``None``."""
    match = _TOKEN_PATTERN.fullmatch(token)
    if match is None:
        return None
    try:
        return EntityType(match.group(1))
    except ValueError:
        return None


class Detokenizer:
    """Restore only syntactically valid tokens present in the supplied vault."""

    def detokenize(
        self,
        text: str,
        mappings: Mapping[str, TokenMapping],
    ) -> str:
        """Resolve known tokens; leave unknown and malformed text unchanged."""

        def replace(match: re.Match[str]) -> str:
            mapping = mappings.get(match.group(0))
            return mapping.value if mapping is not None else match.group(0)

        return _TOKEN_PATTERN.sub(replace, text)
