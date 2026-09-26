"""Safe restoration of known privacy tokens in generated text."""

import re
from collections.abc import Mapping

from app.models.privacy import TokenMapping

_TOKEN_PATTERN = re.compile(r"<PII_[A-Z_]+:[0-9a-f]{32}>")


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
