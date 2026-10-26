import json
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from typing import Any

from libclaim.pricing.config import Repair

JSONLD_TYPE = "application/ld+json"


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.scripts: list[tuple[dict[str, str], str]] = []
        self.list_items: list[str] = []
        self._script: dict[str, str] | None = None
        self._script_text: list[str] = []
        self._items: list[list[str]] = []
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = {name: value or "" for name, value in attrs}
        self.tags.append((tag, attributes))
        if tag == "script":
            self._script, self._script_text = attributes, []
        elif tag == "li":
            self._items.append([])

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.scripts.append((self._script, "".join(self._script_text)))
            self._script = None
        elif tag == "li" and self._items:
            self.list_items.append(" ".join("".join(self._items.pop()).split()))

    def handle_data(self, data):
        if self._script is not None:
            self._script_text.append(data)
        for item in self._items:
            item.append(data)

    @property
    def canonical(self) -> str:
        return next((a.get("href", "") for t, a in self.tags if t == "link" and a.get("rel") == "canonical"), "")

    def script_with(self, **match: str) -> str | None:
        return next((text for attrs, text in self.scripts if all(attrs.get(k) == v for k, v in match.items())), None)


def load_lenient(text: str, repairs: list[Repair]) -> tuple[Any, list[str]] | None:
    try:
        return json.loads(text, strict=False), []
    except ValueError:
        pass
    applied = []
    for repair in repairs:
        repaired = re.sub(repair.pattern, repair.replacement, text)
        if repaired != text:
            text = repaired
            applied.append(repair.name)
    try:
        return json.loads(text, strict=False), applied
    except ValueError:
        return None


def jsonld_blocks(page: Page, repairs: list[Repair]) -> tuple[list[Any], list[str]]:
    blocks, applied = [], []
    for attrs, text in page.scripts:
        if attrs.get("type") != JSONLD_TYPE:
            continue
        loaded = load_lenient(text, repairs)
        if loaded is not None:
            blocks.append(loaded[0])
            applied.extend(loaded[1])
    return blocks, applied


def walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def has_type(node: dict[str, Any], name: str) -> bool:
    kind = node.get("@type")
    return name in kind if isinstance(kind, list) else kind == name
