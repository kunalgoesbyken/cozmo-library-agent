from libclaim.pricing.sources.jsonld_book import JsonLdBookSource


class Bookswagon(JsonLdBookSource):
    name = "Bookswagon"

    @property
    def template(self) -> str:
        return self.config.sources.bookswagon.url

    @property
    def currency(self) -> str:
        return self.config.sources.bookswagon.currency

    @property
    def enabled(self) -> bool:
        return self.config.bookswagon.enabled

    def identifiers(self, response, products):
        label = self.config.sources.bookswagon.isbn_label
        return [item.removeprefix(label).strip() for item in response.page.list_items if item.startswith(label)]

    def matching(self, response, products, isbn):
        return products[:1] if isbn in self.identifiers(response, products) else []
