from libclaim.pricing.sources.jsonld_book import JsonLdBookSource

IDENTIFIER_KEYS = ("isbn", "sku", "mpn")


class Bookchor(JsonLdBookSource):
    name = "Bookchor"

    @property
    def template(self) -> str:
        return self.config.sources.bookchor.url

    def identifiers(self, response, products):
        return [str(product[key]) for product in products for key in IDENTIFIER_KEYS if product.get(key)]
