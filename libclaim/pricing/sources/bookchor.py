from libclaim.pricing.sources.jsonld_book import JsonLdBookSource

IDENTIFIER_KEYS = ("isbn", "sku", "mpn")


def product_identifiers(product: dict) -> list[str]:
    return [str(product[key]) for key in IDENTIFIER_KEYS if product.get(key)]


class Bookchor(JsonLdBookSource):
    name = "Bookchor"

    @property
    def template(self) -> str:
        return self.config.sources.bookchor.url

    @property
    def currency(self) -> str:
        return self.config.sources.bookchor.currency

    def identifiers(self, response, products):
        return [value for product in products for value in product_identifiers(product)]

    def matching(self, response, products, isbn):
        return [product for product in products if isbn in product_identifiers(product)]
