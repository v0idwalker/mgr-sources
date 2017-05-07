import scrapy

class GuardianSpider(scrapy.Spider):
    name = "guardiaLinks"
    urls = [
        'http://www.independent.co.uk/news/world',
    ]

    def parse(self, response):
