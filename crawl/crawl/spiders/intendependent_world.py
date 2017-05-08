import scrapy

class IndepedSpider(scrapy.Spider):
        name="independLinks"
        urlBase="http://www.independent.co.uk"
        urls=[
            'http://www.independent.co.uk/news/world',
        ]

        def parse(self, response):
            return 0