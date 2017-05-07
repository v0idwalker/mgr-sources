import scrapy

class IndepedSpider(scrapy.Spider):
        name="independLinks"
        urls=[
            'http://www.independent.co.uk/news/world',
        ]

        def parse(self, response):
