import scrapy

class GuardianSpider(scrapy.Spider):
    name = 'guardiaLinks'
    urlBase ='https://www.theguardian.com'
    urls = [
        'http://www.independent.co.uk/news/world',
    ]

    def parse(self, response):
        return 0