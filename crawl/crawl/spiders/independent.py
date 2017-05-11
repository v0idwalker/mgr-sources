# -*- coding: utf-8 -*-
import scrapy
from crawl.items import theIndependentItem

class IndependentSpider(scrapy.Spider):
    name = "independent"
    basuURL = "http://www.independent.co.uk/"
    allowed_domains = ["http://www.independent.co.uk/"]
    start_urls = ['http://www.independent.co.uk/news/world/']

    def parse(self, response):
        link_data = response.xpath('//div[@class="content"]/h1/a')
        for l in link_data:
            data = theIndependentItem()
            data['url'] = l.xpath('@href').extract()
            data['name'] = l.xpath('text()').extract()
            data['tag'] = ''
            data['date'] = ''
            # data['']

            yield data