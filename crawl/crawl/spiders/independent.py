# -*- coding: utf-8 -*-
import scrapy
from crawl.items import  theIndependentItem

class IndependentSpider(scrapy.Spider):
    name = "www.independent.co.uk"
    baseURL = "https://www.independent.co.uk"
    allowed_domains = ["https://www.independent.co.uk"]
    start_urls = ['https://www.independent.co.uk/']

    def parse(self, response):
        link_data = response.xpath('//div[@class="content"]/h1/a')
        for l in link_data:
            data = theIndependentItem()
            data['url'] = self.baseURL.join(l.xpath('@href').extract())
            data['name'] = l.xpath('text()').extract()
            # data['tag'] = ''
            # data['date'] = ''
            # data['']
            yield data