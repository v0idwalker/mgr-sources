# -*- coding: utf-8 -*-
import scrapy


class IndependentSpider(scrapy.Spider):
    name = "independent"
    basuURL = "http://www.independent.co.uk/"
    allowed_domains = ["http://www.independent.co.uk/"]
    start_urls = ['http://www.independent.co.uk/news/world/']

    def parse(self, response):
        items = response.css('.content h1 a')
        for i in items:
            print(u + i.extract())
