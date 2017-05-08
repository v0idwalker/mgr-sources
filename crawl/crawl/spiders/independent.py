# -*- coding: utf-8 -*-
import scrapy


class IndependentSpider(scrapy.Spider):
    name = "independent"
    allowed_domains = ["http://www.independent.co.uk/news/world"]
    start_urls = ['http://http://www.independent.co.uk/news/world/']

    def parse(self, response):
        pass
