# -*- coding: utf-8 -*-
import scrapy
from crawl.items import theGuardianItem


class TheguardianSpider(scrapy.Spider):
    name = "theguardian"
    baseURL = "https://www.theguardian.com"
    allowed_domains = ["https://www.theguardian.com"]
    start_urls = ['https://www.theguardian.com/world/']

    def parse(self, response):
        link_data = response.xpath('//a[contains(@data-link-name, "article")]')

        for d in link_data:
            data = theGuardianItem()
            data['url'] = d.xpath('@href').extract()
            data['name'] = d.xpath('text()').extract()
            # data['tag'] = ''
            # data['date'] = ''
            if data['name'] == [' ', ' ']:
                pass
            else:
                yield data
