# fetch data
# for theGuardian: response.xpath('//a[contains(@data-link-name, "article")]').extract() response.css('li.l-row__item').extract()  E
# for Independent: response.css('.content h1 a')
import scrapy
from crawl.crawl import spiders

scrapy