# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class theBaseItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = scrapy.Field()
    url = scrapy.Field()
    # date = scrapy.Field()
    # tag = scrapy.Field()

class theGuardianItem(theBaseItem):
    # define the fields for your item here like:
    # date = scrapy.Field()
    # tag = scrapy.Field()
    pass

class theIndependentItem(theBaseItem):
    # addtional = scrapy.Field()
    pass