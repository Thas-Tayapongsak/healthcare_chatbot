import scrapy
from urlrequest.urlrequest import urlrequest

class QnaspiderSpider(scrapy.Spider):
    name = "qnaspider"
    allowed_domains = ["my.clevelandclinic.org"]
    start_urls = ["https://my.clevelandclinic.org/health"]

    def parse(self, response):

        # get all the urls
        urls = urlrequest()

        # go to the each articles
        for url in urls:
            yield response.follow(url, callback = self.parse_article)

    def parse_article(self, response):

        # get all sections in the article
        sections  = response.css('div[data-identity="rich-text"]')

        # loop through all sections
        for section in sections:

            # get all headlines
            headlines = section.css('[data-identity="headline"]::text')
            
            # loop through all headlines
            for headline in headlines:

                # get the first paragraph after
                paragraph = section.xpath('*[contains(text(), "' + headline.get() + '")]/following-sibling::p[1]')
                
                # extract all text in the paragraph
                txt_fragments = paragraph.css('*::text')

                # answer string
                answer = ""

                # join text
                for fragment in txt_fragments:
                    answer += fragment.get()

                yield {
                    'question' : headline.get(),
                    'answer' : answer
                }
