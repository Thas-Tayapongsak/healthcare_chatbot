# healthcare_chatbot
Create a question answering system using transformers on Health Library Dataset. This GitHub repository is a part of CSS432 Natural Language Processing and Information Retrieval course (2/2023).

## Members
* Natsongwat Yorsaengrat
* Pasin Suksang
* Phanupoom Boomprapasan
* Thas Tayapongsak
* Woraseth Limwanich

## Task
Students will train a transformer-based model to accurately answer questions based on the provided health-related dataset. The chatbot should be capable of providing relevant and informative responses to users' queries about various health topics. Health-related knowledges can be scraped from Cleveland Clinic Health Library.

## Steps
To set up, install all requirements in the terminal
```
pip install -r requirements.txt
```

The questions and answers are already in qna.json, but to run the webscraper.
```
scrapy crawl qnaspider -O qna.json
```

