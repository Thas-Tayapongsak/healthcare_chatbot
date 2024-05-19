from requests import Session
from string import ascii_lowercase


def urlrequest():
    """ Get the url to each article's pages that starts with the letter

    Returns
    -------
    list
        a list of string of url to each pages
    """

    letters = ascii_lowercase + '#'
    url = []
    session = Session()

    # get header
    session.head('https://my.clevelandclinic.org/health')

    # send post request for each letters
    for letter in letters:
        response = session.post(
            url='https://my.clevelandclinic.org/atoz/healthinformationpages',
            data={
                'letter': letter
            },
            headers={
                'Referer': 'https://my.clevelandclinic.org/health',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            }
        )

        # get only urls
        response_json = response.json()

        # append url to the list
        for title_url in response_json:
            url.append(title_url['url'])

    return url


# TRY IT OUT
#
#res = urlrequest()
#
# print(res[0])
# print(res[100])
# print(res[-1])
# print(len(res))

