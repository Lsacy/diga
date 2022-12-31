from requests_html import HTMLSession



session = HTMLSession()
page = 'https://diga.bfarm.de/de/verzeichnis?type=%5B%5D'
r = session.get(page)
r.html.render(sleep=10)

print(r.html.html)

