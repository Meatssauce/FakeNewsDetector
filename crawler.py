from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin


def crawl(base_url, homepage='index.html', page_limit=1000):
    """
    Crawls the web until a page limit is reached and returns the result as a list.
    """

    results = []
    visited = {}
    seed_url = base_url + homepage
    to_visit = [seed_url]
    n_pages_visited = 0

    # Find all outbound links on next page and explore each one
    while to_visit:
        # impose a limit to avoid crawler traps
        if n_pages_visited == page_limit:
            break

        # consume a link from the list of urls to visit
        url = to_visit.pop(0)

        # get the webpage, p(url), for the consumed url
        page = requests.get(url)
        # scrape p(url), save html code to results
        soup = BeautifulSoup(page.text, 'html.parser')
        if n_pages_visited > 0:
            results.append({'url': url, 'soup': soup})

        # add url to visted, add newly discovered urls to to_visit
        visited[url] = True
        hyperlinks = soup.findAll('a')
        for hyperlink in hyperlinks:
            new_url = urljoin(url, hyperlink['href'])
            if new_url not in visited and new_url not in to_visit:
                to_visit.append(new_url)

        n_pages_visited += 1

    print('\nvisited {0:5d} pages; {1:5d} pages in to_visit'.format(len(visited), len(to_visit)))

    return results


if __name__ == "__main__":
    crawl('https://www.facebook.com', homepage='', page_limit=10)
