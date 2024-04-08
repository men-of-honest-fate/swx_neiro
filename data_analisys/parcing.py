from bs4 import BeautifulSoup
import requests

site = requests.get("https://swx.sinp.msu.ru/apps/sep_events_cat/").content.decode("utf-8")
soup = BeautifulSoup(site, "html.parser")
print(soup.prettify())
# for item in soup.find_all("table"):
#     print(item)
