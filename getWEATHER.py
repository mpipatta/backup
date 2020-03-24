import requests
from bs4 import BeautifulSoup

def loadtemp():

    url='https://weather.com/weather/hourbyhour/l/61d235a12c8f0b158c472bb5cf4a6a2d17b42270c214e7285c48666e57f21864'
    page = requests.get(url)
    soup=BeautifulSoup(page.content,"html.parser")
    table=soup.find_all("table",{"class":"twc-table"})
    temp = table[0].find_all("td",{"class":"temp"})[0].text
    temp = (float(temp.split('°')[0])-32)/9*5
    return (temp)