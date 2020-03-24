import urllib.request
import json
import datetime

def loaddata():
    #load = []
    url = "https://www.bems.chula.ac.th/web/cham5-api/api/v1/building/2/building_usage/day/peak"
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    hh = datetime.datetime.now().hour
    load_00 = data['graph'][(hh)*4-4]['y']
    load_15 = data['graph'][(hh)*4-3]['y']
    load_30 = data['graph'][(hh)*4-2]['y']
    load_45 = data['graph'][(hh)*4-1]['y']
 
    return ((load_00+load_15+load_30+load_45)/4)
    #return (hh, mm)