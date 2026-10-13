import json
SUP = {'zh','en','ja','yue'}
def clips():
    d = json.load(open('D:/mini_asr_data/manifest.json', encoding='utf-8'))
    c = [x for x in d['clips'] if x['lang'] in SUP]
    c.sort(key=lambda x: (x['lang'], x['id']))
    return c
