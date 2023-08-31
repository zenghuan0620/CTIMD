import xml.etree.ElementTree as et
import os
import json
import pandas as pd
import json
from Configuration import Config
from ioc.getIoc import getAllIoc
import difflib
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from urllib.parse import urlparse
from rule.ruleset import rulematch

def calcSimilarity(s1,s2):
    return difflib.SequenceMatcher(None,str(s1),str(s2)).quick_ratio()

api2index={}

def getApi(path):
    tree = et.ElementTree(file=path)
    root = tree.getroot()
    apiList=[]
    api_list=root.find('file_list').find('file').find('start_boot').find('action_list').findall('action')
    reboot=root.find('file_list').find('file').find('reboot')
    if reboot!=None:
        api_list.extend(reboot.find('action_list').findall('action'))
    for api in api_list:
        d={}
        if api==None:
            continue
        if 'api_name' not in api.attrib:
            continue
        name=api.attrib['api_name']
        if name not in api2index:
            api2index[name]=len(api2index)+1
        apiArg_list=api.find('apiArg_list').findall('apiArg')
        l=[]
        if len(apiArg_list)>0:
            for arg in apiArg_list:
                l.append(arg.attrib['value'])
        exInfo_list=api.find('exInfo_list').findall('exInfo')
        if len(exInfo_list)>0:
            for arg in exInfo_list:
                l.append(arg.attrib['value'])
        d['name']=name
        d['apiList']=l
        apiList.append(d)
    return apiList

def xmlToJson():
    '''
    将所有的xml样本转换为json格式，每个json格式入下
    {
        sha256：[{‘name’：api名称，‘apiList’：[api参数值组成的列表]}...]
        ...
        sha256：[{‘name’：api名称，‘apiList’：[api参数值组成的列表]}...]
    }
    :return:
    '''
    path='../dataset/train'
    blackAndwhite = os.listdir(path)
    for baw in blackAndwhite:
        name = baw
        filePath = path + "/" + baw
        fileList = os.listdir(filePath)
        d={}
        for file in fileList:
            d[file]=getApi(filePath+"/"+file)
        if name=='black':
            savePath=r'../dataset/black.json'
        else:
            savePath=r'../dataset/white.json'
        with open(savePath, 'w', encoding='utf-16') as outfile:
            json.dump(d, outfile)

def saveDict():
    path=r'../dataset/api2index.json'
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(api2index, outfile)

def match(path,word_index,isBlack):
    fileNameRegularExpression = r'([\{A-Za-z0-9-_\*\\\\.}]+\.(lock|lib|doc|tlb|docx|ocx|url|log|msg|odt|pages|rtf|tex|txt|wpd|wps|csv|dat|ged|key|keychain|pps|ppt|pptx|sdf|tar|tax2016|tax2018|vcf|xml|aif|iff|m3u|m4a|mid|mp3|mpa|wav|wma|3g2|3gp|asf|avi|flv|m4v|mov|mp4|mpg|rm|srt|swf|vob|a|wmv|3dm|3ds|max|obj|bmp|dds|gif|heic|jpg|png|psd|pspimage|tga|thm|tif|tiff|yuv|ai|eps|svg|indd|pct|pdf|xlr|xls|xlsx|accdb|db|dbf|mdb|pdb|sql|apk|app|bat|cgi|exe|gadget|jar|wsf|b|dem|gam|nes|rom|sav|dwg|dxf|gpx|kml|kmz|asp|aspx|cer|cfm|csr|css|dcr|htm|html|js|jsp|rss|xhtml|hta|crx|plugin|fnt|fon|otf|ttf|cab|cpl|cur|deskthemepack|dll|dmp|drv|icns|ico|lnk|sys|cfg|ini|prf|hqx|mim|uue|7z|cbr|deb|gz|pkg|rar|rpm|sitx|tar|gz|zip|zipx|bin|cue|dmg|iso|mdf|toast|vcd|config|class|cpp|cs|dtd|fla|h|java|lua|m|pl|sh|sln|swift|vb|vcxproj|xcodeproj|bak|tmp|crdownload|ics|msi|part|torrent|adb|ads|ahk|applescript|as|au3|bas|cljs|cmd|coffee|c|cpp|ino|egg|egt|erb|hta|ibi|ici|ijs|ipynb|itcl|js|jsfl|lua|m|mrc|ncf|nuc|nud|nut|pde|php|pl|pm|ps1|ps1xml|psc1|psd1|psm1|py|pyc|pyo|r|rb|rdp|rs|sb2|scpt|scptd|sdl|sh|syjs|sypy|tcl|tns|vbs|xpl|ebuild|scr|pif|chm|service|net))'
    filenamematch = re.compile(fileNameRegularExpression)
    ss = "(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)\\.(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)\\.(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)\\.(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)"
    urlmatch = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    pathmatch = re.compile(
        r'^([a-zA-Z]:(\\[^\\/:*?<>|]+)*\\[^\\/:*?<>|]+\.[^\\/:*?<>|]+,)*[a-zA-Z]:(\\[^\\/:*?<>|]+)*\\[^\\/:*?<>|]+\.[^\\/:*?<>|]+$')
    pathIocList, ipIocList, urlIocList, filenameIocList, domainIocList = getAllIoc()
    jsonFile = open(path, encoding='utf-16')
    json_black = json.load(jsonFile)
    ans = {}
    config = Config()
    num = 0
    for sha, apiList in json_black.items():
        t = []
        num += 1
        if num % 1000 == 0: print(num)
        for api in apiList:
            k = []
            name = api['name']
            l = api['apiList']
            if name in word_index:
                k.append(word_index[name])
            else:
                k.append(0)
            flag, max_url, max_path, maxx1, maxx2 = 0, 0, 0, 0, 0
            flag_domain, max_filename, maxx4, maxx5 = 0, 0, 0, 0
            for arg in l:
                if len(str(arg)) <= 4: continue
                arg = str(arg)
                result = re.search(ss, arg)
                if result:
                    for ip in ipIocList:
                        if ip == result[0]: flag = 1
                if urlmatch.match(arg):
                    maxx1 = max([calcSimilarity(url, arg) for url in urlIocList])
                    domain = urlparse(arg).hostname
                    for d in domainIocList:
                        if d == domain: flag_domain = 1
                if pathmatch.match(arg):
                    maxx2 = max([calcSimilarity(path, arg) for path in pathIocList])
                    l = arg.split('\\')
                    file = l[-1]
                    maxx4 = max([calcSimilarity(file, filename) for filename in filenameIocList])
                if filenamematch.match(arg):
                    maxx5 = max([calcSimilarity(arg, filename) for filename in filenameIocList])
                max_url = max(maxx1, max_url)
                max_path = max(maxx2, max_path)
                max_filename = max(max_filename, maxx4, maxx5)
            k.append(flag)
            k.append(int(max_url * 100))
            k.append(int(max_path * 100))
            k.append(flag_domain)
            k.append(int(max_filename * 100))
            t.append(k)
            if len(t) >= config.seq_lenth: break
        if len(t) < config.seq_lenth:
            for i in range(config.seq_lenth - len(t)):
                t.append([0, 0, 0, 0, 0, 0])
        ans[sha] = t
    if isBlack:
        savePath = r'../dataset/black_IOC_match.json'
    else :
        savePath = r'../dataset/white_IOC_match.json'
    with open(savePath, 'w', encoding='utf-8') as outfile:
        json.dump(ans, outfile)

def getDic(urlIocList, pathIocList):
    dic_filename = {}
    dic_domain = {}
    for url in urlIocList:
        domain = urlparse(url).hostname
        if domain in dic_domain:
            dic_domain[domain].append(url)
        else:
            dic_domain[domain] = [url]
    for path in pathIocList:
        l = path.split('\\')
        filename = l[-1]
        if filename in dic_filename:
            dic_filename[filename].append(path)
        else:
            dic_filename[filename] = [path]
    return dic_domain, dic_filename

def match02(path,word_index,isBlack):
    fileNameRegularExpression = r'([\{A-Za-z0-9-_\*\\\\.}]+\.(lock|lib|doc|tlb|docx|ocx|url|log|msg|odt|pages|rtf|tex|txt|wpd|wps|csv|dat|ged|key|keychain|pps|ppt|pptx|sdf|tar|tax2016|tax2018|vcf|xml|aif|iff|m3u|m4a|mid|mp3|mpa|wav|wma|3g2|3gp|asf|avi|flv|m4v|mov|mp4|mpg|rm|srt|swf|vob|a|wmv|3dm|3ds|max|obj|bmp|dds|gif|heic|jpg|png|psd|pspimage|tga|thm|tif|tiff|yuv|ai|eps|svg|indd|pct|pdf|xlr|xls|xlsx|accdb|db|dbf|mdb|pdb|sql|apk|app|bat|cgi|exe|gadget|jar|wsf|b|dem|gam|nes|rom|sav|dwg|dxf|gpx|kml|kmz|asp|aspx|cer|cfm|csr|css|dcr|htm|html|js|jsp|rss|xhtml|hta|crx|plugin|fnt|fon|otf|ttf|cab|cpl|cur|deskthemepack|dll|dmp|drv|icns|ico|lnk|sys|cfg|ini|prf|hqx|mim|uue|7z|cbr|deb|gz|pkg|rar|rpm|sitx|tar|gz|zip|zipx|bin|cue|dmg|iso|mdf|toast|vcd|config|class|cpp|cs|dtd|fla|h|java|lua|m|pl|sh|sln|swift|vb|vcxproj|xcodeproj|bak|tmp|crdownload|ics|msi|part|torrent|adb|ads|ahk|applescript|as|au3|bas|cljs|cmd|coffee|c|cpp|ino|egg|egt|erb|hta|ibi|ici|ijs|ipynb|itcl|js|jsfl|lua|m|mrc|ncf|nuc|nud|nut|pde|php|pl|pm|ps1|ps1xml|psc1|psd1|psm1|py|pyc|pyo|r|rb|rdp|rs|sb2|scpt|scptd|sdl|sh|syjs|sypy|tcl|tns|vbs|xpl|ebuild|scr|pif|chm|service|net))'
    filenamematch = re.compile(fileNameRegularExpression)
    ss = "(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)\\.(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)\\.(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)\\.(25[0-5]|2[0-4]\\d|[0-1]\\d{2}|[1-9]?\\d)"
    urlmatch = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    pathmatch = re.compile(
        r'^([a-zA-Z]:(\\[^\\/:*?<>|]+)*\\[^\\/:*?<>|]+\.[^\\/:*?<>|]+,)*[a-zA-Z]:(\\[^\\/:*?<>|]+)*\\[^\\/:*?<>|]+\.[^\\/:*?<>|]+$')

    pathIocList, ipIocList, urlIocList, filenameIocList, domainIocList = getAllIoc()
    dic_domain, dic_filename=getDic(urlIocList, pathIocList)

    jsonFile = open(path, encoding='utf-16')
    json_black = json.load(jsonFile)
    ans = {}
    config = Config()
    num = 0
    for sha, apiList in json_black.items():
        t = []
        num += 1
        if num % 1000 == 0:
            print(num)
        for api in apiList:
            k = []
            name = api['name']
            l = api['apiList']
            if name in word_index:
                k.append(word_index[name])
            else:
                k.append(0)
            flag, max_url, max_path, maxx1, maxx2 = 0, 0, 0, 0, 0
            flag_domain, max_filename, maxx4, maxx5 = 0, 0, 0, 0
            for arg in l:
                if len(str(arg)) <= 4: continue
                arg = str(arg)
                result = re.search(ss, arg)
                if result:
                    for ip in ipIocList:
                        if ip == result[0]: flag = 1
                if urlmatch.match(arg):
                    domain = urlparse(arg).hostname
                    maxx1 = max([calcSimilarity(url, arg) for url in dic_domain[domain]])
                    for d in domainIocList:
                        if d == domain:
                            flag_domain = 1
                if pathmatch.match(arg):
                    l = arg.split('\\')
                    file = l[-1]
                    maxx2 = max([calcSimilarity(path, arg) for path in dic_filename[file]])
                    maxx4 = max([calcSimilarity(file, filename) for filename in filenameIocList])
                if filenamematch.match(arg):
                    maxx5 = max([calcSimilarity(arg, filename) for filename in filenameIocList])
                max_url = max(maxx1, max_url)
                max_path = max(maxx2, max_path)
                max_filename = max(max_filename, maxx4, maxx5)
            k.append(flag)
            k.append(int(max_url * 100))
            k.append(int(max_path * 100))
            k.append(flag_domain)
            k.append(int(max_filename * 100))
            t.append(k)
            if len(t) >= config.seq_lenth: break
        if len(t) < config.seq_lenth:
            for i in range(config.seq_lenth - len(t)):
                t.append([0, 0, 0, 0, 0, 0])
        ans[sha] = t
    if isBlack:
        savePath = r'../dataset/black_IOC_match.json'
    else :
        savePath = r'../dataset/white_IOC_match.json'
    with open(savePath, 'w', encoding='utf-8') as outfile:
        json.dump(ans, outfile)

def ruleMatch(path, isBlack):
    config = Config()
    ruleNum=config.ruleNum

    jsonFile = open(path, encoding='utf-16')
    json_file = json.load(jsonFile)
    d={}
    for sha, apiList in json_file.items():
        ans=[]
        for api in apiList:
            l = api['apiList']
            ans.append(rulematch(l))
            if len(ans)==1200:break
        if len(ans)<1200:
            while len(ans)<1200:
                ans.append([0]*ruleNum)
        d[sha]=ans

    if isBlack:
        saveParh = r'../dataset/black_rule_match.json'
    else:
        saveParh = r'../dataset/white_rule_match.json'
    with open(saveParh, 'w', encoding='utf-8') as outfile:
        json.dump(d, outfile)

if __name__ == '__main__':
    xmlToJson()
    saveDict()
    print("Extract api sequence done")

    jsonFile = open('data/api2index_xml.json')
    word_index = json.load(jsonFile)
    match(r'../dataset/black.json', word_index, 1)
    match(r'../dataset/white.json', word_index, 0)
    # match02(r'../dataset/black.json', word_index, 1)
    # match02(r'../dataset/white.json', word_index, 0)
    print("IOC match done")

    ruleMatch(r'../dataset/black.json', 1)
    ruleMatch(r'../dataset/white.json', 0)
    print("rule match done")
