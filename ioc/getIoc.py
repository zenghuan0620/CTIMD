
def getIoc(path):
    l = []
    with open(path,'r',encoding='utf-8') as f:
        txt = f.readlines()
        for i in txt:
            l.append(i[0:len(i) - 2])
    return l

def getAllIoc():
    pathIocList=getIoc('path.txt')
    ipIocList=getIoc('ip.txt')
    urlIocList=getIoc('./url.txt')
    filenameIocList=getIoc('filename.txt')
    domainIocList=getIoc('domain.txt')
    # print(len(pathIocList))
    # print(len(ipIocList))
    # print(len(urlIocList))
    return pathIocList,ipIocList,urlIocList,filenameIocList,domainIocList
