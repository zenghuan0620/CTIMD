from Configuration import Config

def rule1(arg):
    # executable file in system direction
    if arg.find('C:\Windows\system') != -1 or arg.find('C:\Windows\system32') != -1:
        return 1
    return 0

def rule2(arg):
    # two or more suffixes
    num=0
    for i in range(len(arg)):
        if arg[i]=='.':num+=1
    if num>=2:return 1
    return 0

def rule3(arg):
    # malware modifies the registry to modify system policies
    l=['HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft', '\Windows\CurrentVersion', 'Run', 'Policies', 'Explorer']
    for s in l:
        if arg.find(s) != -1:
            return 1
    return 0

def rulematch(arglist):
    config = Config()
    t = [0] * config.ruleNum
    for arg in arglist:
        arg = str(arg)
        if t[0]==0:
            t[0] = rule1(arg)
        if t[1]==0:
            t[1] = rule2(arg)
        if t[2]==0:
            t[2] = rule3(arg)
    return t