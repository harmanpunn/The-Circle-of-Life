import sys

def str2bool(v):
  
    if v.lower() == "true":
        return True
    elif v.lower()=="false":
        return False
    else:
        raise RuntimeError("Invalid value")

def strToAgent(x):
    if x=="x":
        return 10
    elif x=="q":
        return 6
    else:
        p = int(x)
        if p>=1 and p<=9:
            return p
        else:
            raise ValueError()        
        
allowed_args = {
    "ui":str2bool,
    "node_count":int,
    "mode":int,
    "agent":strToAgent,
    "noisy":str2bool,
    "quiet":str2bool,
    "noisy_agent":str2bool,
    "graphs":int,
    "games":int,
    "p3":str2bool
}

def processArgs():
    if len(sys.argv)>1:
        
        argv = sys.argv[1:]
        args = {}
        for x in argv:
            if x[0:2]!="--":
                raise RuntimeError("Args start with '--'")
            x = x.split('--')[1]
            
            if len(x.split("="))!=2:
                raise RuntimeError("Arg needs single '='")
            x = x.split("=")
            if not x[0] in allowed_args.keys():
                raise RuntimeError("Invalid Argument")

            try:
                args[x[0]]= allowed_args[x[0]](x[1])
            except:
                raise ValueError("Invalid value '"+x[1]+"' for "+x[0])
        
        return args    
    return {}