import pickle

def readDumpFile(i):
    file= open('datadump/data-'+str(i), 'rb')
    data = pickle.load(file)
    file.close()
    return data 