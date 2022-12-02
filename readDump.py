import pickle
file= open('datadump/data-1', 'rb')
data = pickle.load(file)

file.close()

print(data)