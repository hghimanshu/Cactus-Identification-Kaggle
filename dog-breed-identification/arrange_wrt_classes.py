import os
import pandas as pd 

path = os.getcwd()
data = pd.read_csv(path + '/labels.csv')
total_images = data['id']

for i, j in zip(data['id'], data['breed']):
	try:
		if not os.path.exists(path + '/data/train/' + j):
			os.makedirs(path + '/data/train/' + j)
		image_path = path + '/train/' + i + '.jpg'
		comm = 'mv ' + image_path + ' ' + path + '/data/train/' + j + '/'
		os.system(comm)
	except Exception as e:
		print(str(e))
