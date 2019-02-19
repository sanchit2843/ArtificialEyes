#flickr dataset
#import supportlib
!pip install -i https://test.pypi.org/simple/ supportlib
import supportlib.gettingdata as getdata

#Download data from kaggle
getdata.kaggle() # Upload kaggle.json on google colab
!kaggle datasets download -d hsankesara/flickr-image-dataset
getdata.zipextract('/content/flickr-image-dataset.zip')

#Caption csv

import pandas as pd
result = pd.read_csv('/content/flickr30k_images/results.csv',delimiter='|')

#converting result into numpy array
result = result.iloc[:,:].values


from tqdm import tqdm
result1 = list()
for i in tqdm(range(len(result))):
  try:
    if(int(result[i][1])== 0):
      result1.append(result[i])
  except:
    pass
result1 = np.asarray(result1)
result1 = np.delete(result1,1,1)
