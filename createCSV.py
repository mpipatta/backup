import pandas as pd
from datetime import datetime

now = datetime.now()
df = pd.DataFrame([[1,2,3,4,5]], [now.strftime("%Y-%m-%d %H:00")],['Lt', 'T+1','Lt+1MLR','Lt+1ANN','Lt+1LSTM'])
df.to_csv('output.csv')