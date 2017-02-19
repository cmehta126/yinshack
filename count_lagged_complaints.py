import pandas as pd; import numpy as np;
def count_complaints_in_year(year, x):
    x['year'] = pd.DatetimeIndex(x['incident_date']).year 
    x1 = x[(x.year == year)]
    x2 = x1.groupby(['officer_id']).size()
    return x2

def add_complaints_by_year(base_year, nlags, df_complaints, df_officers):
    Y = df_officers
    for year in range(base_year,base_year-nlags,-1):
        q = count_complaints_in_year(year, df_complaints).to_frame()
        j = year-base_year
        jstr = 'Lag%d' % j
        q.columns = [jstr]
        Y = Y.merge(q, left_on = 'officer_id', right_index = 1, how = 'outer')
    Y.fillna(value = 0, inplace = True)
    return Y
        
# Load Data
complaints = pd.read_csv('data/toy.complaint_data.csv'); 
officers = pd.read_csv('data/toy.officer_data.csv'); 
r = add_complaints_by_year(2015, 5, complaints, officers)


