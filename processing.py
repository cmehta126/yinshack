import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime

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

def add_lag_to_complaints(df_complaints, scale_days, base_year, base_month, base_day):
    p = timedelta(days=scale_days)
    base_time = datetime(base_year, base_month, base_day);
    incident_time = pd.DatetimeIndex(df_complaints['incident_date'])
    df_complaints['LAG'] = (base_time - incident_time)/p
    df_complaints['LAG'] = df_complaints.LAG.apply(np.floor)
    df_complaints['LAG'] = df_complaints.LAG.apply(int)
    df_complaints = df_complaints[df_complaints.LAG >= 0]
    return(df_complaints)
