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

def add_complaint_severity(df_complaints):
    # Identifying worst complaints.
    df_complaints['complaintcategory_short_name'] = df_complaints['complaintcategory'].str[0:3]
    # Identifying worst complaints.
    worst_complaints = ["01A","01B","01C","03E","04H","05A","05B","05C","05D","05E","05F","05G","05H","05J","05K","05L","05M","05N","05P","05Q","05T"]
    medium_complaints = ["04A","04B","04C","04D","04E","04F","04G","04J"]
    df_complaints['complaint_type'] = df_complaints.complaintcategory_short_name.apply(lambda x: 2 if x in worst_complaints else 1 if x in medium_complaints else 0)
    df_complaints.drop('complaintcategory_short_name', axis=1, inplace=True)
    return(df_complaints)


def add_complaints_by_year_v2(base_year, nlags, df_complaints, df_officers, prefix):
    Y = df_officers
    for year in range(base_year,base_year-nlags,-1):
        q = count_complaints_in_year(year, df_complaints).to_frame()
        j = year-base_year
        jstr = prefix + '_Lag%d' % j
        q.columns = [jstr]
        Y = Y.merge(q, left_on = 'officer_id', right_index = 1, how = 'outer')
    Y.fillna(value = 0, inplace = True)
    return Y

def add_complaints_by_year_and_severity(base_year, nlags, df_complaints, df_officers):
    Y = df_officers
    type_labels = ['LowSeverity', 'MedSeverity','HighSeverity']
    for severity in range(0,3,1):
            X = df_complaints.loc[df_complaints['complaint_type'] == severity]
            Y = add_complaints_by_year_v2(base_year, nlags, X, Y, type_labels[severity]);
    return Y

