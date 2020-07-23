#!/usr/bin/env python
# coding: utf-8

# ### GOAL
# To build statistical model for estimating EL(Expected Loss)
# 
# EL = PD * EAD * LGD

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Import Data

# In[3]:


loan_data_backup = pd.read_csv("loan_data_2007_2014.csv")


# In[6]:


loan_data = loan_data_backup.copy()


# ### Data Exploration

# In[7]:


loan_data.head(10)


# In[8]:


pd.options.display.max_columns = 10


# #### Displaying all columns in the data

# In[10]:


loan_data.columns.values


# #### Column Data Type

# In[11]:


loan_data.info()


# ### Preprocessing for Continous Variables

# In[9]:


# Converting emp_legnth columns and term to numeric value


# In[12]:


loan_data.emp_length.unique()


# In[13]:


loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',str(0))


# In[14]:


# To numeric
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])


# In[15]:


type(loan_data['emp_length_int'][0])


# In[16]:


loan_data.term.unique()


# In[17]:


loan_data['term_int'] = loan_data['term'].str.replace(' 36 months',str(36))
loan_data['term_int'] = loan_data['term_int'].str.replace(' 60 months',str(60))


# In[18]:


loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])


# In[19]:


type(loan_data['term_int'][0])


# In[20]:


loan_data.head(10)


# In[21]:


loan_data['earliest_cr_line'].head(12)


# In[28]:


# Convert to date time column from object(text string)
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'],format = '%b-%y')


# In[29]:


type(loan_data['earliest_cr_line_date'][0])


# In[30]:


# checking how many days before earliest loan was given(toadays date is taken as reference)
df = pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']


# In[31]:


# Converting days into months (using timedelta)
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric(df/np.timedelta64(1,'M')))


# In[32]:


loan_data['mths_since_earliest_cr_line'].describe()


# In[33]:


loan_data.loc[:,['earliest_cr_line','earliest_cr_line_date','mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line']<0] 


# In[34]:


# The different between future time period and past time period shouldn't come negative
# This is because while converting to datetime, the dataframe has taken many columns of date 2060 instead of 1960


# In[35]:


# Now to convert from 20 to 19 in each row in earliest_cr_line_date is not an easy task(Origin of built in time scale starts from 1960)
# Instead we are directly imputing


# In[36]:


loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] <0] = loan_data['mths_since_earliest_cr_line'].max()


# In[37]:


min(loan_data['mths_since_earliest_cr_line'])


# In[38]:


# Homework do the same as above for 'term' and 'issue_date' variable


# In[39]:


loan_data['term_int'].describe()


# In[40]:


loan_data['issue_d']


# In[41]:


loan_data['issue_d_dateTime'] = pd.to_datetime(loan_data['issue_d'],format = '%b-%y')
type(loan_data['issue_d_dateTime'][0])
df1 = pd.to_datetime('2017-12-01') - loan_data['issue_d_dateTime']
loan_data['mths_since_issue'] = round(pd.to_numeric(df1/np.timedelta64(1,'M')))


# In[42]:


loan_data['mths_since_issue'].describe()


# In[43]:


loan_data.loc[:,['issue_d','issue_d_dateTime','mths_since_issue']]


# ### Discrete / categorical preprocessing

# In[44]:


loan_data.head(5)


# ### Create dummy variable for discrete variables
# ### create a new data frame for dummy variables than concat in loan_data 

# In[46]:


pd.get_dummies(loan_data['grade'],prefix='grade',prefix_sep=":")


# In[47]:


loan_data.columns


# In[48]:


loan_data_dummies = [pd.get_dummies(loan_data['grade'],prefix='grade',prefix_sep=':'),
                    pd.get_dummies(loan_data['sub_grade'],prefix='sub_grade',prefix_sep=':'),
                    pd.get_dummies(loan_data['home_ownership'],prefix='home_ownership',prefix_sep=':'),
                    pd.get_dummies(loan_data['verification_status'],prefix='verification_status',prefix_sep=':'),
                    pd.get_dummies(loan_data['loan_status'],prefix='loan_status',prefix_sep=':'),
                    pd.get_dummies(loan_data['purpose'],prefix='purpose',prefix_sep=':'),
                    pd.get_dummies(loan_data['addr_state'],prefix='addr_state',prefix_sep=':'),
                    pd.get_dummies(loan_data['initial_list_status'],prefix='initial_list_status',prefix_sep=':')]


# In[49]:


type(loan_data_dummies)


# In[50]:


loan_data_dummies = pd.concat(loan_data_dummies,axis=1)


# In[51]:


type(loan_data_dummies)


# In[52]:


loan_data_dummies.head(10)


# In[53]:


loan_data.head(10)


# In[54]:


loan_data = pd.concat([loan_data,loan_data_dummies],axis=1)


# In[55]:


loan_data.columns.values


# ### Dealing with missing values

# In[56]:


loan_data.isna().sum()


# In[57]:


# pd.options.display.max_rows = 100
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'],inplace=True)


# In[58]:


loan_data['total_rev_hi_lim'].isna().sum()


# In[59]:


loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(),inplace=True)


# In[60]:


loan_data['annual_inc'].isna().sum()


# In[61]:


loan_data['mths_since_earliest_cr_line'].fillna(0,inplace=True)


# In[62]:


loan_data['mths_since_earliest_cr_line'].isna().sum()


# In[63]:


loan_data['acc_now_delinq'].fillna(0,inplace=True)
loan_data['total_acc'].fillna(0,inplace=True)
loan_data['pub_rec'].fillna(0,inplace=True)
loan_data['open_acc'].fillna(0,inplace=True)
loan_data['inq_last_6mths'].fillna(0,inplace=True)
loan_data['delinq_2yrs'].fillna(0,inplace=True)
loan_data['emp_length_int'].fillna(0,inplace=True)


# In[64]:


#pd.options.display.max_rows=None
loan_data.isna().sum()


# In[65]:


loan_data.head(10)


# # PD Model

# ### Data Preparation

# ##### Dependent Variables

# In[66]:


loan_data['loan_status'].unique()


# In[67]:


loan_data['loan_status'].value_counts()


# In[68]:


loan_data['good/bad'] = np.where(loan_data['loan_status'].isin(['Charged Off','Default','Late (31-120 days)',
                                                                'Late (16-30 days)',
                                                                'Does not meet the credit policy. Status:Charged Off']),0,1)


# In[69]:


loan_data['good/bad'].head()


# ### Independent Variables

# At the end scorecord should contain whether a guy should get a loan or not i.e.  1 or 0
# Discrete independent variables such as home ownership , age etc can be converted directly into dummy variables
# However categorizing continous variables is not easy, first fine classing is done which is initial binning of data into 
# between 20 and 50 fine granular bins 
# Coarse classing is where a binning process is applied to the fine granular bins to merge those with similar risk and 
# create fewer bins, usually up to ten. The purpose is to achieve simplicity by creating fewer bins, each with distinctively 
# different risk factors, while minimizing information loss. However, to create a robust model that is resilient to overfitting
# , each bin should contain a sufficient number of observations from the total account (5% is the minimum recommended by most
# practitioners)
# 
# From initial fine classing, coarse classing is done based on the weight of evidence 

# ###### Splitting the data

# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


train_test_split(loan_data.drop('good/bad',axis=1),loan_data['good/bad'])


# In[72]:


loan_data_inputs_train,loan_data_inputs_test,loan_data_outputs_train,loan_data_outputs_test =  train_test_split(loan_data.drop('good/bad',axis=1),loan_data['good/bad'],test_size=0.2,random_state=42)


# In[73]:


loan_data_inputs_train.shape


# In[74]:


loan_data_inputs_test.shape


# In[75]:


loan_data_outputs_train.shape


# In[76]:


loan_data_outputs_test.shape


# In[77]:


df_input_prep = loan_data_inputs_train
df_output_prep = loan_data_outputs_train


# In[78]:


df_input_prep.head(10)


# In[79]:


##### Dicrete Data Preprocessing
##### Dicrete variable is already categorical so here we have no need to calculate dummy variables using fine and coarse classing
##### Only calculate WOE and Information value to estimate if the variable can be included for predicting dependent variable


# In[80]:


df_input_prep['grade'].unique()


# In[81]:


df1 = pd.concat([df_input_prep['grade'],df_output_prep],axis=1)


# In[82]:


df1.head(10)


# In[83]:


df1.tail(10)


# ## Weight of evidence of discrete variable Grade

# ![WOE.PNG](attachment:WOE.PNG)

# In[84]:


df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count()


# In[85]:


df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()


# In[86]:


df1 = pd.concat([df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count(),
                df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()],axis=1)


# In[87]:


df1.head(10)


# In[88]:


df1 = df1.iloc[:,[0,1,3]]


# In[89]:


df1.head(5)


# In[90]:


df1.columns = [df1.columns.values[0],'n_obs','prop_good']


# In[91]:


df1.head(5)


# In[92]:


df1['no_good'] = df1['prop_good'] * df1['n_obs']
df1['no_bad'] = (1- df1['prop_good']) * df1['n_obs']


# In[93]:


df1.head(5)


# In[94]:


df1['Final_good'] = df1['no_good']/df1['no_good'].sum()
df1['Final_bad'] = df1['no_bad']/df1['no_bad'].sum()


# In[95]:


df1.head(5)


# In[96]:


df1['WOE'] = np.log(df1['Final_good']/df1['Final_bad'])


# In[97]:


df1.head(5)


# In[98]:


df1 = df1.sort_values(['WOE'])


# In[99]:


df1


# In[100]:


df1.reset_index(drop=True)


# In[101]:


df1['IV'] = (df1['Final_good']-df1['Final_bad']) * df1['WOE']


# In[102]:


df1['IV'] = df1['IV'].sum()


# In[103]:


df1


# In[104]:


### Grade Information value is 0.29 which comes under the bracket of 0.1-0.3
### It means medium predictive power to obtain output variable


# #### Function to calculate WOE

# In[105]:


def woe_discrete(df,independent_variable,dependent_variable):
    df = pd.concat([df[independent_variable],dependent_variable],axis=1)
    df = pd.concat([df.groupby(df.columns.values[0],as_index=False)[df.columns.values[1]].count(),
    df.groupby(df.columns.values[0],as_index=False)[df.columns.values[1]].mean()],axis=1)
    df = df.iloc[:,[0,1,3]]
    df.columns = [df.columns.values[0],'n_obs','prop_good']
    df['no_good'] = df['prop_good'] * df['n_obs']
    df['no_bad'] = (1- df['prop_good']) * df['n_obs']
    df['Final_good'] = df['no_good']/df['no_good'].sum()
    df['Final_bad'] = df['no_bad']/df['no_bad'].sum()
    df['WOE'] = np.log(df['Final_good']/df['Final_bad'])
    df = df.sort_values(['WOE'])
    df = df.reset_index(drop=True)
    df['IV'] = (df['Final_good']-df['Final_bad']) * df['WOE']
    df['IV'] = df['IV'].sum()
    return df


# In[106]:


df_temp=woe_discrete(df_input_prep,'grade',df_output_prep)


# In[107]:


df_temp


# #### Visualizing WOE for dicerete variables to interpret it

# In[108]:


sns.set()


# In[109]:


def plot_by_woe(df_woe,rotation_of_x_labels=0):
    x = np.array(df_woe.iloc[:,0].apply(str)) ## matplotlib works better with array than dataframes
    y = df_woe['WOE']
    plt.figure(figsize=(18,6))
    plt.plot(x,y,marker='o',linestyle='--',color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of evidence')
    plt.title(str('Weight of evidence by' + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_x_labels)


# In[110]:


plot_by_woe(df_temp)


# In[111]:


### Keeping dummy variable G (grade) as reference
### All other in regression model


# ##### Home Ownership Variable

# In[112]:


df_input_prep.head()


# In[113]:


df_home_owner=woe_discrete(df_input_prep,'home_ownership',df_output_prep)


# In[114]:


df_home_owner.head()


# In[115]:


df_home_owner.tail()


# In[116]:


plot_by_woe(df_home_owner)


# In[117]:


df_home_owner


# In 2nd column(n_obs) it is clearly visible that OTHER, NONE and ANY has few values in the dataset, therefore it is less
# WOE to predict loan default, but it is not good to delete those variables as those are most riskiest values, better if we combine them to get good amount of information
# 
# For RENT also, WOE is very low so we can combine it with OTHER,NONE and ANY

# In[119]:


df_input_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_input_prep['home_ownership:OTHER'] ,df_input_prep['home_ownership:RENT'],
                                                          df_input_prep['home_ownership:NONE'],df_input_prep['home_ownership:ANY']])


# From a set of categorical variables that represent one original independent variable, we make a reference category the 
# category with lowest WOE value

# #### address state discrete variable

# In[120]:


df_input_prep['addr_state'].unique()


# In[121]:


df_addr_state=woe_discrete(df_input_prep,'addr_state',df_output_prep)


# In[122]:


df_addr_state.head()


# In[123]:


plot_by_woe(df_addr_state)


# In[124]:


if ['addr_state:ND'] in df_input_prep.columns.values:
    pass
else:
    df_input_prep['addr_state:ND'] = 0


# In[125]:


plot_by_woe(df_addr_state.iloc[2:-2,:])


# Earlier first two and last two states were making us believe that all states from NV to DC wee kind of similar but ideally 
# it is not

# Combine NE, IA, NV, FL, Al, HI based on WOE and number of observation, all of these are having worst borrowers , WOE is lowest
# Being conservative, add ND(North Dakota earlier not in the list) also in this category

# Last four WV,NH,WY,DC and ME,ID are having good borrowers -combine them

# In[127]:


plot_by_woe(df_addr_state.iloc[6:-6,:])


# VA,NM,NY,TN,MO,LA,OK,NC,MD,CA have similar WOE
# However NY and CA have many borrowers so they will be a seperate dummy variable

# Final categories from VA to CA will be;
# 1. VA,NM
# 2. NY
# 3. TN,MO,LA,OK,NC,MA
# 4. CA

# In[128]:


### THEN UT,NJ,AZ,KY


# #### ![ADDR_STATE_DUMMYvARIABLES.PNG](attachment:ADDR_STATE_DUMMYvARIABLES.PNG)

# In[129]:


# We create the following categories:
# 'ND' 'NE' 'IA' NV' 'FL' 'HI' 'AL'
# 'NM' 'VA'
# 'NY'
# 'OK' 'TN' 'MO' 'LA' 'MD' 'NC'
# 'CA'
# 'UT' 'KY' 'AZ' 'NJ'
# 'AR' 'MI' 'PA' 'OH' 'MN'
# 'RI' 'MA' 'DE' 'SD' 'IN'
# 'GA' 'WA' 'OR'
# 'WI' 'MT'
# 'TX'
# 'IL' 'CT'
# 'KS' 'SC' 'CO' 'VT' 'AK' 'MS'
# 'WV' 'NH' 'WY' 'DC' 'ME' 'ID'

# 'IA_NV_HI_ID_AL_FL' will be the reference category.
df_inputs_prepr = df_input_prep.copy()

df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'],
                                              df_inputs_prepr['addr_state:IA'], df_inputs_prepr['addr_state:NV'],
                                              df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'],
                                                          df_inputs_prepr['addr_state:AL']])

df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])

df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                              df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'],
                                              df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])

df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'],
                                              df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])

df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                              df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'],
                                              df_inputs_prepr['addr_state:MN']])

df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                              df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'],
                                              df_inputs_prepr['addr_state:IN']])

df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'],
                                              df_inputs_prepr['addr_state:OR']])

df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])

df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])

df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                              df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'],
                                              df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])

df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                              df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'],
                                              df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])


# In[130]:


df_inputs_prepr.head()


# #### verification status discrete variable

# In[131]:


df_inputs_prepr['verification_status'].unique()


# In[132]:


df_verification_status=woe_discrete(df_input_prep,'verification_status',df_output_prep)


# In[133]:


df_verification_status.head()


# In[134]:


plot_by_woe(df_verification_status)


# #### purpose discrete variable

# In[135]:


df_inputs_prepr['purpose'].unique()


# In[136]:


df_purpose=woe_discrete(df_input_prep,'purpose',df_output_prep)


# In[137]:


df_purpose.head()


# In[138]:


plot_by_woe(df_purpose)


# In[189]:


# We combine 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house' in one category: 'educ__sm_b__wedd__ren_en__mov__house'.
# We combine 'other', 'medical', 'vacation' in one category: 'oth__med__vacation'.
# We combine 'major_purchase', 'car', 'home_improvement' in one category: 'major_purch__car__home_impr'.
# We leave 'debt_consolidtion' in a separate category.
# We leave 'credit_card' in a separate category.
# 'educ__sm_b__wedd__ren_en__mov__house' will be the reference category.
df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                 df_inputs_prepr['purpose:wedding'], df_inputs_prepr['purpose:renewable_energy'],
                                                                 df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                             df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                        df_inputs_prepr['purpose:home_improvement']])


# In[190]:


# 'initial_list_status'
df_initial_list_status = woe_discrete(df_inputs_prepr, 'initial_list_status', df_output_prep)
df_initial_list_status


# In[191]:


plot_by_woe(df_initial_list_status)
# We plot the weight of evidence values.


# ### Preprocessing Continuous Variables: Automating Calculations and Visualizing Results

# When we calculate and plot the weights of evidence of continuous variables categories, what do we sort them by their own
# values in ascending order

# In[ ]:





# In[193]:


# WoE function for ordered discrete and continuous variables
def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
# Here we define a function similar to the one above, ...
# ... with one slight difference: we order the results by the values of a different column.
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.


# In[206]:


def plot_by_woe(df_woe,rotation_of_x_labels=0):
    x = np.array(df_woe.iloc[:,0].apply(str)) ## matplotlib works better with array than dataframes
    y = df_woe['WoE']
    plt.figure(figsize=(18,6))
    plt.plot(x,y,marker='o',linestyle='--',color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of evidence')
    plt.title(str('Weight of evidence by' + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_x_labels)


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 1

# In[207]:


# term
df_inputs_prepr['term_int'].unique()
# There are only two unique values, 36 and 60.


# In[208]:


df_term_int = woe_ordered_continuous(df_inputs_prepr, 'term_int', df_output_prep)
# We calculate weight of evidence.
df_term_int


# In[209]:


plot_by_woe(df_term_int)
# We plot the weight of evidence values.


# #### emp_length_int

# In[211]:


# Leave as is.
# '60' will be the reference category.
df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)


# In[212]:


# emp_length_int
df_inputs_prepr['emp_length_int'].unique()
# Has only 11 levels: from 0 to 10. Hence, we turn it into a factor with 11 levels.


# In[213]:


df_temp = woe_ordered_continuous(df_inputs_prepr, 'emp_length_int', df_output_prep)
# We calculate weight of evidence.
df_temp


# In[214]:


plot_by_woe(df_temp)


# In[215]:


# We create the following categories: '0', '1', '2 - 4', '5 - 6', '7 - 9', '10'
# '0' will be the reference category
df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)


# #### months since issue

# In[217]:


df_inputs_prepr.head(5)


# In[218]:


df_inputs_prepr.mths_since_issue.unique()


# Fine classing of continous or discrete high ordered variable

# In[220]:


df_inputs_prepr['mths_since_issue'] = pd.cut(df_inputs_prepr['mths_since_issue'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.


# In[221]:


df_inputs_prepr.mths_since_issue.unique()


# In[222]:


df_inputs_prepr['mths_since_issue']


# In[223]:


# mths_since_issue_d
df_mnths_since_issue = woe_ordered_continuous(df_inputs_prepr, 'mths_since_issue', df_output_prep)
# We calculate weight of evidence.
df_mnths_since_issue.head(10)


# In[224]:


plot_by_woe(df_mnths_since_issue)


# In[225]:


plot_by_woe(df_mnths_since_issue,rotation_of_x_labels=90)


# In[226]:


plot_by_woe(df_mnths_since_issue.iloc[3: , : ], 90)
# We plot the weight of evidence values.


# In[227]:


# We create the following categories:
# < 38, 38 - 39, 40 - 41, 42 - 48, 49 - 52, 53 - 64, 65 - 84, > 84.
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(38)), 1, 0)
df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(65, 85)), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(85, 127)), 1, 0)


# ### Fine classing

# In[229]:


# int_rate
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.


# In[230]:


df_inputs_prepr['int_rate_factor'].unique()


# In[231]:


df_inputs_prepr['int_rate_factor']


# In[232]:


df_temp = woe_ordered_continuous(df_inputs_prepr, 'int_rate_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head(10)


# In[233]:


plot_by_woe(df_temp,rotation_of_x_labels=90)


# Greater the interest rate, lower the WOE and higher the probability of default (riskier)

# In[235]:


# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'


# In[236]:


df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)


# In[ ]:





# In[237]:


df_inputs_prepr.head(3)


# In[238]:


df_inputs_prepr['funded_amnt'].unique()


# In[239]:


# funded_amnt
df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'funded_amnt_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head(5)


# In[240]:


plot_by_woe(df_temp,rotation_of_x_labels=90)


# In[241]:


### No need to inlude funded amount in the pD model as WOE is independent of the WOE


# ### Data Preparation: Continuous Variables, Part 1 and 2

# In[242]:


# mths_since_earliest_cr_line
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head(5)


# In[243]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[244]:


plot_by_woe(df_temp.iloc[6: , : ], 90)
# We plot the weight of evidence values.


# In[245]:


# We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)


# In[246]:


# delinq_2yrs
df_temp = woe_ordered_continuous(df_inputs_prepr, 'delinq_2yrs', df_output_prep)
# We calculate weight of evidence.
df_temp.head(5)


# In[247]:


plot_by_woe(df_temp)
# We plot the weight of evidence values.


# In[248]:


# Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)


# In[250]:


# inq_last_6mths
df_temp = woe_ordered_continuous(df_inputs_prepr, 'inq_last_6mths', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[251]:


plot_by_woe(df_temp)
# We plot the weight of evidence values.


# In[252]:


# Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)


# In[253]:


# open_acc
df_temp = woe_ordered_continuous(df_inputs_prepr, 'open_acc', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[254]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[255]:


plot_by_woe(df_temp.iloc[ : 40, :], 90)
# We plot the weight of evidence values.


# In[256]:


# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)


# In[258]:


# pub_rec
df_temp = woe_ordered_continuous(df_inputs_prepr, 'pub_rec', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[259]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[260]:


# Categories '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)


# In[261]:


# total_acc
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[262]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[263]:


# Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)


# In[264]:


# acc_now_delinq
df_temp = woe_ordered_continuous(df_inputs_prepr, 'acc_now_delinq', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[265]:


plot_by_woe(df_temp)
# We plot the weight of evidence values.


# In[266]:


# Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)


# In[267]:


# total_rev_hi_lim
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
# Here we do fine-classing: using the 'cut' method, we split the variable into 2000 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[268]:


plot_by_woe(df_temp.iloc[: 50, : ], 90)
# We plot the weight of evidence values.


# In[269]:


# Categories
# '<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)


# In[271]:


# installment
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'installment_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[272]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3

# In[273]:


# annual_inc
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[274]:


df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[276]:


# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]
#loan_data_temp = loan_data_temp.reset_index(drop = True)
#df_inputs_prepr_temp


# In[278]:


df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[279]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[280]:


# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)


# In[281]:


# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[282]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[283]:


# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3
# 

# In[284]:


# annual_inc
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# 50 classes are not enough to fine class annual income as more than 94% lies in first class

# In[286]:


df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[287]:


# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]
#loan_data_temp = loan_data_temp.reset_index(drop = True)
#df_inputs_prepr_temp


# In[288]:


df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[289]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# ######![IncomeVariable.PNG](attachment:IncomeVariable.PNG)

# In[290]:


# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)


# In[291]:


# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[292]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[293]:


# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3

# In[294]:


# dti
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'dti_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[295]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[296]:


# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ]


# In[297]:


df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'dti_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[298]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[299]:


# Categories:
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)


# In[300]:


# mths_since_last_record
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]
#sum(loan_data_temp['mths_since_last_record'].isnull())
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_record_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[301]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[302]:


# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)


# In[303]:


loan_data_inputs_train = df_inputs_prepr.copy()


# In[304]:


loan_data_inputs_train.describe()


# ## Preprocessing the Test Dataset

# In[305]:


df_input_prep = loan_data_inputs_test
df_output_prep = loan_data_outputs_test


# In[306]:


df_input_prep.head()


# Dicrete Data Preprocessing
# Dicrete variable is already categorical so here we have no need to calculate dummy variables using fine and coarse classing
# Only calculate WOE and Information value to estimate if the variable can be included for predicting dependent variable

# In[308]:


df_input_prep['grade'].unique()


# In[309]:


df1 = pd.concat([df_input_prep['grade'],df_output_prep],axis=1)


# In[310]:


df1.head()


# In[311]:


df1.tail()


# ## Weight of evidence of discrete variable Grade

# ![WOE.PNG](attachment:WOE.PNG)

# In[312]:


df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count()


# In[313]:


df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()


# In[314]:


df1 = pd.concat([df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count(),
                df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()],axis=1)


# In[315]:


df1.head()


# In[316]:


df1 = df1.iloc[:,[0,1,3]]


# In[317]:


df1.columns = [df1.columns.values[0],'n_obs','prop_good']


# In[319]:


df1['no_good'] = df1['prop_good'] * df1['n_obs']
df1['no_bad'] = (1- df1['prop_good']) * df1['n_obs']


# In[320]:


df1['Final_good'] = df1['no_good']/df1['no_good'].sum()
df1['Final_bad'] = df1['no_bad']/df1['no_bad'].sum()


# In[321]:


df1['WOE'] = np.log(df1['Final_good']/df1['Final_bad'])


# In[322]:


df1


# In[323]:


df1 = df1.sort_values(['WOE'])


# In[324]:


df1.head()


# In[ ]:


df1.reset_index(drop=True)


# In[326]:


df1['IV'] = (df1['Final_good']-df1['Final_bad']) * df1['WOE']


# In[327]:


df1['IV'] = df1['IV'].sum()


# In[328]:


df1.head()


# Grade Information value is 0.29 which comes under the bracket of 0.1-0.3
# It means medium predictive power to obtain output variable

# #### Function to calculate WOE

# In[330]:


def woe_discrete(df,independent_variable,dependent_variable):
    df = pd.concat([df[independent_variable],dependent_variable],axis=1)
    df = pd.concat([df.groupby(df.columns.values[0],as_index=False)[df.columns.values[1]].count(),
    df.groupby(df.columns.values[0],as_index=False)[df.columns.values[1]].mean()],axis=1)
    df = df.iloc[:,[0,1,3]]
    df.columns = [df.columns.values[0],'n_obs','prop_good']
    df['no_good'] = df['prop_good'] * df['n_obs']
    df['no_bad'] = (1- df['prop_good']) * df['n_obs']
    df['Final_good'] = df['no_good']/df['no_good'].sum()
    df['Final_bad'] = df['no_bad']/df['no_bad'].sum()
    df['WOE'] = np.log(df['Final_good']/df['Final_bad'])
    df = df.sort_values(['WOE'])
    df = df.reset_index(drop=True)
    df['IV'] = (df['Final_good']-df['Final_bad']) * df['WOE']
    df['IV'] = df['IV'].sum()
    return df


# In[331]:


df_temp=woe_discrete(df_input_prep,'grade',df_output_prep)


# In[332]:


df_temp.head()


# #### Visualizing WOE for dicerete variables to interpret it

# In[333]:


sns.set()


# In[334]:


def plot_by_woe(df_woe,rotation_of_x_labels=0):
    x = np.array(df_woe.iloc[:,0].apply(str)) ## matplotlib works better with array than dataframes
    y = df_woe['WOE']
    plt.figure(figsize=(18,6))
    plt.plot(x,y,marker='o',linestyle='--',color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of evidence')
    plt.title(str('Weight of evidence by' + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_x_labels)


# In[335]:


plot_by_woe(df_temp)


# Keeping dummy variable G (grade) as reference
# All other in regression model

# ##### Home Ownership Variable

# In[337]:


df_input_prep.head()


# In[338]:


df_home_owner=woe_discrete(df_input_prep,'home_ownership',df_output_prep)


# In[339]:


df_home_owner.head()


# In[340]:


df_home_owner.tail()


# In[341]:


plot_by_woe(df_home_owner)


# In[342]:


df_home_owner.head()


# in 2nd column(n_obs) it is clearly visible that OTHER, NONE and ANY has few values in the dataset, therefore it is less
# WOE to predict loan default, but it is not good to delete those variables as those are most riskiest values
# , better if we combine them to get good amount of information
# 
# For RENT also, WOE is very low so we can combine it with OTHER,NONE and ANY

# In[344]:


df_input_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_input_prep['home_ownership:OTHER'] ,df_input_prep['home_ownership:RENT'],
                                                          df_input_prep['home_ownership:NONE'],df_input_prep['home_ownership:ANY']])


# From a set of categorical variables that represent one original independent variable, we make a reference category the 
# category with lowest WOE value

# #### address state discrete variable

# In[346]:


df_input_prep['addr_state'].unique()


# In[347]:


df_addr_state=woe_discrete(df_input_prep,'addr_state',df_output_prep)


# In[348]:


df_addr_state.head()


# In[349]:


plot_by_woe(df_addr_state)


# In[350]:


if ['addr_state:ND'] in df_input_prep.columns.values:
    pass
else:
    df_input_prep['addr_state:ND'] = 0


# In[351]:


plot_by_woe(df_addr_state.iloc[2:-2,:])


# Earlier first two and last two states were making us believe that all states from NV to DC wee kind of similar but ideally 
# it is not

# Combine NE, IA, NV, FL, Al, HI based on WOE and number of observation, all of these are having worst borrowers , WOE is lowest
# Being conservative, add ND(North Dakota earlier not in the list) also in this category

# Last four WV,NH,WY,DC and ME,ID are having good borrowers -combine them

# In[355]:


plot_by_woe(df_addr_state.iloc[6:-6,:])


# VA,NM,NY,TN,MO,LA,OK,NC,MD,CA have similar WOE
# However NY and CA have many borrowers so they will be a seperate dummy variable

# Final categories from VA to CA will be;
# 1. VA,NM
# 2. NY
# 3. TN,MO,LA,OK,NC,MA
# 4. CA

# In[358]:


### THEN UT,NJ,AZ,KY


# #### ![ADDR_STATE_DUMMYvARIABLES.PNG](attachment:ADDR_STATE_DUMMYvARIABLES.PNG)

# In[359]:


# We create the following categories:
# 'ND' 'NE' 'IA' NV' 'FL' 'HI' 'AL'
# 'NM' 'VA'
# 'NY'
# 'OK' 'TN' 'MO' 'LA' 'MD' 'NC'
# 'CA'
# 'UT' 'KY' 'AZ' 'NJ'
# 'AR' 'MI' 'PA' 'OH' 'MN'
# 'RI' 'MA' 'DE' 'SD' 'IN'
# 'GA' 'WA' 'OR'
# 'WI' 'MT'
# 'TX'
# 'IL' 'CT'
# 'KS' 'SC' 'CO' 'VT' 'AK' 'MS'
# 'WV' 'NH' 'WY' 'DC' 'ME' 'ID'

# 'IA_NV_HI_ID_AL_FL' will be the reference category.
df_inputs_prepr = df_input_prep.copy()

df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'],
                                              df_inputs_prepr['addr_state:IA'], df_inputs_prepr['addr_state:NV'],
                                              df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'],
                                                          df_inputs_prepr['addr_state:AL']])

df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])

df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                              df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'],
                                              df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])

df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'],
                                              df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])

df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                              df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'],
                                              df_inputs_prepr['addr_state:MN']])

df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                              df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'],
                                              df_inputs_prepr['addr_state:IN']])

df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'],
                                              df_inputs_prepr['addr_state:OR']])

df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])

df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])

df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                              df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'],
                                              df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])

df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                              df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'],
                                              df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])


# In[361]:


df_inputs_prepr.head()


# #### verification status discrete variable

# In[362]:


df_inputs_prepr['verification_status'].unique()


# In[363]:


df_verification_status=woe_discrete(df_input_prep,'verification_status',df_output_prep)


# In[364]:


df_verification_status.head()


# In[365]:


plot_by_woe(df_verification_status)


# #### purpose discrete variable

# In[366]:


df_inputs_prepr['purpose'].unique()


# In[367]:


df_purpose=woe_discrete(df_input_prep,'purpose',df_output_prep)


# In[368]:


df_purpose.head()


# In[369]:


plot_by_woe(df_purpose)


# In[370]:


# We combine 'educational', 'small_business', 'wedding', 'renewable_energy', 'moving', 'house' in one category: 'educ__sm_b__wedd__ren_en__mov__house'.
# We combine 'other', 'medical', 'vacation' in one category: 'oth__med__vacation'.
# We combine 'major_purchase', 'car', 'home_improvement' in one category: 'major_purch__car__home_impr'.
# We leave 'debt_consolidtion' in a separate category.
# We leave 'credit_card' in a separate category.
# 'educ__sm_b__wedd__ren_en__mov__house' will be the reference category.
df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                 df_inputs_prepr['purpose:wedding'], df_inputs_prepr['purpose:renewable_energy'],
                                                                 df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                             df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                        df_inputs_prepr['purpose:home_improvement']])


# In[371]:


# 'initial_list_status'
df_initial_list_status = woe_discrete(df_inputs_prepr, 'initial_list_status', df_output_prep)
df_initial_list_status


# In[372]:


plot_by_woe(df_initial_list_status)
# We plot the weight of evidence values.


# ### Preprocessing Continuous Variables: Automating Calculations and Visualizing Results

# When we calculate and plot the weights of evidence of continuous variables categories, what do we sort them by their own
# values in ascending order

# In[139]:


# WoE function for ordered discrete and continuous variables
def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
# Here we define a function similar to the one above, ...
# ... with one slight difference: we order the results by the values of a different column.
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 1

# In[140]:


# term
df_inputs_prepr['term_int'].unique()
# There are only two unique values, 36 and 60.


# In[141]:


df_term_int = woe_ordered_continuous(df_inputs_prepr, 'term_int', df_output_prep)
# We calculate weight of evidence.
df_term_int


# In[142]:


def plot_by_woe(df_woe,rotation_of_x_labels=0):
    x = np.array(df_woe.iloc[:,0].apply(str)) ## matplotlib works better with array than dataframes
    y = df_woe['WoE']
    plt.figure(figsize=(18,6))
    plt.plot(x,y,marker='o',linestyle='--',color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel('Weight of evidence')
    plt.title(str('Weight of evidence by' + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_x_labels)


# In[143]:


plot_by_woe(df_term_int)
# We plot the weight of evidence values.


# In[144]:


##emp_length_int


# In[145]:


# Leave as is.
# '60' will be the reference category.
df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)


# In[146]:


# emp_length_int
df_inputs_prepr['emp_length_int'].unique()
# Has only 11 levels: from 0 to 10. Hence, we turn it into a factor with 11 levels.


# In[147]:


df_temp = woe_ordered_continuous(df_inputs_prepr, 'emp_length_int', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[148]:


plot_by_woe(df_temp)


# In[149]:


# We create the following categories: '0', '1', '2 - 4', '5 - 6', '7 - 9', '10'
# '0' will be the reference category
df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)


# In[150]:


## months since issue


# In[151]:


df_inputs_prepr.head(5)


# In[152]:


df_inputs_prepr.mths_since_issue.unique()


# In[153]:


### Fine classing of continous or discrete high ordered variable


# In[154]:


df_inputs_prepr['mths_since_issue'] = pd.cut(df_inputs_prepr['mths_since_issue'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.


# In[155]:


df_inputs_prepr.mths_since_issue.unique()


# In[156]:


df_inputs_prepr['mths_since_issue']


# In[157]:


# mths_since_issue_d
df_mnths_since_issue = woe_ordered_continuous(df_inputs_prepr, 'mths_since_issue', df_output_prep)
# We calculate weight of evidence.
df_mnths_since_issue.head()


# In[158]:


plot_by_woe(df_mnths_since_issue)


# In[159]:


plot_by_woe(df_mnths_since_issue,rotation_of_x_labels=90)


# In[160]:


plot_by_woe(df_mnths_since_issue.iloc[3: , : ], 90)
# We plot the weight of evidence values.


# In[161]:


# We create the following categories:
# < 38, 38 - 39, 40 - 41, 42 - 48, 49 - 52, 53 - 64, 65 - 84, > 84.
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(38)), 1, 0)
df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(65, 85)), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue'].isin(range(85, 127)), 1, 0)


# In[162]:


df_inputs_prepr['int_rate'].unique()


# ### Fine classing

# In[163]:


# int_rate
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.


# In[164]:


df_inputs_prepr['int_rate_factor'].unique()


# In[165]:


df_inputs_prepr['int_rate_factor'].head()


# In[166]:


df_temp = woe_ordered_continuous(df_inputs_prepr, 'int_rate_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[167]:


plot_by_woe(df_temp,rotation_of_x_labels=90)


# In[168]:


#### greater the interest rate, lower the WOE and higher the probability of default (riskier)


# In[169]:


# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'


# In[170]:


df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)


# In[ ]:





# In[171]:


df_inputs_prepr.head(3)


# In[172]:


df_inputs_prepr['funded_amnt'].unique()


# In[173]:


# funded_amnt
df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'funded_amnt_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[174]:


plot_by_woe(df_temp,rotation_of_x_labels=90)


# In[175]:


### No need to inlude funded amount in the pD model as WOE is independent of the WOE


# ### Data Preparation: Continuous Variables, Part 1 and 2

# In[176]:


# mths_since_earliest_cr_line
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[177]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[178]:


plot_by_woe(df_temp.iloc[6: , : ], 90)
# We plot the weight of evidence values.


# In[179]:


# We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)


# In[180]:


# delinq_2yrs
df_temp = woe_ordered_continuous(df_inputs_prepr, 'delinq_2yrs', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[181]:


plot_by_woe(df_temp)
# We plot the weight of evidence values.


# In[182]:


# Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)


# In[183]:


# inq_last_6mths
df_temp = woe_ordered_continuous(df_inputs_prepr, 'inq_last_6mths', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[184]:


plot_by_woe(df_temp)
# We plot the weight of evidence values.


# In[185]:


# Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)


# In[186]:


# open_acc
df_temp = woe_ordered_continuous(df_inputs_prepr, 'open_acc', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[187]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[188]:


plot_by_woe(df_temp.iloc[ : 40, :], 90)
# We plot the weight of evidence values.


# In[189]:


# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)


# In[190]:


# pub_rec
df_temp = woe_ordered_continuous(df_inputs_prepr, 'pub_rec', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[191]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[192]:


# Categories '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)


# In[193]:


# total_acc
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[194]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[195]:


# Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)


# In[196]:


# acc_now_delinq
df_temp = woe_ordered_continuous(df_inputs_prepr, 'acc_now_delinq', df_output_prep)
df_temp.head()


# In[197]:


plot_by_woe(df_temp)
# We plot the weight of evidence values.


# In[198]:


# Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)


# In[199]:


# total_rev_hi_lim
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
# Here we do fine-classing: using the 'cut' method, we split the variable into 2000 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[200]:


plot_by_woe(df_temp.iloc[: 50, : ], 90)
# We plot the weight of evidence values.


# In[201]:


# Categories
# '<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)


# In[202]:


# installment
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'installment_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[203]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3

# In[204]:


# annual_inc
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[205]:


df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[206]:


# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]
#loan_data_temp = loan_data_temp.reset_index(drop = True)
#df_inputs_prepr_temp


# In[207]:


df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[208]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[209]:


# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)


# In[210]:


# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[211]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[212]:


# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3
# 

# In[213]:


# annual_inc
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[214]:


### 50 classes are not enough to fine class annual income as more than 94% lies in first class


# In[215]:


df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[216]:


# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]
#loan_data_temp = loan_data_temp.reset_index(drop = True)
#df_inputs_prepr_temp


# In[217]:


df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[218]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# ######![IncomeVariable.PNG](attachment:IncomeVariable.PNG)

# In[219]:


# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)


# In[220]:


# mths_since_last_delinq
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[221]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[222]:


# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)


# ### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3

# In[223]:


# dti
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
# Here we do fine-classing: using the 'cut' method, we split the variable into 100 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'dti_factor', df_output_prep)
# We calculate weight of evidence.
df_temp.head()


# In[224]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[225]:


# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ]


# In[226]:


df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'dti_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[227]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[228]:


# Categories:
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)


# In[229]:


# mths_since_last_record
# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]
#sum(loan_data_temp['mths_since_last_record'].isnull())
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_record_factor', df_output_prep[df_inputs_prepr_temp.index])
# We calculate weight of evidence.
df_temp.head()


# In[230]:


plot_by_woe(df_temp, 90)
# We plot the weight of evidence values.


# In[231]:


# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)


# In[232]:


df_inputs_prepr.describe()


# In[233]:


loan_data_inputs_test = df_inputs_prepr.copy()


# In[ ]:





# In[234]:


# loan_data_inputs_train.to_csv('loan_data_inputs_train.csv')
# loan_data_outputs_train.to_csv('loan_data_targets_train.csv')
# loan_data_inputs_test.to_csv('loan_data_inputs_test.csv')
# loan_data_outputs_test.to_csv('loan_data_targets_test.csv')


# In[ ]:




