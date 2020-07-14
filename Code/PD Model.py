#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[2]:


loan_data_inputs_train = pd.read_csv('loan_data_inputs_train.csv',index_col=0)
loan_data_outputs_train = pd.read_csv('loan_data_targets_train.csv',index_col=0,header=None)
loan_data_inputs_test = pd.read_csv('loan_data_inputs_test.csv',index_col=0)
loan_data_outputs_test = pd.read_csv('loan_data_targets_test.csv',index_col=0,header=None)


# In[4]:


pd.options.display.max_rows = None


# In[5]:


loan_data_inputs_train.head()


# In[6]:


loan_data_outputs_train.head()


# In[7]:


loan_data_outputs_train.describe()


# In[8]:


loan_data_inputs_train.shape


# In[9]:


loan_data_outputs_train.shape


# In[10]:


loan_data_inputs_test.shape


# In[11]:


loan_data_outputs_test.shape


# ## Selecting the features

# In[3]:


# Here we select a limited set of input variables in a new dataframe.
inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>=86',
]]


# In[4]:


### Remove the dummy variable references to prevent from getting into dummy trap


# In[5]:


# Here we store the names of the reference category dummy variables in a list.
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'open_acc:0',
'pub_rec:0-2',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']


# In[6]:


inputs_train = inputs_train_with_ref_cat.drop(ref_categories,axis=1)


# In[7]:


inputs_train.head(5)


# In[8]:


inputs_train['mths_since_last_record:>=86'].head()


# In[9]:


inputs_train = inputs_train.drop(['mths_since_last_record:>=86'],axis=1)


# In[19]:


inputs_train.head(3)


# In[ ]:





# In[20]:


loan_data_outputs_train.head(10)


# In[21]:


loan_data_outputs_train.isna().sum()


# In[22]:


reg = LogisticRegression()


# In[23]:


reg.fit(inputs_train,loan_data_outputs_train)


# In[24]:


reg.coef_


# In[25]:


reg.intercept_


# In[26]:


feature_name = inputs_train.columns.values


# In[27]:


summary_table = pd.DataFrame(columns=['feature_name'],data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1 # moving index one row down
summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table.head()


# In[28]:


## Measuring which independent variable is important to measure loan default
## Statistical significance of independent variable is calculated


# In[29]:


### Build a Logitics Regression Model with P values


# In[10]:


# P values for sklearn logistic regression.

# Class to display p-values for logistic regression in sklearn.

from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


# In[11]:


reg = LogisticRegression_with_p_values()


# In[12]:


reg.fit(inputs_train,loan_data_outputs_train)


# In[13]:


reg


# ### Saving the Model

# In[14]:


import pickle


# In[16]:


pickle.dump(reg, open('pd_model.sav', 'wb'))
# Here we export our model to a 'SAV' file with file name 'lgd_model_stage_1.sav'.


# In[ ]:





# In[33]:


summary_table = pd.DataFrame(columns=['feature_name'],data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1 # moving index one row down
summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table.head(10)


# In[34]:


p_values = reg.p_values


# In[35]:


p_values = np.append(np.nan,np.array(p_values))


# In[36]:


summary_table['p_values'] = p_values


# In[37]:


summary_table.head()


# In[38]:


#summary_table.to_csv('All_Columns_Coeff_PVal.csv')


# In[39]:


inputs_train = inputs_train.drop(['delinq_2yrs:0',
'delinq_2yrs:1-3',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:28-51',
'total_acc:>=52',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K'],axis=1)


# In[40]:


inputs_train.head()


# In[41]:


reg = LogisticRegression_with_p_values()


# In[42]:


reg.fit(inputs_train,loan_data_outputs_train)


# In[43]:


reg.coef_


# In[44]:


# Here we select a limited set of input variables in a new dataframe.
inputs_test_with_ref_cat = loan_data_inputs_test.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>=86',
]]


# In[45]:


inputs_test = inputs_test_with_ref_cat.drop(ref_categories,axis=1)


# In[46]:


inputs_test.head()


# In[47]:


inputs_train.head()


# In[48]:


inputs_test = inputs_test.drop(['delinq_2yrs:0',
'delinq_2yrs:1-3',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:28-51',
'total_acc:>=52',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K'],axis=1)


# In[49]:


inputs_test.head()


# In[50]:


inputs_test = inputs_test.drop('mths_since_last_record:>=86',axis=1)


# In[51]:


inputs_test.head()


# ![LR.PNG](attachment:LR.PNG)

# In[52]:


y_hat_test = reg.model.predict(inputs_test)


# In[53]:


y_hat_test


# In[54]:


y_hat_test_proba = reg.model.predict_proba(inputs_test)


# In[55]:


y_hat_test_proba


# ## First column is probability of default or probability of being a being a bad borrower
# ## Second column is probability of not default or probability of being a good borrower

# ## We are concerned with the probability of good borrower

# In[56]:


y_hat_test_proba = y_hat_test_proba[:][:,1]


# In[57]:


y_hat_test_proba


# In[58]:


loan_data_outputs_test_temp = loan_data_outputs_test.copy()


# In[59]:


loan_data_outputs_test_temp.reset_index(drop=True,inplace=True)


# In[60]:


df_actual_predicted_probs = pd.concat([loan_data_outputs_test_temp,pd.DataFrame(y_hat_test_proba)],axis=1)


# In[61]:


df_actual_predicted_probs.head()


# In[62]:


df_actual_predicted_probs.columns = ['Loan(Good/Bad) Actual','Probability of Good/Bad Borrower']


# In[63]:


df_actual_predicted_probs.index = loan_data_inputs_test.index


# In[64]:


df_actual_predicted_probs.tail()


# In[65]:


df_actual_predicted_probs.describe()


# ## Accuracy and Area under the Curve (PD Model)

# In[66]:


# cut off theshold variable tr
tr = 0.9
df_actual_predicted_probs['Loan(Good/Bad) Predicted'] = np.where(df_actual_predicted_probs['Probability of Good/Bad Borrower']>tr,1,0)


# In[67]:


df_actual_predicted_probs.head()


# ### COnfusion Matrix

# In[68]:


pd.crosstab(df_actual_predicted_probs['Loan(Good/Bad) Actual'],df_actual_predicted_probs['Loan(Good/Bad) Predicted'],
           rownames=['Actual'],colnames=['Predicted'])


# In[69]:


pd.crosstab(df_actual_predicted_probs['Loan(Good/Bad) Actual'],df_actual_predicted_probs['Loan(Good/Bad) Predicted'],
           rownames=['Actual'],colnames=['Predicted'])/df_actual_predicted_probs.shape[0]


# In[70]:


## Not a good model -- change the value of the threshold


# In[71]:


## Checking ROC curve


# In[72]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[73]:


## ROC curve is a curve plotting between False positive and true positive


# In[74]:


roc_curve(df_actual_predicted_probs['Loan(Good/Bad) Actual'],df_actual_predicted_probs['Probability of Good/Bad Borrower'])


# In[75]:


## Output is three arrays-
### First array is false positive
### Second array is true positive
### Third array is threshold


# In[76]:


fpr,tpr,thresholds = roc_curve(df_actual_predicted_probs['Loan(Good/Bad) Actual'],
                               df_actual_predicted_probs['Probability of Good/Bad Borrower'])


# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[78]:


plt.plot(fpr,tpr)
plt.plot(fpr,fpr,linestyle='--',color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')


# In[79]:


AUROC = roc_auc_score(df_actual_predicted_probs['Loan(Good/Bad) Actual'],
                               df_actual_predicted_probs['Probability of Good/Bad Borrower'])
AUROC


# ## Model Performance using Gini coefficient and Kolomogrov - Smirnov

# In[80]:


## Gini coefficient
## measures the inequality between defaulter and non defualters in a population
## More the area of curve, better is the model


# In[81]:


## Kolomogrov - Smirnov coeffecient
## Maximum difference between the cumulative distribution functions of good and bad borrowers
## More the difference, better the model


# In[82]:


df_actual_predicted_probs = df_actual_predicted_probs.sort_values('Loan(Good/Bad) Predicted')
# Sorts a dataframe by the values of a specific column.


# In[83]:


df_actual_predicted_probs.head()


# In[84]:


df_actual_predicted_probs.tail()


# In[85]:


df_actual_predicted_probs = df_actual_predicted_probs.reset_index()
# We reset the index of a dataframe and overwrite it.


# In[86]:


df_actual_predicted_probs.head()


# In[87]:


df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1
# We calculate the cumulative number of all observations.
# We use the new index for that. Since indexing in ython starts from 0, we add 1 to each index.
df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['Loan(Good/Bad) Actual'].cumsum()
# We calculate cumulative number of 'good', which is the cumulative sum of the column with actual observations.
df_actual_predicted_probs['Cumulative N Bad'] = df_actual_predicted_probs['Cumulative N Population'] - df_actual_predicted_probs['Loan(Good/Bad) Actual'].cumsum()
# We calculate cumulative number of 'bad', which is
# the difference between the cumulative number of all observations and cumulative number of 'good' for each row.


# In[88]:


df_actual_predicted_probs.head()


# In[89]:


df_actual_predicted_probs.shape[0]


# In[90]:


df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] / (df_actual_predicted_probs.shape[0])
# We calculate the cumulative percentage of all observations.
df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / df_actual_predicted_probs['Loan(Good/Bad) Actual'].sum()
# We calculate cumulative percentage of 'good'.
df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['Loan(Good/Bad) Actual'].sum())
# We calculate the cumulative percentage of 'bad'.


# In[91]:


df_actual_predicted_probs.head()


# In[92]:


# Plot Gini
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Bad'])
# We plot the cumulative percentage of all along the x-axis and the cumulative percentage 'good' along the y-axis,
# thus plotting the Gini curve.
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Population'], linestyle = '--', color = 'k')
# We plot a seconary diagonal line, with dashed line style and black color.
plt.xlabel('Cumulative % Population')
# We name the x-axis "Cumulative % Population".
plt.ylabel('Cumulative % Bad')
# We name the y-axis "Cumulative % Bad".
plt.title('Gini')
# We name the graph "Gini".


# In[93]:


Gini = AUROC * 2 - 1
# Here we calculate Gini from AUROC.
Gini


# In[94]:


# Plot KS
plt.plot(df_actual_predicted_probs['Loan(Good/Bad) Predicted'], df_actual_predicted_probs['Cumulative Perc Bad'], color = 'r')
# We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'bad' along the y-axis,
# colored in red.
plt.plot(df_actual_predicted_probs['Loan(Good/Bad) Predicted'], df_actual_predicted_probs['Cumulative Perc Good'], color = 'b')
# We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'good' along the y-axis,
# colored in red.
plt.xlabel('Estimated Probability for being Good')
# We name the x-axis "Estimated Probability for being Good".
plt.ylabel('Cumulative %')
# We name the y-axis "Cumulative %".
plt.title('Kolmogorov-Smirnov')
# We name the graph "Kolmogorov-Smirnov".


# In[95]:


KS = max(df_actual_predicted_probs['Cumulative Perc Bad'] - df_actual_predicted_probs['Cumulative Perc Good'])
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS


# ## Applying the PD model for decision making

# In[96]:


pd.options.display.max_columns = None


# In[97]:


inputs_test_with_ref_cat.head()


# In[98]:


summary_table


# In[99]:


##### As shown below for example the probability that first borrower will not default is 92.04%


# In[100]:


## Probability of being a good borrower
y_hat_test_proba


# ### Creating a Scorecard

# In[101]:


min_score = 300
max_score = 850


# In[102]:


summary_table


# In[103]:


ref_categories


# In[104]:


df_ref_categories = pd.DataFrame(ref_categories,columns=['feature_name'])
df_ref_categories['Coefficients'] = 0


# In[105]:


df_ref_categories['p values'] = np.nan


# In[106]:


df_ref_categories


# In[107]:


df_scorecard = pd.concat([summary_table,df_ref_categories])
df_scorecard = df_scorecard.reset_index()
df_scorecard


# In[108]:


df_scorecard['Orginal feature name'] = df_scorecard['feature_name'].str.split(':').str[0]


# In[109]:


df_scorecard


# In[110]:


df_scorecard.groupby('Orginal feature name')['Coefficients'].min()


# In[111]:


min_sum_coeff = df_scorecard.groupby('Orginal feature name')['Coefficients'].min().sum()
min_sum_coeff


# In[112]:


df_scorecard.groupby('Orginal feature name')['Coefficients'].max()


# In[113]:


max_sum_coeff = df_scorecard.groupby('Orginal feature name')['Coefficients'].max().sum()
max_sum_coeff


# ![Scaling%20Score.PNG](attachment:Scaling%20Score.PNG)

# In[114]:


df_scorecard['Score -- Calculcation'] = df_scorecard['Coefficients'] * ((max_score-min_score)/(max_sum_coeff-min_sum_coeff))
df_scorecard


# In[115]:


df_scorecard['Score -- Calculcation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coeff) / (max_sum_coeff - min_sum_coeff)) * (max_score - min_score) + min_score
df_scorecard


# In[116]:


df_scorecard['Score - Preliminary'] = df_scorecard['Score -- Calculcation'].round()
df_scorecard


# In[117]:


df_scorecard.groupby('Orginal feature name')['Score - Preliminary'].min().sum()


# In[118]:


df_scorecard.groupby('Orginal feature name')['Score - Preliminary'].max().sum()


# In[119]:


df_scorecard['difference'] = df_scorecard['Score -- Calculcation']-df_scorecard['Score - Preliminary']


# In[120]:


df_scorecard


# In[121]:


df_scorecard['difference'].min()


# In[122]:


df_scorecard['score_final'] = df_scorecard['Score - Preliminary']
df_scorecard['score_final'][62] = -14


# In[123]:


df_scorecard.groupby('Orginal feature name')['score_final'].min().sum()


# In[124]:


df_scorecard.groupby('Orginal feature name')['score_final'].max().sum()


# In[125]:


## Intercept decides the starting score
df_scorecard


# In[126]:


inputs_test_with_ref_cat.head()


# In[127]:


inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat


# In[128]:


inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
# We insert a column in the dataframe, with an index of 0, that is, in the beginning of the dataframe.
# The name of that column is 'Intercept', and its values are 1s.


# In[129]:


inputs_test_with_ref_cat_w_intercept.head()


# In[130]:


inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[df_scorecard['feature_name'].values]
# Here, from the 'inputs_test_with_ref_cat_w_intercept' dataframe, we keep only the columns with column names,
# exactly equal to the row values of the 'Feature name' column from the 'df_scorecard' dataframe.


# In[131]:


inputs_test_with_ref_cat_w_intercept.head()


# In[132]:


scorecard_scores = df_scorecard['score_final']


# In[133]:


inputs_test_with_ref_cat_w_intercept.shape


# In[134]:


scorecard_scores.shape


# In[135]:


scorecard_scores = scorecard_scores.values.reshape(126, 1)


# In[136]:


scorecard_scores.shape


# ## Final score using dot product

# In[137]:


y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
# Here we multiply the values of each row of the dataframe by the values of each column of the variable,
# which is an argument of the 'dot' method, and sum them. It's essentially the sum of the products.


# In[138]:


y_scores.head()


# In[139]:


y_scores.tail()


# ## Turning a credit score card to pd scorecard

# ![score%20to%20pd.PNG](attachment:score%20to%20pd.PNG)

# In[140]:


sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coeff - min_sum_coeff) + min_sum_coeff
# We divide the difference between the scores and the minimum score by
# the difference between the maximum score and the minimum score.
# Then, we multiply that by the difference between the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we add the minimum sum of coefficients.


# In[141]:


y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
# Here we divide an exponent raised to sum of coefficients from score by
# an exponent raised to sum of coefficients from score plus one.
y_hat_proba_from_score.head()


# In[142]:


y_hat_test_proba[0: 5]


# ## Final Decision Making based on cutoff

# In[143]:


# cut off theshold variable tr
tr = 0.9
df_actual_predicted_probs['Loan(Good/Bad) Predicted'] = np.where(df_actual_predicted_probs['Probability of Good/Bad Borrower']>tr,1,0)


# In[144]:


df_actual_predicted_probs.head()


# In[145]:


pd.crosstab(df_actual_predicted_probs['Loan(Good/Bad) Actual'],df_actual_predicted_probs['Loan(Good/Bad) Predicted'],
           rownames=['Actual'],colnames=['Predicted'])


# In[146]:


pd.crosstab(df_actual_predicted_probs['Loan(Good/Bad) Actual'],df_actual_predicted_probs['Loan(Good/Bad) Predicted'],
           rownames=['Actual'],colnames=['Predicted'])/df_actual_predicted_probs.shape[0]


# In[147]:


## Not a good model -- change the value of the threshold


# In[148]:


## Checking ROC curve


# In[149]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[150]:


## ROC curve is a curve plotting between False positive and true positive


# In[151]:


roc_curve(df_actual_predicted_probs['Loan(Good/Bad) Actual'],df_actual_predicted_probs['Probability of Good/Bad Borrower'])


# In[152]:


## Output is three arrays-
### First array is false positive
### Second array is true positive
### Third array is threshold


# In[153]:


fpr,tpr,thresholds = roc_curve(df_actual_predicted_probs['Loan(Good/Bad) Actual'],
                               df_actual_predicted_probs['Probability of Good/Bad Borrower'])


# In[154]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[155]:


plt.plot(fpr,tpr)
plt.plot(fpr,fpr,linestyle='--',color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')


# In[156]:


AUROC = roc_auc_score(df_actual_predicted_probs['Loan(Good/Bad) Actual'],
                               df_actual_predicted_probs['Loan(Good/Bad) Predicted'])
AUROC


# In[157]:


thresholds


# In[158]:


thresholds.shape


# In[159]:


df_cutoffs = pd.concat([pd.DataFrame(thresholds),pd.DataFrame(fpr),pd.DataFrame(tpr)],axis=1)


# In[160]:


df_cutoffs.columns = ['thresholds','fpr','tpr']


# In[161]:


df_cutoffs.head()


# In[162]:


df_cutoffs['thresholds'][0] = 1 - 1 / np.power(10, 16)
# Let the first threshold (the value of the thresholds column with index 0) be equal to a number, very close to 1
# but smaller than 1, say 1 - 1 / 10 ^ 16.


# In[163]:


df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coeff) * ((max_score - min_score) / (max_sum_coeff - min_sum_coeff)) + min_score).round()
# The score corresponsing to each threshold equals:
# The the difference between the natural logarithm of the ratio of the threshold and 1 minus the threshold and
# the minimum sum of coefficients
# multiplied by
# the sum of the minimum score and the ratio of the difference between the maximum score and minimum score and 
# the difference between the maximum sum of coefficients and the minimum sum of coefficients.


# In[164]:


df_cutoffs.head()


# In[165]:


df_cutoffs['Score'][0] = max_score


# In[166]:


df_cutoffs.head()


# In[167]:


df_cutoffs.tail()


# In[168]:


df_actual_predicted_probs.head(10)


# In[169]:


# We define a function called 'n_approved' which assigns a value of 1 if a predicted probability
# is greater than the parameter p, which is a threshold, and a value of 0, if it is not.
# Then it sums the column.
# Thus, if given any percentage values, the function will return
# the number of rows wih estimated probabilites greater than the threshold. 
def n_approved(p):
    return np.where(df_actual_predicted_probs['Probability of Good/Bad Borrower'] >= p, 1, 0).sum()


# In[170]:


df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)
# Assuming that all credit applications above a given probability of being 'good' will be approved,
# when we apply the 'n_approved' function to a threshold, it will return the number of approved applications.
# Thus, here we calculate the number of approved appliations for al thresholds.
df_cutoffs['N Rejected'] = df_actual_predicted_probs['Probability of Good/Bad Borrower'].shape[0] - df_cutoffs['N Approved']
# Then, we calculate the number of rejected applications for each threshold.
# It is the difference between the total number of applications and the approved applications for that threshold.
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / df_actual_predicted_probs['Probability of Good/Bad Borrower'].shape[0]
# Approval rate equalts the ratio of the approved applications and all applications.
df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']
# Rejection rate equals one minus approval rate.


# In[171]:


df_cutoffs.head()


# In[172]:


df_cutoffs.tail()


# In[173]:


## At threshold (cutoff probability) of 10 %, see index 5184, the acception of loan application rate is 51.8 % and rejection is
## 48.16
## Similarly we can use credit scores for cutoff


# In[174]:


df_cutoffs.iloc[5000: 6200, ]
# Here we display the dataframe with cutoffs form line with index 5000 to line with index 6200.


# In[175]:


## At threshold (cutoff probability) of 5 %, see index 1118, the acception of loan application rate is only 19.66 % and rejection is
## 80.33
## Similarly we can use credit scores for cutoff


# In[176]:


df_cutoffs.iloc[1000: 2000, ]
# Here we display the dataframe with cutoffs form line with index 1000 to line with index 2000.


# In[177]:


inputs_train_with_ref_cat.to_csv('inputs_train_with_ref_cat.csv')


# In[178]:


df_scorecard.to_csv('df_scorecard.csv')


# In[ ]:




