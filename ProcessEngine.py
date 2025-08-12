import pandas as pd
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


def get_between(string, char1, char2):
    content=string.split(char1)[1].strip()
    content=content.split(char2)[0].strip()
    return content



class Line:
    def __init__(self, text_line):
        self.text_line=text_line

        parts=self.text_line.split(':', 1)
        self.concern=parts[0].strip()
        scheme=parts[1].strip()
        self.com_sep_elements=[s.strip() for s in scheme.split(',')]
        self.alternate_concerns=['TARGET', 'REMOVE COLUMNS']
        self.norm_call=('NORM' in self.text_line)
        self.outlier_call=('OUTLIER' in self.text_line)
        self.impute_call=('IMPUTE' in self.text_line)
        self.simple_impute=('IMPUTE(mean)' in self.text_line or 'IMPUTE(median)' in self.text_line or 'IMPUTE(mode)' in self.text_line)
        self.bad_values=('[' in self.text_line and ']' in self.text_line)
        self.KN_impute=('IMPUTE(KN' in self.text_line)
        self.remove_row_call=('REMOVE' in self.text_line and 'REMOVE' not in self.concern)
        self.is_mapping=('=' in self.text_line)
        self.is_OHE=(self.concern not in self.alternate_concerns and 'NORM' not in self.text_line and 'OUTLIER' not in self.text_line and '=' not in self.text_line)
        self.remove_columns=(self.concern=='REMOVE COLUMNS')


    def get_mappings(self):
        if self.concern in ['TARGET', 'REMOVE COLUMNS', 'KNEIGHBOURS']: return None
        dictionary={}
        for element in self.com_sep_elements:
            if '=' in element:
                raw, formal = element.split('=', 1)
                raw, formal = raw.strip(), formal.strip()
                
                if formal=='REMOVE' or formal.startswith('IMPUTE'):
                    continue
                
                dictionary[raw]=float(formal)
            else:
                continue
        
        if dictionary=={}: return
        return dictionary


    def get_outlier_criteria(self):
        if self.concern in ['TARGET', 'REMOVE COLUMNS', 'KNEIGHBOURS']: return None
        d={}
        for element in self.com_sep_elements:
            if element.startswith('OUTLIER'):
                if element=='OUTLIER(IQR)':
                    d['method'], d['cutoff'] = element.split('(')[1].strip().split(')')[0].strip(), 1.5
                    return d
                
                method=element.split('(')[1].strip()
                cutoff=element.split('(')[2].strip().split(')')[0].strip()
                cutoff=float(cutoff)
                d['method'], d['cutoff'] = method, cutoff
                return d


    def get_simple_method(self):
        if 'IMPUTE(mode)' in self.text_line:
            return 'mode'
        if 'IMPUTE(median)' in self.text_line:
            return 'median'
        if 'IMPUTE(mean)' in self.text_line:
            return 'mean'
        
    
    def get_K(self):
        split=self.text_line.split('IMPUTE(KN(')[1].strip()
        value=int(split.split(')')[0].strip())
        return value

    
    def get_bad_values(self):
        bad_values=get_between(self.text_line, '[', ']')
        if ',' not in bad_values:
            return [bad_values]

        if ',' in bad_values:
            return [s.strip() for s in bad_values.split(',')]
    
    def is_norm(self):
        return 'NORM' in self.com_sep_elements



class FunctionalFormatter: # This object applies one-hot encodings, formal mappings, and centralizes the format of missing values into nan 
    def __init__(self):
        self.mappings={}
        self.OHE_dict={}
        self.bad_values_dict={}
        self.row_removals={}
        self.col_removals=[]
        self.is_fit=False
    

    def fit(self, config):
        text_lines=config.strip().splitlines()
        for text_line in text_lines:
            if text_line=='':
                continue
            line=Line(text_line)
            concern=line.concern
            if line.is_mapping: self.mappings[concern]=line.get_mappings()
            if line.is_OHE: self.OHE_dict[concern]=line.com_sep_elements
            if line.bad_values: self.bad_values_dict[concern]=line.get_bad_values()
            if concern=='REMOVE COLUMNS': self.col_removals=line.com_sep_elements
            if 'REMOVE' in text_line and 'REMOVE' not in concern: self.row_removals[concern]=[value for value in line.get_bad_values()]

        self.is_fit=True
    
    def transform(self, df):

        if self.is_fit==False: raise ValueError('You must first fit the functional transformer before applying it')

        for col in self.mappings.keys():
            df[col]=df[col].replace(self.mappings[col])

        for col in self.OHE_dict.keys():
            for cat in self.OHE_dict[col]:
                df[cat]=df[col].apply(lambda x: 1 if x==cat else 0)
        
        for col in self.bad_values_dict.keys():
            for val in self.bad_values_dict[col]:
                if col in self.row_removals.keys():
                    df=df[df[col]!=val]
                else:
                    df[col]=df[col].replace({val:np.nan})
        
        cols_to_drop=[]
        for col in self.col_removals:
            cols_to_drop.append(col)
        df=df.drop(columns=cols_to_drop)

        
        return df
    
    def fit_transform(self, df, config):
        self.fit(config)
        return self.transform(df)
    




class OutlierProcessor:
    def __init__(self):
        self.outlier_dict={}
        self.is_fit=False
    
    def fit(self, df, config):

        text_lines=config.strip().splitlines()
        for text_line in text_lines:
            
            if text_line=='':
                continue
            
            line=Line(text_line)
            concern=line.concern

            if line.outlier_call:
                self.outlier_dict[concern]=line.get_outlier_criteria()
                method=self.outlier_dict[concern]['method']
                col=concern
                if method=='percentile':
                    pct=self.outlier_dict[col]['cutoff']/100
                    low=df[col].quantile(1-pct)
                    high=df[col].quantile(pct)

                if method=='z':
                    z=self.outlier_dict[col]['cutoff']
                    low=df[col].mean()-z*df[col].std()
                    high=df[col].mean()+z*df[col].std()
            
                if method=='IQR':
                    Q1=df[col].quantile(.25)
                    Q3=df[col].quantile(.75)
                    cutoff=self.outlier_dict[col]['cutoff']
                    IQR=Q3-Q1
                    low=Q1-cutoff*IQR
                    high=Q3+cutoff*IQR
        
                self.outlier_dict[col]['low']=low
                self.outlier_dict[col]['high']=high

            self.is_fit=True

    
    def transform(self, df):
        if self.is_fit==False: raise ValueError('You must fit before transforming')
        if self.outlier_dict == {}: return df

        for col in self.outlier_dict.keys():
            low=self.outlier_dict[col]['low']
            high=self.outlier_dict[col]['high']
            df = df[(df[col] > low) & (df[col] < high)]
        return df
    
    def fit_transform(self, df, config):
        self.fit(df, config)
        return self.transform(df)
    


class SimpleImputer:
    def __init__(self):
        self.method_dict={}
        self.impute_dict={}
        self.is_fit=False
    

    def fit(self, df, config):
        text_lines=config.strip().splitlines()
        for text_line in text_lines:
            if text_line=='':
                continue
            line=Line(text_line)
            concern=line.concern
            if line.simple_impute:
                self.method_dict[concern]=line.get_simple_method()
                method=self.method_dict[concern]
                if method=='mean': self.impute_dict[concern]=df[concern].mean()
                if method=='median': self.impute_dict[concern]=df[concern].median()
                if method=='mode': self.impute_dict[concern]=df[concern].mode().iloc[0]
        self.is_fit=True
    
    def transform(self, df):

        if self.is_fit==False: raise ValueError('You must first fit the simple imputer before transforming')
        if self.method_dict == {}: return df

        for col in self.impute_dict.keys():
            imputer=self.impute_dict[col]
            df[col]=df[col].fillna(imputer)
        return df
    
    def fit_transform(self, df, config):
        self.fit(df, config)
        return self.transform(df)




class KNeighboursImputer:
    def __init__(self):
        self.impute_dict={}
        self.K_dict={}
        self.norm_dict={}
        self.feature_types={'continuous': [], 'rank or category': []}
        self.is_fit=False
        self.target=None
    

    def get_target(self, config):
        text_lines=config.strip().splitlines()

        for text_line in text_lines:
            if text_line.split(':')[0].strip()=='TARGET':
                self.target = text_line.split(':')[1].strip()
        if self.target==None: raise ValueError('You must define a target variable!')

    def define_types(self, df, threshold=10):
        for col in df.keys():
            n=df[col].nunique(dropna=True)
            if n <= threshold: 
                self.feature_types['rank or category'].append(col)
            else:
                self.feature_types['continuous'].append(col)
                self.norm_dict[col]={'mean': df[col].mean(), 'std': df[col].std()}


    def prepare_KN_helper(self, df, col):
        df_copy=df.copy()
        columns_to_drop=[]
        for c in self.K_dict.keys():
            if c==col: continue
            columns_to_drop.append(c)
            
        for col_continuous in self.feature_types['continuous']:
            mu=self.norm_dict[col_continuous]['mean']
            std=self.norm_dict[col_continuous]['std']
            df_copy[col_continuous]=(df_copy[col_continuous]-mu)/std
            
        columns_to_drop.append(self.target)
        df_copy=df_copy.drop(columns=columns_to_drop)
        return df_copy

    
    def prepare_KN_fit(self, df, col):
        df=self.prepare_KN_helper(df,col)
        return df.dropna()
    
    def prepare_KN_transform(self, df, col):
        df=self.prepare_KN_helper(df,col)
        df=df[df[col].isna()]
        df=df.drop(columns=[col])
        return df


    

    def fit(self, df, config):
        self.define_types(df)
        self.get_target(config)
        text_lines = config.strip().splitlines()

        self.feature_names_ = {}  # store per-column feature order

        for text_line in text_lines:
            if text_line == '':
                continue
            line = Line(text_line)
            concern = line.concern

            if line.KN_impute:
                k = line.get_K()
                self.K_dict[concern] = k
            
        for col in self.K_dict.keys():
            k=self.K_dict[col]
            df_KN = self.prepare_KN_fit(df, col)
            y_train = df_KN[col]
            X_train = df_KN.drop(columns=[col])

            # record training feature names for this target column
            #self.feature_names_[col] = X_train.columns.tolist()

            if col in self.feature_types['rank or category']:
                imputer = KNeighborsClassifier(n_neighbors=k)
            elif col in self.feature_types['continuous']:
                imputer = KNeighborsRegressor(n_neighbors=k)

            imputer.fit(X_train, y_train)
            self.impute_dict[col] = imputer
        
        self.is_fit=True

        

    def transform(self, df):
        if self.impute_dict == {}:
            return df

        for col in self.impute_dict.keys():
            df_transform_input = self.prepare_KN_transform(df, col)
            if df_transform_input.empty:
                continue

            # force columns to match fit-time ordering, inserting NaNs if missing
            #df_transform_input = df_transform_input.reindex(
            #    columns=self.feature_names_[col]
            #)

            predicted = self.impute_dict[col].predict(df_transform_input)
            mask = df[col].isna()
            df.loc[mask, col] = predicted

        return df


    def fit_transform(self, df, config):
        self.fit(df, config)
        return self.transform(df)

        
class Normer:

    def __init__(self):
        self.norm_dict={}
        self.is_fit=False
    
    def fit(self, df, config):
        text_lines=config.strip().splitlines()

        for text_line in text_lines:
            if text_line=='':
                continue
            line=Line(text_line)
            concern=line.concern
            if 'NORM' in line.text_line:
                self.norm_dict[concern]={'mean': df[concern].mean(), 'std': df[concern].std()}
        
        self.is_fit=True
            

    def transform(self,df):
        if self.is_fit==False: raise ValueError('You must first fit the NormManager before transforming with it.')
        if self.norm_dict == {}: return df

        for col in self.norm_dict.keys():
            mu=self.norm_dict[col]['mean']
            std=self.norm_dict[col]['std']
            df[col]=(df[col]-mu)/std
        
        return df

    def fit_transform(self, df, config):
        self.fit(df, config)
        return self.transform(df)

    





class Processor:
    def __init__(self):
        self.config=None
        self.FunctionalFormatter=FunctionalFormatter()
        self.OutlierProcessor=OutlierProcessor()
        self.SimpleImputer=SimpleImputer()
        self.KNeighboursImputer=KNeighboursImputer()
        self.Normer=Normer()

    def fit(self, df, config):

        df=self.FunctionalFormatter.fit_transform(df, config)
        df=self.OutlierProcessor.fit_transform(df, config)
        df=self.SimpleImputer.fit_transform(df, config)
        df=self.KNeighboursImputer.fit_transform(df, config)
        df=self.Normer.fit(df, config)
        self.config=config
        self.is_fit=True
    
    def transform(self, df):
        if self.is_fit==False: raise ValueError("You must fit ProcessEngine's Processor before using its transform function.")
        df=self.FunctionalFormatter.transform(df)
        df=self.OutlierProcessor.transform(df)
        df=self.SimpleImputer.transform(df)
        df=self.KNeighboursImputer.transform(df)
        df=self.Normer.transform(df)
        return df
      
    
    def fit_transform(self, df, config):
        self.fit(df, config)
        return self.transform(df)