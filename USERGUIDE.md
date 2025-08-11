## User schema format acceptable by *Process Engine* 

#### Mappings

Suppose you had a pandas dataframe column called "Gender". You want to assign 1 to "female" and 0 to "male". In *Process Engine*, you would write:  

- *Gender: female=1, male=0*

It is required by *Process Engine* to separate individual mappings by a comma.

#### One-hot encodings
You simply reference your column the same way, but your comma-separated values after the colon are just the names of the columns. If we apply this to gender, we have

- *Gender: female, male*

#### Column removals
Type (all-caps required) "REMOVE COLUMNS:" followed by the comma-separated names of the columns  

- *REMOVE COLUMNS: Gender, Age, Income*


#### Handling bad values: Row removals and imputations

Suppose you wanted to remove the invalid values [!, ?, unknown] of a column. In our gender example, you would have  

- *Gender: female=1, male=0, [!, ?, unknown] <- REMOVE*  

If you wanted to use imputations like KNeighbours, mode, median, or mean imputation, this is how you would type each example respectively  

- *Gender: female=1, male=0, [!, ?, unknown] <- IMPUTE(KN(5))*    
- *Gender: female=1, male=0, [!, ?, unknown] <- IMPUTE(mode)*  
- *Gender: female=1, male=0, [!, ?, unknown] <- IMPUTE(median)*  
- *Gender: female=1, male=0, [!, ?, unknown] <- IMPUTE(mean)*  

#### Outlier handling  

If we had a column, say, "Income", we could remove outliers based on IQR, z-score, or percentile. Respective examples would be  

- *Income: OUTLIER(IQR)*  
- *Income: OUTLIER(percentile(97))*  
- *Income: OUTLIER(z(3))*

#### Normalization  

Suppose you use the column "Income". Simply type  

- *Income: NORM*

#### Defining the target  
You **must** define the target variable if you are calling for KNeighbours imputation of any column. Process Engine needs a target variable to remove from the training inputs.  
If you had a target column "Price", then  
- *TARGET: Price*

#### Combining operations for a given reference to a column.

You may combine mappings, handling bad values, outlier handling, and normalization in one line **as long as they are comma-separated**. Examples:  

- *Risk: low=1, intermediate=2, high=3, OUTLIER(IQR), NORM, [!,?,unknown] <- IMPUTE(mode)*
- *Risk: low=1, intermediate=2, high=3, OUTLIER(IQR), NORM, [!,?,unknown] <- REMOVE*

  
