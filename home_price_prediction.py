
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import random


    
boston = load_boston()
boston_train_X, boston_test_X, boston_train_Y, boston_test_Y = train_test_split(boston.data, boston.target, train_size=.75)
boston_train_X_main, boston_train_X_validation, boston_train_Y_main, boston_train_Y_validation = train_test_split(boston_train_X, boston_train_Y, train_size=.75)


def regression_model_validation(regr_model_id, isdefault, alpha_val):
    
    if regr_model_id == 0:
        
        if isdefault == True:
            
            regr_model = linear_model.Lasso()
            
        else:
            
            regr_model = linear_model.Lasso(alpha = alpha_val)
            
            
    elif regr_model_id == 1: 
        
        if isdefault == True:
            
            regr_model = linear_model.Ridge()
            
        else:
            
            regr_model = linear_model.Ridge(alpha = alpha_val)
            
            
    
    regr_model.fit(boston_train_X_main, boston_train_Y_main)
    
    regr_model_predict = regr_model.predict(boston_train_X_validation)

    mean_squared_error_value = mean_squared_error(boston_train_Y_validation, regr_model_predict)
    
    
    if regr_model_id == 0:
        
        if isdefault == True:
            
            print("Lasso","alpha(Default):", mean_squared_error_value)
            
        else:
            
            print("Lasso","alpha =", alpha_val, " :", mean_squared_error_value)
            
            
    elif regr_model_id == 1:
        
        if isdefault == True:
            
            print("Ridge","alpha(Default):", mean_squared_error_value)
            
        else:
            
            print("Ridge","alpha =",alpha_val," :", mean_squared_error_value)
            
            
def lassodefault():
    
    regr_lasso_default = linear_model.Lasso()
    regr_lasso_default.fit(boston_train_X, boston_train_Y)  
    regr_predict_lasso_default = regr_lasso_default.predict(boston_test_X)
    
    mean_squared_error_lasso_default = mean_squared_error(boston_test_Y, regr_predict_lasso_default)
    
    print("Lasso","alpha(Default):", mean_squared_error_lasso_default)
    
    
    
def ridgedefault():
    
    regr_ridge_default = linear_model.Ridge()
    regr_ridge_default.fit(boston_train_X, boston_train_Y)  
    regr_predict_ridge_default = regr_ridge_default.predict(boston_test_X)
    
    mean_squared_error_ridge_default = mean_squared_error(boston_test_Y, regr_predict_ridge_default)
    
    print("ridge","alpha(Default):", mean_squared_error_ridge_default)
    
   
def regression_test_tuned(regr_model_id, alpha_val):
    
    if regr_model_id == 0:
        
        regr_model = linear_model.Lasso(alpha = alpha_val)
        
        
    elif regr_model_id == 1:  
        
        regr_model = linear_model.Ridge(alpha = alpha_val)
        
    
    regr_model.fit(boston_train_X, boston_train_Y)
    
    regr_model_predict_test = regr_model.predict(boston_test_X)

    mean_squared_error_value_test = mean_squared_error(boston_test_Y, regr_model_predict_test)
    
    if regr_model_id == 0:
        
        print("Lasso","alpha =", alpha_val, " :", mean_squared_error_value_test)
        
        
    elif regr_model_id == 1:
        
        print("Ridge","alpha =",alpha_val," :", mean_squared_error_value_test)

 
      
print("----------------------------------------------------------------------------------------------------------------")
print("                                  Mean Squared Error(Validation)")
print("----------------------------------------------Lasso---------------------------------------------------------------")

regression_model_validation(0, True, -1)
regression_model_validation(0, False, 0.005)
regression_model_validation(0, False, 0.105)
regression_model_validation(0, False, 0.305)
regression_model_validation(0, False, 0.405)
regression_model_validation(0, False, 0.000305)
print("---------------------------------------------Ridge---------------------------------------------------------------")
regression_model_validation(1, True, -1)
regression_model_validation(1, False, 0.056)
regression_model_validation(1, False, 0.03)
regression_model_validation(1, False, 0.8883)
regression_model_validation(1, False, 0.89029)
regression_model_validation(1, False, 0.9029)
print("-----------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------------------")
print("--------------------------------------Mean Square Error Test(Lasso params tuned)------------------------------------")

lassodefault()
print("------------------------------------------------------")
regression_test_tuned(0, 0.005)
regression_test_tuned(0, 0.105)
regression_test_tuned(0, 0.405)
regression_test_tuned(0, 0.000305)
print("--------------------------------------Mean Square Error Test(Ridge params tuned)---------------------------------------")
ridgedefault()
print("------------------------------------------------------")
regression_test_tuned(1, 0.056)
regression_test_tuned(1, 0.03)
regression_test_tuned(1, 0.883)
regression_test_tuned(1, 0.9029)

