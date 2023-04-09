#Python XAISuite code for xaisuiteweb
# (C) Shreyan Mitra

import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import ast
import numpy as np
from sklearn.datasets import*

#Machine Learning Models and Training
from sklearn.model_selection import train_test_split

#I decided to trade-off memory for security. This is a list of all accepted models. Without this, the program works perfectly, but malicious users could hijack the system to execute their own code.
acceptedModels = {"SVC": "sklearn.svm", "NuSVC": "sklearn.svm", "LinearSVC": "sklearn.svm", "SVR": "sklearn.svm", "NuSVR": "sklearn.svm", "LinearSVR": "sklearn.svm", 
                 "AdaBoostClassifier": "sklearn.ensemble", "AdaBoostRegressor": "sklearn.ensemble", "BaggingClassifier": "sklearn.ensemble", "BaggingRegressor": "sklearn.ensemble",
                 "ExtraTreesClassifier": "sklearn.ensemble", "ExtraTreesRegressor": "sklearn.ensemble", 
                 "GradientBoostingClassifier": "sklearn.ensemble", "GradientBoostingRegressor": "sklearn.ensemble",
                 "RandomForestClassifier": "sklearn.ensemble", "RandomForestRegressor": "sklearn.ensemble",
                 "StackingClassifier": "sklearn.ensemble", "StackingRegressor": "sklearn.ensemble",
                 "VotingClassifier": "sklearn.ensemble", "VotingRegressor": "sklearn.ensemble",
                 "HistGradientBoostingClassifier": "sklearn.ensemble", "HistGradientBoostingRegressor": "sklearn.ensemble",
                 "GaussianProcessClassifier": "sklearn.gaussian_process", "GaussianProcessRegressor": "sklearn.gaussian_process", 
                 "IsotonicRegression": "sklearn.isotonic", "KernelRidge": "sklearn.kernel_ridge", 
                 "LogisticRegression": "sklearn.linear_model", "LogisticRegressionCV": "sklearn.linear_model",
                 "PassiveAgressiveClassifier": "sklearn.linear_model", "Perceptron": "sklearn.linear_model", 
                  "RidgeClassifier": "sklearn.linear_model", "RidgeClassifierCV": "sklearn.linear_model",
                  "SGDClassifier": "sklearn.linear_model", "SGDOneClassSVM": "sklearn.linear_model", 
                  "LinearRegression": "sklearn.linear_model", "Ridge": "sklearn.linear_model", 
                  "RidgeCV": "sklearn.linear_model", "SGDRegressor": "sklearn.linear_model",
                  "ElasticNet": "sklearn.linear_model", "ElasticNetCV": "sklearn.linear_model",
                  "Lars": "sklearn.linear_model", "LarsCV": "sklearn.linear_model", 
                  "Lasso": "sklearn.linear_model", "LassoCV": "sklearn.linear_model",
                  "LassoLars": "sklearn.linear_model", "LassoLarsCV": "sklearn.linear_model",
                  "LassoLarsIC": "sklearn.linear_model", "OrthogonalMatchingPursuit": "sklearn.linear_model",
                  "OrthogonalMatchingPursuitCV": "sklearn.linear_model", "ARDRegression": "sklearn.linear_model",
                  "BayesianRidge": "sklearn.linear_model", "MultiTaskElasticNet": "sklearn.linear_model", 
                  "MultiTaskElasticNetCV": "sklearn.linear_model", "MultiTaskLasso": "sklearn.linear_model",
                  "MultiTaskLassoCV": "sklearn.linear_model", "HuberRegressor": "sklearn.linear_model",
                  "QuantileRegressor": "sklearn.linear_model", "RANSACRegressor": "sklearn.linear_model",
                  "TheilSenRegressor": "sklearn.linear_model", "PoissonRegressor": "sklearn.linear_model",
                  "TweedieRegressor": "sklearn.linear_model", "GammaRegressor": "sklearn.linear_model", 
                  "PassiveAggressiveRegressor": "sklearn.linear_model", "BayesianGaussianMixture": "sklearn.mixture",
                  "GaussianMixture": "sklearn.mixture", 
                  "OneVsOneClassifier": "sklearn.multiclass", "OneVsRestClassifier": "sklearn.multiclass", 
                  "OutputCodeClassifier": "sklearn.multiclass", "ClassifierChain": "sklearn.multioutput", 
                   "RegressorChain": "sklearn.multioutput",  "MultiOutputRegressor": "sklearn.multioutput",
                   "MultiOutputClassifier": "sklearn.multioutput", "BernoulliNB": "sklearn.naive_bayes", 
                  "CategoricalNB": "sklearn.naive_bayes", "ComplementNB": "sklearn.naive_bayes", 
                  "GaussianNB": "sklearn.naive_bayes", "MultinomialNB": "sklearn.naive_bayes", 
                  "KNeighborsClassifier": "sklearn.neighbors", "KNeighborsRegressor": "sklearn.neighbors", 
                  "BernoulliRBM": "sklearn.neural_network", "MLPClassifier": "sklearn.neural_network", "MLPRegressor": "sklearn.neural_network",  
                  "DecisionTreeClassifier": "sklearn.tree", "DecisionTreeRegressor": "sklearn.tree",
                  "ExtraTreeClassifier": "sklearn.tree", "ExtraTreeRegressor": "sklearn.tree"
                 }

from sklearn.base import is_classifier, is_regressor #For model type identification. Necessary for explanation generation and to ensure model is not unsupervised

#OmniXAI Explanatory Models and Visualization
from omnixai.data.tabular import Tabular
from sklearn.preprocessing import*
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard

#For documentation
from typing import Union, Dict

#Explandable data loder. Users can propose functions to load data from a variety of places. As of now, the data loading harbor contains two piers: one for CSV file and one for sk-learn premade datasets.


def load_data_CSV(data:str, target:str, cut: Union[str, list] = None) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular dataset.
    
    :param str data: Pathname for the CSV file where the dataset is found.
    :param str target: Target variable used for training data
    :param Union[str, list] cut: Variables that should be ignored in training. By default, None.
    :return: Tabular object instance representing 'data'
    :rtype: Tabular
    :raises ValueError: if data cannot be loaded
    '''
    try:
        df = pd.read_csv(data) #We read the dataset onto a dataframes object
        if cut is not None:
          df.drop(cut, axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
          print(df)
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object

def load_data_sklearn(datastore:dict, target:str, cut: Union[str, list] = None) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular sklearn dataset for demoing.
 
    :param dict datastore: A dictionary object containing the data
    :param str target: Target variable used for training data
    :param Union[str, list] cut: Variables that should be ignored in training. By default, None
    :return: Tabular object instance representing 'data'
    :rtype: Tabular
    :raises ValueError: if data cannot be loaded
    '''
    try:
        df = pd.DataFrame(data=datastore.data, columns=datastore.feature_names) #We read the dataset (exluding target) onto a dataframes object
        df["target"] = datastore.target #We add the target column
        if cut is not None:
          df.drop([cut], axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object
        
def generate_data(type:str, target:str, cut: Union[int, list] = None, **generationArgs) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular sklearn dataset for demoing.
 
    :param type str: Type of data to generate ("classification" or "regression")
    :param target str: name of target variable
    :param ``**generationArgs``: Arguments to be passed onto the data generation function
    :return: Tabular object instance representing randomly generated dataset
    :rtype: Tabular
    :raises ValueError: if data cannot be loaded
    '''
    df = pd.DataFrame()
    try:
        if (type == "classification"):
            data = make_classification(**generationArgs)
            df = pd.DataFrame(data[0])
            df["target"] = data[1]
        elif (type == "regression"):
            data = make_regression(**generationArgs)
            df = pd.DataFrame(data[0])
            df["target"] = data[1]
        else:
            raise ValueError("Not correct type. Choices are 'classification' or 'regression'.")
          
        if cut is not None:
          df.drop([cut], axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object


def train_and_explainModel(model:str, tabular_data:Tabular, x_ai:list = [], indexList:list = [], scale:bool = True, scaleType:str = "StandardScaler", addendum:str = "", verbose:bool = False, **modelSpecificArgs): # Returns the model function, scaler, and testing dataframe (if applicable)
    ''' A function that attempts to train and explain a particular sklearn model.
    
    :param str model: Name of Model
    :param Tabular tabular_data: Tabular object representing data set to be used in training
    :param list x_ai: List of explanatory models to be used
    :param list indexList: Specific test data instance to be explained, by default empty (indicating all instances should be explained)
    :param bool scale: Whether data should be scaled before training, by default True
    :param str scaleType: Default Scaler type is "StandardScaler". Example: Use "MinMaxScaler" for MultinomialNB model.
    :param str addendum: Added string to explanation files in case multiple models are being trained and explained within the same directory, to prevent overwriting. By default, empty string.
    :param bool verbose: Whether debugging information should be printed, by default False
    :param ``**modelSpecificArgs``: Specific arguments to pass on to the model function
    
    :return: The learning model, a table of predictions, and the scaler (if applicable) if user wants to predict more values. In List format
    :rtype: list  
    '''
   
    #The processing template includes the model, data, explainerlist, and other parameters
    
    returnList = [] #Initialize the list to return 
    
    try:
      assert model in acceptedModels.keys(), "Model is not accepted." #Security Feature: Model string must be recognized by the acceptedModels storage
      exec('from ' + acceptedModels.get(model) + ' import*') #If the model is allowed, import the necessary package
      modeler= eval(model + "( **modelSpecificArgs )") #Create model function from provided model name. This will not work if model is not accepted. This line converts the string representation of the model to a concrete object. Users are given the option to pass in arguments to the new model.
    except Exception as e:
      print("Provided model name is incorrect or is not part of sklearn library. Only supervised learning models are supported. Refer to models by their associated functions. For example, if you want to use support vector regression, pass in \"SVR\". \n Error message: " + str(e))
      log = open("Failed_Models.txt", 'a', newline = '\n') #If model failed to be recognized or created, we update the error log.
      log.write(model + ": " + str(e) + "\n") 
      return None #There is nothing to return if the model failed to be created.
    

    #Tranform data: Further processing of the data in preparation for model training
    transformer = TabularTransform(
            target_transform=Identity()
        ).fit(tabular_data)
    x = transformer.transform(tabular_data) #To simplify the code, we create a new variable to refer to the data


    #Create training and testing portions for the model by splitting the dataset into 80% train and 20% test. This proportion cannot be changed as of the current version.
    x_train, x_test, y_train, y_test = \
        train_test_split(x[:, :-1], x[:, -1], test_size = 0.2, random_state = 0)

    #Scale data
    if (scale): #If the user wants to scale the data
        scaler = eval(scaleType + "()") #TODO: to be fixed in future versions. Possible security risk. 

        x_train = scaler.fit_transform(x_train) #Fit the scaler to the feature training values
        
        returnList.append(scaler) #Append the fitted scaler to the return list. This is necessay in case the user wants to test the model on a separate dataset. It also ensures the model is usable for however long the user wants to use it after running the program.

        x_test = scaler.transform(x_test) #Use the scaler to normalize / transform the testing dataset

   
    try:
        modeler.fit(x_train, y_train) #This is where the MAGIC happens! The model is trained.
    except Exception as e: #If something went wrong, we need to update the log
        print("Provided model could not be fit to data. Error message: " + str(e))
        log = open("Failed_Models.txt", 'a', newline = '\n')
        log.write(model + ": " + str(e) + "\n") 
        return None
    
    returnList.append(modeler) #We append the ready-to-be-used model to the return list
    
    print("\n")
    comparison_data = pd.DataFrame(data = list(zip(modeler.predict(x_test), y_test, [a_i - b_i for a_i, b_i in zip(modeler.predict(x_test), y_test)])), columns = [model + ' Predicted ' + tabular_data.target_column, 'Actual ' + tabular_data.target_column, "Difference"]) #Model results to be returned or displayed (if verbose is enabled)
    
    returnList.append(comparison_data)
    
    try:
        score = modeler.score(x_test, y_test) #Calculates R^2 value by comparing predicted target values and actual target values
        print(model + " score is " + str(score)) #Print model score
    except Exception as e:
        print("Could not retrieve model score. " + str(e)) #This will occur if an error occurred while calculating score or if model does not support score calculation

    
    if(verbose): #Useful debugging data
        
        print(returnList)
        print("LEARNING FROM DATA...\n ") #Redundant title to separate output from other possible debugging messages
        #print(comparison_data) #Print results of model training with Actual Target and Predicted Target
        filepath = Path('modelresults.csv')  
        comparison_data.to_csv(filepath) #Save data to CSV file


    if (len(x_ai) == 0): #List of explanatory methods is blank
      print("NO EXPLANATIONS REQUESTED BY USER") #Print that no explanations were requested by user
      return returnList #Return the trained model for future use by user
        
    # Convert the transformed data back to Tabular instances. We need to reverse the data processing done at the beginning of this module
    train_data = transformer.invert(x_train)
    test_data = transformer.invert(x_test)
    
    
    #Again, works for only sklearn models. Checks whether model is a regressor or classifier. No unsupervised learning models allowed. 
    if (is_regressor(eval(model + "()"))): 
        detectedmode = 'regression'
    elif(is_classifier(eval(model + "()"))):
        detectedmode = 'classification'
    else:
        raise ValueError('Model provided is not supervised.')
    
    #Create Tabular Explainer object from processing template in preparation for generating explanations
    explainers = TabularExplainer(
        explainers=x_ai,
        mode=detectedmode,
        data=train_data,
        model=modeler,
        preprocess= lambda z: transformer.transform(z), #The data processing function is also provided to the TabularExplainer object
        params = {"shap": {"link": "identity"}}
    )
    
    # Another piece of MAGIC and the most important line! Generate explanations
    test_instances = test_data
    
    #Get explanations, both local and global
    local_explanations = explainers.explain(X=test_instances)
    global_explanations = explainers.explain_global(params = {"pdp": {"features": (tabular_data.to_pd().drop([tabular_data.target_column], axis=1)).columns}})
    if (verbose):
      print("GENERATING EXPLANATIONS FOR MODEL...\n")
      for k,v in local_explanations.items(): #For debugging, print dictionary containing local explanations
        print(k, v.get_explanations()) 

    #If verbose is enabled, display graphs of the explanations
    try:
      for i in range (len(x_ai)): #For each explanatory method requested by user
          if(verbose):
            print(x_ai[i].upper() + " Results:") #Print name of explanatory method as title
          if (x_ai[i] in local_explanations.keys()): #If the explanatory method is local (it explains one instance at a time)
            if (verbose):
                print(local_explanations[x_ai[i]].get_explanations()) #Print explanations if debugging

            #Store explanations in CSV file
            keys = local_explanations[x_ai[i]].get_explanations()[0].keys() 

            CSVFile = x_ai[i] + " ImportanceScores - " + model + " " + tabular_data.target_column + addendum + ".csv" 

            with open(CSVFile, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(local_explanations[x_ai[i]].get_explanations())

            #Now, show explanation graphs for requested instances (only verbose = True)
            if (verbose):
                if (not indexList): #If no requested instances provided
                  indexList = range(0,len(test_instances)) #Show all instances
                for index in indexList:
                  local_explanations[x_ai[i]].ipython_plot(index) #Otherwise, show requested instances
          else: #Explanatory method is global (for the entire dataset)
              try:
                 if(verbose):
                  global_explanations[x_ai[i]].ipython_plot() #Show explanation graph
              except:
                  raise ValueError(x_ai[i] + " is not a valid explanatory method or was not requested.") #If we get to this line, explanation method is not valid because it is not present in either local explanations or global explanations dictionaries
    except Exception as e:
      print("EXPLANATIONS FAILED - " + str(e)) #Something went wrong

    return returnList #In the end, return trained model, scaler, and model results for future use by user. Compatible with analyzer, so model results may be fed in a addendum information to compare_explanations()

def compare_explanations(filenames:list, showGraphics = True, verbose = False, **addendumkwargs): #Analyze the generated explanations in given files
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    
    :param list filenames: File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    :param bool showGraphics = True: enables analysis graphics
    :params bool verbose = False: enables debugging dialogue
    :param ``**addendumkwargs``: Any additional columns to be added to analysis, such as dataset name. Any list to be added to graphics should be of the form addendumName = [addendumList]
    :return: None
    '''
  dataset = addendumkwargs["dataset"] if "dataset" in addendumkwargs else "" #If the user passes in the name of the dataset to be used as an addendum to the generated analysis CSV, store the value in a variable called dataset, eitherwise the adendum to the file name is empty
  if "dataset" in addendumkwargs: #If user passes in the name of the dataset, delete it from the addendumkwargs after sotring its value. All other addendumkwargs must be of the form addendumName = [addendumList]] and will be plotted on analysis graphics
    del addendumkwargs["dataset"]
  
  model = filenames[0].split()[3] #All files should have the same model, so we take the model name to be the third word in the file name of the first file. If you want to compare with other models, use the addendum option
  print(model.upper() + "\n" + "============") # Title
  print("There are " + str(len(filenames)) + " files.") #Clarify the user has passed in a certain number of files
  corrList = [] #Create a list to store the correlations between explainers if the number of explainers is 2
  try:
    df = pd.read_csv(filenames[0]) #Read the first file onto a dataframe
    df['features'][0] = ast.literal_eval(df['features'][0]) #Make sure to compare the string representation of the list of features for the first instance into an actual list
    for feature in sorted(df['features'][0]): #For each feature listed on the first instance (sorted so that this applies to the first instances for any model and explainer). We assume that all instances provide importance scores for all features
        if len(filenames) != 2: #If there are more than 2 explainers to compare (with 1 explainer, there will be nothing to compare)
          compare_explanationssinglef(filenames, feature, verbose, **addendumkwargs) #Compare the explanations for the specific features
        else: #If there are exactly 2 explainers to compare, then a single correlation can be found and stored
           corrList.append(compare_explanationssinglef(filenames, feature, verbose, **addendumkwargs)) #Compare the explanations for the specific features
    
    if len(filenames) == 2: #Find average of correlations if there are only 2 explainers to compare
      import statistics
      print("Average correlation is " + str(statistics.fmean(corrList)))
    
    if len(filenames) == 2: #Graphics and correlation information are only supported for 2 explainers
      print("Printing in-depth information since 2 explainers are provided.")
      try:
        data = pd.read_csv("featuresvsmodel" + dataset + ".csv") #Read the data storage file onto a dataframe if it already exists
      except Exception as e: #Data storage file not found or could not be retrieved
        with open("featuresvsmodel" + dataset + ".csv", 'w', newline='') as file: #Create the data storage file
          writer = csv.writer(file)
          writer.writerow(["Model"] + sorted(df['features'][0])) #Create the header for the file: Model | Feature 1 | Feature 2 etc. Note the order of the features is as they appear in the first file's first instance
        data = pd.read_csv("featuresvsmodel" + dataset + ".csv") #Now read the data storage file onto a dataframe
      finally: #Now that we have the data storage file created and read onto a dataframe
        data.loc[len(data.index)] = [model] + corrList # Add a new row with the model name and the explainer correlations for each feature
        data.set_index('Model', inplace=True, drop=True) #Set the index of the dataframe to the model name instead of 0, 1, 2, 3...
        display(data) #Display the dataframe
        plt.matshow(data) #Create a heatmap of correlations
        plt.title("Correlation between " + filenames[0].split()[0] + " and " + filenames[1].split()[0]) #For clarification, print the two explainers the user wants to compare
        #Set the axis labels for the heat map to be Features vs Models
        plt.xlabel("Features")
        plt.ylabel("Model")
        plt.xticks(ticks = range (0, len(df['features'][0])), labels = sorted(df['features'][0]))
        plt.yticks(ticks = range (0, len(data.index)), labels = data.index)
        plt.colorbar() #Create the color key for the heat map
        plt.savefig('Heat Map ' + dataset + " .png")
        plt.show() if showGraphics == True else print("No graphics shown as requested.") #We construct the graph anyway but only show it if it is asked for
        data.to_csv("featuresvsmodel" + dataset + ".csv") #Write the specific model's explainer correlation scores for all features to the storage file in case the same dataset is trained on different models
        
      
  except Exception as e:
    print("An error occurred while analyzing the graph. " + str(e)) #Something went wrong
  

def compare_explanationssinglef(filenames:list, feature:str, verbose = False, **addendumkwargs): #Analyze the generated explanations in given files for a specific feature. If feature is not available for any instance or any filename, the importance is assumed to be 0, but behavior is unpredictable.
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    
    :param list filenames: File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    :param str feature: Feature whose importance scores are to be compared
    :params bool verbose = False: enables debugging dialogue
    :param ``**addendumkwargs``: Any additional columns to be added to analysis. Each new parameter should be of the form addendumName = [addendumList]]
    :return: Correlation between different explainer files
    '''
  explainers = []
  features = [] #All files should have same features (though not necessarily in the same order) If a feature in one document is absent in the other, the feature's importance score is not considered, or considered to be zero, but behavior is unpredictable.
  model = filenames[0].split()[3] #All files should have the same model
  data = pd.DataFrame()
  for filename in filenames: #For exah explainer file
    try: 
        df = pd.read_csv(filename) #Load the explainer file onto a dataframe

        for i in range(len(df['features'])):
            if(verbose):
                print (df['features'][i])
            df['features'][i] = ast.literal_eval(df['features'][i]) #Make sure all the feature lists are in readable form
            #df.loc[:, ('features', i)] = ast.literal_eval(df['features'][i])

        for i in range(len(df['scores'])):
            if(verbose):
                print (df['features'][i])
            df['scores'][i] = ast.literal_eval(df['scores'][i]) #Make sure all the score lists are in readable form
            #df.loc[:, ('scores', i)] = ast.literal_eval(df['scores'][i])

        #To simplify access to data
        features = df['features']
        scores = df['scores']
        explainer = filename.split()[0]
        explainers.append(explainer) #Create list of explainers

        
        featurescore = [] #List to hold the importance scores of the particular feature 
        for i in range(len(df['features'])): #For each instance
          try:
            featurescore.append(scores[i][features[i].index(feature)]) #Add the score for the feature we are looking for to the list
          except Exception as w:
            if (verbose):
              print("Warning: " + feature + " not found for instance " + str(i) + " in file " + filename + ". Assuming zero importance for that specific instance.") #Sometimes files exclude features whose importances are close to 0, which is why we assume importance for a missing feature is 0. Note that this can lead to unpredictable behavior if two explainer files for different datasets (and therefore nonsimilar features) are chosen.
            featurescore.append(0.0)
            
        data[explainer] =  featurescore #Set a column of the dataframe to the list of scores for the feature given a particular explainer
        
    except Exception as e:
        print("An error occurred while analyzing the graph. " + str(e)) #Something went wrong

 
  for key, value in addendumkwargs.items(): #For each additional list to be plotted
    data[key] = value #Add it to the dataframe
  data.plot(title = feature) #Plot the data
  data.plot(title = feature).get_figure().savefig("Correlation " + feature + " " + filenames[0].replace("/", " ") + " .png")
  data.to_csv(feature + " " + model + ' .csv', index = False) #Store the dataframe on a file
  return data.corr() if len(data.columns) != 2 else data.corr()[explainers[0]][explainers[1]] #Return the correlation if only 2 explainers are being compared
    
def maxImportanceScoreGenerator(filenames:list, verbose = False): #Generate the maxScores addendum list, which can be passed onto the other compare functions or analyzed independently
    maxdf = pd.DataFrame() #final storage for max scores and features
    explainers = [] #store list of explainers
    for filename in filenames: #For eaxh explainer file
        df = pd.read_csv(filename) #Read the file
        explainer = filename.split()[0] #Get the corresponding explainer name
        explainers.append(explainer) #Append the explainer to the list of all explainers
        maxScore = [] #Create a list to temporarily store max scores
        maxScoreFeature = [] #Create a list to temporarily store the most important feature in each instance
        for i in range(len(df["scores"])): #For each instance
            df['features'][i] = ast.literal_eval(df['features'][i]) #Make sure all feature lists are in readable form
            df['scores'][i] = ast.literal_eval(df['scores'][i]) #Make sure all the score lists are in readable form
            #For easier readability
            features = df['features'] 
            scores = df['scores']
            if(verbose):
               print(scores[i])
               print (features[i])
            #Get max score and max score feature
            highestscore = max(scores[i], key=abs)
            indexofhighestscore = scores[i].index(highestscore)
            maxScore.append(highestscore)
            if(verbose):
              print("Index of highest score: " + str(indexofhighestscore))
            maxScoreFeature.append(features[i][indexofhighestscore])
        #Unload temporary storage of max score and max score feature onto permanent storage
        maxdf[explainer + " maxScore"] = maxScore
        maxdf[explainer + " maxScoreFeature"] = maxScoreFeature
    display(maxdf) #Display our data so far
    return maxdf.corr() if len(explainers) != 2 else maxdf.corr()[explainers[0] + " maxScore"][explainers[1] + " maxScore"] #Return the correlation if only 2 explainers are being compared
    return maxdf
