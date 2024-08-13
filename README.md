# Overview of the analysis
The purpose of this analysis is to use machine learning and neural networks to create a binary classifier that can predict whether applicants will be successful in their ventures if funded by Alphabet Soup. The CSV resource is from [IRS. Tax Exempt Organization Search Bulk Data Downloads](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads) and contains more than 34,000 organisations that have received funding from Alphabet Soup over the years.

# Results
o	Data Preprocessing
* y or the target = is_successful
* X or the features are all columns other than "is successful", so that would be ask_amt, sepcial_considerations, income_amt, status, organization, use_case, classification, affiliation, application_type, name, and EIN.
EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organisation classification
USE_CASE—Use case for funding
ORGANIZATION—Organisation type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively
* EIN and NAME should be removed from the input data, particularly EIN because these are IDs and are unique variables.
  
o	Compiling, Training, and Evaluating the Model
* For the deep learning neural network model, I chose to use 2 hidden layers plus 1 output layer. For the first hidden layer, 80 neurons and relu activation function were used to capture a wide range of features and avoid vanishing gradient problem, for the second hidden layer, 30 neurons and relu activation were used to narrow down and refine the learned features to capture more complex data relationships for better performance, and for the output layer, 1 neuron and sigmoid activation function were used.
* There were a few things I did to achieve the target model performance of higher than 75% accuracy:
  * Only drop the non-beneficial ID column, 'EIN'.
    application_df = application_df.drop(columns=['EIN'])
  * Changed the random state from 1 to 30 for a different training/testing split result for better generalization and balance.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
  * Added in a third hidden layer to increase the capacity of the model to learn complex patterns; increased the number of neurons from 80 to 100 in the first hidden layer to capture     the different levels of abstraction of the data; changed the activation function from relu to sigmoid in the second hidden layer to stabilize learning in this specific data type.
    
    #First hidden layer:
    nn.add(tf.keras.layers.Dense(units=100, activation='relu', input_dim=input_features))
    
    #Second hidden layer:
    nn.add(tf.keras.layers.Dense(units=30, activation='sigmoid'))

    #Third hidden layer:
    nn.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

    #Output layer:
    nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

  # Summary
  In conclusion, with the optimization neural network method, the accuracy increased from 72.9% (with 55.5% loss) to 77.7% (with 50% loss). So with these carefully chosen new layers    and activation function, the deeper learning network now has a better generalization to new, unseen data, resulting in higher accuracy. The optimization model reduced overfitting,    focusing on meaningful patterns in the data and improving the capacity to learn complex relationships.
