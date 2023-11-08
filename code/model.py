import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
import pickle
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.estimators import BicScore
from pgmpy.estimators import PC

# Desarrollo de la sección 3. Ahora con otra red y datos
# Define some important functions

def get_probs_dropout(probs_vector):
    probs_final=[]
    for prob in probs_vector:
        value_prob=prob.values[1]
        probs_final.append(value_prob)

    return probs_final

def evaluate_performance(predictions_prob, true_labels, threshold=0.5):
    # Convert the probabilities into predictions based on the specified threshold
    predictions = [1 if prob >= threshold else 0 for prob in predictions_prob]

    # Compute TP, FP, TN, and FN
    TP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 1])
    FP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 0])
    TN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 0])
    FN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 1])

    # Compute the metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "True positives": TP,
        "False positives": FP,
        "True negatives": TN,
        "False negatives": FN
    }

def plot_roc(probs, true_labels, title="ROC Curve", figsize=(10, 6)):
    """
    This function plots a ROC curve given predicted probabilities and actual labels.
    
    :param probs: Array-like, predicted probabilities.
    :param true_labels: Array-like, actual target values.
    :param title: String, desired title of the plot.
    :param figsize: tuple, size of the figure.
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    
    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    
    # Set aesthetics
    sns.set_style("whitegrid")
    
    # Create the plot
    plt.figure(figsize=figsize)
    lw = 2  # Line width
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=13)
    
    # Display the plot
    plt.show()



# Importar los paquetes requeridos


# Definir el path en dónde se encuentran los datos
path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto 2/data/processed'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_1/project_1_ACTD/data/processed'
path_datos_actual = path_datos_samuel
# Cargar los datos
train_data = pd.read_csv(path_datos_actual+'/train.csv')
test_data = pd.read_csv(path_datos_actual+'/test.csv')
train_data.head()
train_data.columns
test_data.head()
# model1
# Definir la red bayesiana
model1 = BayesianNetwork([("Unemployment rate", "perc_approved_sem1"), ("Inflation rate", "perc_approved_sem1"),
                         ("Debtor", "perc_approved_sem1"), ("Scholarship holder", "perc_approved_sem1"),
                         ("perc_approved_sem1", "perc_approved_sem2"), ("perc_approved_sem2","actual_target"),
                         ("Age at enrollment","actual_target")])

model1.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in model1.nodes():
    print(model1.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model1)
# For each row in the test_data, predict the probability of "lung"
target_probabilities1 = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Age at enrollment": row["Age at enrollment"],
                                                                  "Unemployment rate": row["Unemployment rate"],
                                                                  "Inflation rate": row["Inflation rate"],
                                                                  "Debtor": row["Debtor"],
                                                                  "Scholarship holder": row["Scholarship holder"]})
    target_probabilities1.append(prob)

# Structure learning based on the evidence of model 1


# Let's say you want to keep only the columns 'Age at enrollment', 'Target', and 'Debtor'
columns_to_keep = ['Age at enrollment', 'actual_target', 'Debtor', 'Unemployment rate', 'Scholarship holder', 'Inflation rate']

# Filter the DataFrame
filtered_data = train_data[columns_to_keep]
est = PC(data=filtered_data)
# Restriction Method
estimated_model = est.estimate(variant="stable", max_cond_vars=10)
print(estimated_model)
print(estimated_model.nodes())
print(estimated_model.edges())

##Scoring Methods

# Hill Climb Search Method

scoring_method = K2Score(data=filtered_data)
esth = HillClimbSearch(data=filtered_data)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())
print(scoring_method.score(estimated_modelh))

#Big Score Method
scoring_method = BicScore(data=filtered_data)
esth = HillClimbSearch(data=filtered_data)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=10, max_iter=int(1e4)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())
print(scoring_method.score(estimated_modelh))

## Estimated Model by restriction method is dismissed because of loop dependencies structure
##modelRM = BayesianNetwork([ ("Scholarship holder", "Age at enrollment"), ("actual_target", "Scholarship holder"),
##                            ("Age at enrollment", "actual_target")])

##modelRM.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
##for i in modelRM.nodes():
##    print(modelRM.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
##inference = VariableElimination(modelRM)
# For each row in the test_data, predict the probability of "lung"
##target_probabilitiesRM = []
##for index, row in test_data.iterrows():
##    prob = inference.query(variables=["actual_target"], evidence={"Scholarship holder": row["Scholarship holder"],
##                                                                  "Age at enrollment": row["Age at enrollment"]})
##    target_probabilitiesRM.append(prob)

## Estimated Model by Hill Climb Search Method
modelHCS = BayesianNetwork([ ("Scholarship holder", "actual_target")])

modelHCS.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in modelHCS.nodes():
    print(modelHCS.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(modelHCS)
# For each row in the test_data, predict the probability of "lung"
target_probabilitiesHCS = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Scholarship holder": row["Scholarship holder"]})
    target_probabilitiesHCS.append(prob)
# Estimated Model by Big Score Method
# Definir la red bayesiana
modelBS = BayesianNetwork([("Unemployment rate", "Inflation rate"), ("Inflation rate", "Debtor"),
                         ("Debtor", "actual_target")])

modelBS.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in modelBS.nodes():
    print(modelBS.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(modelBS)
# For each row in the test_data, predict the probability of "lung"
target_probabilitiesBS = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Debtor": row["Debtor"]})
    target_probabilitiesBS.append(prob)


# UNTIL HERE WE ALREADY FOUND PREDICTED PROBAILITIES ON TEST

# Get the final probabilities of each model
probs_dropout1 = get_probs_dropout(target_probabilities1)
##probs_dropoutRM = get_probs_dropout(target_probabilitiesRM)
probs_dropoutHCS = get_probs_dropout(target_probabilitiesHCS)
probs_dropoutBS = get_probs_dropout(target_probabilitiesBS)




probs_df_1 = pd.DataFrame(probs_dropout1, columns=['Probability'])
##probs_df_RM = pd.DataFrame(probs_dropoutRM, columns=['Probability'])
probs_df_HCS = pd.DataFrame(probs_dropoutHCS, columns=['Probability'])
probs_df_BS = pd.DataFrame(probs_dropoutBS, columns=['Probability'])

# Get descriptive statistics
quantile_cutoff = 1 - train_data['actual_target'].mean()
print(quantile_cutoff)


cutoff1 = probs_df_1['Probability'].quantile(quantile_cutoff)
##cutoffRM = probs_df_RM['Probability'].quantile(quantile_cutoff)
cutoffHCS = probs_df_HCS['Probability'].quantile(quantile_cutoff)
cutoffBS = probs_df_BS['Probability'].quantile(quantile_cutoff)+0.001

# Métricas de desempeño
performance1 = evaluate_performance(probs_dropout1, test_data['actual_target'], threshold=cutoff1)
##performanceRM = evaluate_performance(probs_dropoutRM, test_data['actual_target'], threshold=cutoffRM)
performanceHCS = evaluate_performance(probs_dropoutHCS, test_data['actual_target'], threshold=cutoffHCS)
performanceBS = evaluate_performance(probs_dropoutBS, test_data['actual_target'], threshold=cutoffBS)

print('------------------------')
print('Model 1 results')
for key, value in performance1.items():
    print(f"{key}: {value}")

print('------------------------')
print('Estimated Model by Hill Climb Search method results')
for key, value in performanceHCS.items():
    print(f"{key}: {value}")

print('------------------------')
print('Estimated Model by Bis Score results')
for key, value in performanceBS.items():
    print(f"{key}: {value}")


plot_roc(probs_dropout1, test_data['actual_target'], title="Model 1 ROC Curve: Testing data")
##plot_roc(probs_dropoutRM, test_data['actual_target'], title="Restriction Method Model ROC Curve: Testing data")
plot_roc(probs_dropoutHCS, test_data['actual_target'], title="Hill Climb Search Model ROC Curve: Testing data")
plot_roc(probs_dropoutBS, test_data['actual_target'], title="Bic Score Model ROC Curve: Testing data")
# The best model was model #1

# Save the best model: Model 1
parts = path_datos_actual.split('/')

desired_path = '/'.join(parts[:(len(parts)-1)])

with open(desired_path+'/model1.pkl', 'wb') as file:
    pickle.dump(model1, file)
# Save : Estimated Model
parts = path_datos_actual.split('/')

desired_path = '/'.join(parts[:(len(parts)-1)])

with open(desired_path+'/hill_climb_search_model.pkl', 'wb') as file:
    pickle.dump(modelHCS, file)