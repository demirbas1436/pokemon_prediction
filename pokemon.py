import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
#import graphviz
################################################
# Decision Tree Classification: CART
################################################
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)


##############################
# EXPLORATORY DATA ANALYSIS
##############################
#######################
# 1.) Data Collection
#######################
df = pd.read_csv("Pokemon.csv")
print(df)
########################################################################################################################
# #: Bu değer, Pokémon’un Pokedex adlı bir veritabanında kayıtlı olan numarasını gösterir.
# Name: Bu değer, Pokémon’un adını gösterir.
# Type 1: Bu değer, Pokémon’un birincil türünü gösterir. Pokémon’un türü, onun sahip olduğu
#         özellikleri, güçlü ve zayıf olduğu elementleri ve kullanabildiği veya etkilendiği
#         hareketleri belirler. Örneğin, Squirtle, su türü bir Pokémon’dur ve ateş türü Pokémon’lara
#         karşı güçlü, elektrik türü Pokémon’lara karşı zayıftır.
# Type 2: Bu değer, Pokémon’un ikincil türünü gösterir. Bazı Pokémon’lar, birden fazla türe sahip olabilirler
#         ve bu da onların farklı özelliklere, güçlülüklere ve zayıflıklara sahip olmalarını sağlar.
#         Örneğin, Bulbasaur, hem çimen hem de zehir türü bir Pokémon’dur ve su türü Pokémon’lara karşı güçlü,
#         ateş türü Pokémon’lara karşı zayıftır.
# Total: Bu değer, Pokémon’un toplam istatistik değerini gösterir. Toplam istatistik değeri, Pokémon’un HP, Attack,
#        Defense, Sp. Atk, Sp. Def ve Speed değerlerinin toplamıdır. Toplam istatistik değeri, Pokémon’un genel
#        gücünü ve savaş potansiyelini yansıtır. Örneğin, Mewtwo’nun toplam istatistik değeri 680’dir ve bu da
#        onu en güçlü Pokémon’lardan biri yapar.
# HP: Bu değer, Pokémon’un can puanını gösterir. Can puanı, Pokémon’un ne kadar dayanıklı olduğunu ve ne kadar hasar
#     alabileceğini belirler. Can puanı sıfıra düşen Pokémon, bayılır ve savaşamaz. Örneğin, Chansey’nin can puanı
#     250’dir ve bu da onu en dayanıklı Pokémon’lardan biri yapar.
# Attack: Bu değer, Pokémon’un fiziksel saldırı gücünü gösterir. Fiziksel saldırı gücü, Pokémon’un fiziksel
#         temas gerektiren hareketlerle ne kadar hasar verebildiğini belirler. Fiziksel saldırı gücü yüksek olan
#         Pokémon’lar, fiziksel savunma gücü düşük olan Pokémon’lara karşı avantajlıdır. Örneğin, Machamp’in fiziksel
#         saldırı gücü 130’dur ve bu da onu en güçlü fiziksel saldırganlardan biri yapar.
# Defense: Bu değer, Pokémon’un fiziksel savunma gücünü gösterir. Fiziksel savunma gücü, Pokémon’un fiziksel temas
#          gerektiren hareketlerden ne kadar hasar aldığını belirler. Fiziksel savunma gücü yüksek olan Pokémon’lar,
#          fiziksel saldırı gücü yüksek olan Pokémon’lara karşı avantajlıdır. Örneğin, Steelix’in fiziksel savunma
#          gücü 200’dür ve bu da onu en dayanıklı fiziksel savunmacılardan biri yapar.
# Sp. Atk: Bu değer, Pokémon’un özel saldırı gücünü gösterir. Özel saldırı gücü, Pokémon’un fiziksel temas
#          gerektirmeyen hareketlerle ne kadar hasar verebildiğini belirler. Özel saldırı gücü yüksek olan
#          Pokémon’lar, özel savunma gücü düşük olan Pokémon’lara karşı avantajlıdır. Örneğin, Alakazam’ın özel
#          saldırı gücü 135’tir ve bu da onu en güçlü özel saldırganlardan biri yapar.
# Sp. Def: Bu değer, Pokémon’un özel savunma gücünü gösterir. Özel savunma gücü, Pokémon’un fiziksel temas
#          gerektirmeyen hareketlerden ne kadar hasar aldığını belirler. Özel savunma gücü yüksek olan Pokémon’lar,
#          özel saldırı gücü yüksek olan Pokémon’lara karşı avantajlıdır. Örneğin, Shuckle’ın özel savunma gücü
#          230’dur ve bu da onu en dayanıklı özel savunmacılardan biri yapar.
# Speed: Bu değer, Pokémon’un hızını gösterir. Hız, Pokémon’un savaşta ilk hareket eden olma olasılığını belirler.
#        Hızı yüksek olan Pokémon’lar, hızı düşük olan Pokémon’lara karşı avantajlıdır. Örneğin, Ninjask’ın hızı
#        160’tır ve bu da onu en hızlı Pokémon’lardan biri yapar.
# Generation: Bu değer, Pokémon’un hangi nesilde ortaya çıktığını gösterir. Nesil, Pokémon’un yeni türlerinin
#             tanıtıldığı video oyunu ve anime serilerini ifade eder. Şu ana kadar, sekiz nesil tanıtılmıştır.
#             Örneğin, Charizard, birinci nesil bir Pokémon’dur ve ilk olarak 1996 yılında Pokémon Red ve Pokémon
#             Green oyunlarında görülmüştür.
# Legendary: Bu değer, Pokémon’un efsanevi olup olmadığını gösterir. Efsanevi Pokémon’lar, çok nadir ve çok güçlü
#            olan Pokémon’lardır. Efsanevi Pokémon’lar, genellikle oyunun veya anime’nin hikayesinde önemli bir rol
#            oynarlar.
########################################################################################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
print(cat_cols) #['Type 1', 'Type 2', 'Generation', 'Legendary']
print(num_cols) #['#', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
print(cat_but_car) #['Name']

#######################
# 2.) Data Cleaning
#######################
df = df.dropna()
df = df.drop("#", axis=1)
###########################
# 3.) Univariate Analysis
###########################
###########################
# 4.) Bivariate Analysis
###########################
################################################
# 5.) Dealing with Outliers and Missing Value
################################################

########################
# Base Model
########################
from sklearn.linear_model import LogisticRegression
df = pd.get_dummies(df)
y = df["Legendary"]
X = df.drop("Legendary", axis=1)
log_model = LogisticRegression().fit(X, y)
cart_model = DecisionTreeClassifier().fit(X, y)

y_pred = log_model.predict(X)
y_prob = log_model.predict_proba(X)[:, 1]

#print(classification_report(y, y_pred))
#print(roc_auc_score(y, y_prob))
############################
# for Base cart_model
# accuracy = 1.00
# precision = 1.00
# recall = 1.00
# f1-score = 1.00
# roc_auc_score = 1.0
############################
############################
# for Base log_model
# accuracy = 0.92
# precision = 0.68
# recall = 0.38
# f1-score = 0.48
# roc_auc_score = 0.9616
############################


#####################
# Holdout
#####################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)
log_modelh = LogisticRegression().fit(X_train, y_train)
cart_modelh = DecisionTreeClassifier().fit(X_train, y_train)

y_predh = log_modelh.predict(X_test)
y_probh = log_modelh.predict_proba(X_test)[:, 1]

#print(classification_report(y_test, y_predh))
#print(roc_auc_score(y_test, y_probh))
################################
# for Base cart_model to test
# accuracy = 0.91
# precision = 0.67
# recall = 0.53
# f1-score = 0.59
# roc_auc_score = 0.748
################################
################################
# for Base log_model to train
# accuracy = 0.95
# precision = 0.86
# recall = 0.48
# f1-score = 0.62
# roc_auc_score = 0.9751
################################

#####################
# CV
#####################
log_modelcv = LogisticRegression().fit(X, y)
cart_modelcv = DecisionTreeClassifier().fit(X, y)

cv_results = cross_validate(log_modelcv,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

#print(cv_results["test_accuracy"].mean())
# cart = 0.9153687922421392 log = 0.9082280340875698
#print(cv_results["test_f1"].mean())
# cart = 0.5291353383458647 log = 0.3741880341880342
#print(cv_results["test_roc_auc"].mean())
# cart = 0.7410765765765766 log = 0.7750765765765766


################################################
# Hyperparameter Optimization with GridSearchCV
################################################
y = df["Legendary"]
X = df.drop("Legendary", axis=1)
cart_modelgcv = DecisionTreeClassifier().fit(X, y)
#print(cart_model.get_params())
cart_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)}
cart_best_grid = GridSearchCV(cart_modelgcv, cart_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#print(cart_best_grid)
#print(cart_best_grid.best_params_) #{'max_depth': 6, 'min_samples_split': 6}
#print(cart_best_grid.best_score_) #0.9201880693505731


cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_).fit(X, y)
cart_final = cart_modelgcv.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_resultsgcv = cross_validate(cart_final,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])
#print(cv_resultsgcv["test_accuracy"].mean())
# cart = 0.9129297678518954
#print(cv_resultsgcv["test_f1"].mean())
# cart = 0.5230812324929972
#print(cv_resultsgcv["test_roc_auc"].mean())
# cart = 0.7287432432432432


################################################
# 6. Feature Importance
################################################
#print(cart_final.feature_importances_)
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


#plot_importance(cart_final, X, num=5)

################################################
# Saving and Loading Model
################################################
joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")