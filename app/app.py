from flask import Flask, render_template, flash, request , redirect , url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField , IntegerField
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
# import sys
# import logging

rrr = np.zeros(15)

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567hkhjghjf45'

final_user_ratings1 = []
final_user_ratings2 = []
final_user_ratings3 = []

# client = MongoClient('localhost' , 27017)
client = MongoClient("mongodb+srv://username:password@cluster1-st6po.mongodb.net/xxx")

app.debug = True

db = client.movie_recommendations

# ratings_count = db.ratings_compressed.count()
# movies_count = db.movies_compressed.count()
# user_mf_count = db.user_ratings.count()
P = np.zeros((1000,500))
P_orig = np.zeros((1000,500))
# cosine_sim = [[]]
# cosine_sim_item = [[]]
user_ratings_mean = []
new_ratings = np.zeros(500)

user_mf = pd.DataFrame(list(db.user_ratings.find()))
ratings = pd.DataFrame(list(db.ratings_compressed.find()))
movies = pd.DataFrame(list(db.movies_compressed.find()))
user_mf = user_mf.sort_values(['userId' , 'movieId'])

first_15_names = movies.loc[:15,'title']

A = ratings['userId']
B = ratings['movieId']
C = ratings['rating']
# print(user_mf)
# movies = iterator2dataframes(db.movies_compressed.find(), movies_count)
# user_mf = iterator2dataframes(db.user_ratings_for_mf.find(), user_mf_count)

# print(ratings.head())
# print(movies.head())
# print(yo.head())

db.user_user.remove({})
db.item_item.remove({})
db.matrix_fact.remove({})

class ReusableForm(Form):
    name1 = IntegerField('Movie 1:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name2 = IntegerField('Movie 2:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name3 = IntegerField('Movie 3:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name4 = IntegerField('Movie 4:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name5 = IntegerField('Movie 5:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name6 = IntegerField('Movie 6:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name7 = IntegerField('Movie 7:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name8 = IntegerField('Movie 8:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name9 = IntegerField('Movie 9:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name10 = IntegerField('Movie 10:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name11 = IntegerField('Movie 11:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name12 = IntegerField('Movie 12:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name13 = IntegerField('Movie 13:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name14 = IntegerField('Movie 14:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])
    name15 = IntegerField('Movie 15:', validators=[validators.required() , validators.NumberRange(min=1, max=5)])

@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    print(form.errors)
    if request.method == 'POST':
        rating1=request.form['name1']
        rating2=request.form['name2']
        rating3=request.form['name3']
        rating4=request.form['name4']
        rating5=request.form['name5']
        rating6=request.form['name6']
        rating7=request.form['name7']
        rating8=request.form['name8']
        rating9=request.form['name9']
        rating10=request.form['name10']
        rating11=request.form['name11']
        rating12=request.form['name12']
        rating13=request.form['name13']
        rating14=request.form['name14']
        rating15=request.form['name15']
        
        print(rating1, " " , rating2, " " , rating3, " " , rating4, " " , rating5, " " , rating6, " " , rating7, " " , rating8, " " , rating9, " " , rating10, " " , rating11, " " , rating12, " " , rating13, " " , rating14, " " , rating15, " ")
        db.new_ratings.remove({})

        db.new_ratings.insert_one(
            { "rating" : rating1 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating2 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating3 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating4 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating5 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating6 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating7 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating8 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating9 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating10 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating11 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating12 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating13 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating4 }
        )
        db.new_ratings.insert_one(
            { "rating" : rating15 }
        )

        # print(rrr[0] , ' ' , rrr[1] , ' ' , rrr[2] , ' ' , rrr[3] , ' ' , rrr[4] , ' ' ,  rrr[5] , ' ' , rrr[6] , ' ' , rrr[7] , ' ' , rrr[8] , ' ' , rrr[9] , ' ' , rrr[10] , ' ' , rrr[11] , ' ' , rrr[12] , ' ' ,  rrr[13] , ' ' , rrr[14] , ' ')
        return redirect(url_for('a'))
    ratings_count = db.ratings_compressed.count()
    movies_count = db.movies_compressed.count()
    user_mf_count = db.user_ratings.count()
    user_mf_count
    

    
    if form.validate():
    # Save the comment here.
        flash('Thanks for ratings')
    else:
        flash('Error: All the form fields are required. ')
     
    ratings_count = db.ratings_compressed.count()
    movies_count = db.movies_compressed.count()
    user_mf_count = db.user_ratings.count()

    return render_template('hello.html', form=form , first_15_names = first_15_names)



@app.route("/a", methods=['GET'])
def a():
    global P
    global A
    global B
    global C
    global rrr
    qwerty = pd.DataFrame(list(db.new_ratings.find()))
    qwerty.dropna()
    rrr = qwerty['rating'] 
    print('rrr values:' , rrr[0] , ' ' , rrr[1] , ' ' , rrr[2] , ' ' ,rrr[3] , ' ' , rrr[4] , ' ' ,  rrr[5] , ' ' , rrr[6] , ' ' , rrr[7] , ' ' , rrr[8] , ' ' , rrr[9] , ' ' , rrr[10] , ' ' , rrr[11] , ' ' , rrr[12] , ' ' ,  rrr[13] , ' ' , rrr[14] , ' ')
    
    j = len(A)//2

    for i in range(j):
        if(A[i] <= 1000 and B[i] < 500):
            P[A[i] - 1 , B[i]] = C[i]
    print(111111)
    return redirect(url_for('asubs'))

@app.route("/asubs", methods=['GET'])
def asubs():
    global A
    global B
    global C
    global P     
    j = len(A)//2
    for i in range(j , len(A)):
        if(A[i] <= 1000 and B[i] < 500):
            P[A[i] - 1 , B[i]] = C[i]
    print(222222)
    return redirect(url_for('bij'))    

@app.route("/bij" , methods = ['GET']) 
def bij():  
    global user_mf
    global ratings
    global movies
    global A
    global B
    global C
    global P     
    global rrr 
    global new_ratings
    j = 0
    k = []
    count = 0
    for i in P:
        a = i.sum()
        b = np.count_nonzero(i)
        if(a == 0):
            print(j)
            count += 1
            k.append(j)
        j += 1
        #print(a,b)
    #np.any(np.isnan(P))
    P = np.delete(P , k , 0)
    print('rrr values`1:' , rrr[0] , ' ' , rrr[1] , ' ' , rrr[2] , ' ' , rrr[3] , ' ' ,  rrr[4] , ' ' ,  rrr[5] , ' ' , rrr[6] , ' ' , rrr[7] , ' ' , rrr[8] , ' ' , rrr[9] , ' ' , rrr[10] , ' ' , rrr[11] , ' ' , rrr[12] , ' ' ,  rrr[13] , ' ' , rrr[14] , ' ')

    for i in range(15):
        new_ratings[i] = rrr[i]
        user_mf = user_mf.append({'userId' :  1001 - count , 'movieId' : i , 'rating' : new_ratings[i]} , ignore_index = True)

    return redirect(url_for('cres'))

@app.route("/cres" , methods = ['GET'])
def cres():
    global user_mf
    global ratings
    global movies
    global P
    global P_orig 
    global user_ratings_mean
    global new_ratings  
    P = np.vstack([P, new_ratings])
    # print('P shape after inputs: ' ,  P.shape)
    P_orig = P.copy()
    print(len(P))
    print('P(old)' , P[993])
    user_ratings_mean = np.mean(P, axis = 1)
    # P = P - user_ratings_mean.reshape(-1, 1)
    for i in range(P.shape[0]):
        sum1 = sum(P[i])
        count = np.count_nonzero(P[i])
        for j in range(P.shape[1]):
            if(P[i,j] > 0):
                P[i,j] -= sum1/count
    print('P(new)' ,  P[993])

    return redirect(url_for('dief'))

@app.route("/dief" , methods = ['GET'])
def dief():
    global user_mf
    global ratings
    global movies
    global cosine_sim
    global P
    #cosine_sim = cosine_similarity(P[: , [0,1]] , P[: , [0,1]])
    cosine_sim = cosine_similarity(P , P)
    # print('cosine_sim: ' , cosine_sim.shape)
    # print('cosine sim length' , len(cosine_sim))

    AP = cosine_sim[len(cosine_sim) - 1].argsort()[-16:][::-1]
    print('AP: ' , AP)

    L = np.zeros(500)
    for i in range(1,len(AP)):
        S = P[AP[i]].argsort()[-15:][::-1]
        print(S)
        for j in S:
            L[j] = 1
    T = L.argsort()[-15:][::-1]

    movie_rat = pd.DataFrame(columns=['rating','sim','score'] , index=range(0,500))
    movie_rat['rating'] = 0
    movie_rat['sim'] = 0
    movie_rat['score'] = 0
    # movie_rat['count'] = 0
 
    for i in T:
        for j in range(P.shape[1]):
            movie_rat.loc[j, 'rating'] += cosine_sim[len(cosine_sim) - 1,i]*P[i,j]
            movie_rat.loc[j,'sim'] += cosine_sim[len(cosine_sim) - 1,i]
    movie_rat['score'] = movie_rat['rating']/movie_rat['sim']


    movie_rat = movie_rat.sort_values('score' , ascending = False)

    K = movie_rat.index[:30]
    K_ = []
    for i in range(len(K)):
        if(K[i] > 14):
            K_.append(K[i])
    K_ = K_[:15]

    answer_user_ratings = pd.DataFrame()
    for i in K_:
        answer_user_ratings = answer_user_ratings.append(movies.loc[movies['movieId'] == i])
    # answer_user_ratings
    
    final_user_ratings1 = (answer_user_ratings['title'])
    for i in final_user_ratings1:
        db.user_user.insert_one(
           { 'title' : i }
        )
    # print(final_user_ratings1)
    return redirect(url_for('eqw'))

@app.route("/eqw" , methods = ['GET'])
def eqw():
    global user_mf
    global ratings
    global movies
    global cosine_sim_item
    global rrr
    global P
    Q = np.transpose(P[:len(P) - 1])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim_item = cosine_similarity(Q , Q)
    cosine_sim_item

    for i in range(15):
        for j in range(15):
             cosine_sim_item[i,j] = 0
    cosine_sim_item

    L = []
    for i in range(15):
        if(rrr[i] == 1):
            L.append(cosine_sim_item[i].argsort()[-115:-100][::-1])
        elif(rrr[i] == 2):
            L.append(cosine_sim_item[i].argsort()[-75:-60][::-1])
        elif(rrr[i] == 3):
            L.append(cosine_sim_item[i].argsort()[-45:-30][::-1])
        elif(rrr[i] == 4):
            L.append(cosine_sim_item[i].argsort()[-25:-10][::-1])
        elif(rrr[i] == 5):
            L.append(cosine_sim_item[i].argsort()[-15:][::-1])

    movie_rat_1 = pd.DataFrame(columns=['rating','count','score'] , index=range(0,500))
    movie_rat_1['rating'] = 0
    movie_rat_1['count'] = 0
    movie_rat_1['score'] = 0
    movie_rat_1
    for index,row in movie_rat_1.iterrows():
        p = ratings.loc[ratings['movieId'] == index].shape[0]
        q = ratings.loc[ratings['movieId'] == index].rating.sum()
        movie_rat_1.set_value(index , 'rating' , q)
        movie_rat_1.set_value(index , 'count' , p)   
    movie_rat_1['score'] = movie_rat_1['rating']/movie_rat_1['count']
    movie_rat_1 = movie_rat_1.sort_values(['score' , 'count'] , ascending = [False , False])
    # movie_rat_2 = pd.DataFrame(columns=['rating','count','score'] , index=range(0,500))
    # movie_rat_2['rating'] = 0
    # movie_rat_2['count'] = 0
    # movie_rat_2['score'] = 0

    # for i in L:
    #     for j in range(len(i)):
    #         movie_rat_2.loc[i[j],'rating'] += movie_rat_1.loc[i[j] , 'score']
    #         movie_rat_2.loc[i[j],'count'] += 1
    # movie_rat_2['score'] = movie_rat_2['rating']/movie_rat_2['count']
    # movie_rat_2 = movie_rat_2.sort_values(['count' , 'score'] , ascending = False)

    ans = movie_rat_1.index[:30]
    print(ans)
    ans_ = []
    for i in range(len(ans)):
        if(ans[i] > 14):
            ans_.append(ans[i])
    ans_ = ans_[:15]
    
    answer_item_ratings = pd.DataFrame()
    for i in ans_:
        answer_item_ratings = answer_item_ratings.append(movies.loc[movies['movieId'] == i])
    final_user_ratings2 = (answer_item_ratings['title'])
    # print(final_user_ratings2)
    for i in final_user_ratings2:
        db.item_item.insert_one(
           { 'title' : i }
        )

    return redirect(url_for('faqw'))

@app.route("/faqw" , methods = ['GET'])
def faqw():
    global user_mf
    global ratings
    global movies
    global user_ratings_mean
    global P_orig
    global P
    E = pd.DataFrame(P_orig)

    from scipy.sparse.linalg import svds
    U, sigma, Vt = svds(P, k = 50)

    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = E.columns)

    def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
        
        # Get and sort the user's predictions
        user_row_number = userID - 1 # UserID starts at 1, not 0
    #     sorted_user_predictions = pd.DataFrame(0 , index = range(1) , columns = range(500))
    #     print(sorted_user_predictions)
        sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
        sorted_user_predictions = pd.DataFrame(sorted_user_predictions)
        sorted_user_predictions.index.names = ['movieId']
        
        # Get the user's data and merge in the movie information.
        user_data = original_ratings_df[original_ratings_df.userId == (userID)]
        user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                         sort_values(['rating'], ascending=False)
                     )
        # print(sorted_user_predictions)
        print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
        print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
        
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',left_on = 'movieId',
                   right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1])
    #     print(recommendations)
        return user_full, recommendations

    already_rated,answer_mf_ratings = recommend_movies(preds_df, len(P), movies, user_mf, 15)

    final_user_ratings3 = (answer_mf_ratings['title'])

    for i in final_user_ratings3:
        db.matrix_fact.insert_one(
           { 'title' : i }
        )
    return redirect(url_for('predictions'))

@app.route("/predictions", methods=['GET'])
def predictions():
    global user_mf
    global ratings
    global movies
    X = pd.DataFrame(list(db.user_user.find()))
    if(X.empty == False):
    # #     time.sleep(10)
    # #     return redirect(url_for('predictions'))
        final_user_ratings1 = X['title']
    Y = pd.DataFrame(list(db.item_item.find()))
    if(Y.empty == False):
    # #     time.sleep(10)
    # #     return redirect(url_for('predictions'))
        final_user_ratings2 = Y['title']
    Z = pd.DataFrame(list(db.matrix_fact.find()))
    if(Z.empty == False):
    # #     time.sleep(10)
    # #     return redirect(url_for('predictions'))
        final_user_ratings3 = Z['title']
    
    db.user_user.remove({})
    db.item_item.remove({})
    db.matrix_fact.remove({})   

    return render_template('prediction_fe.html' , result1 = final_user_ratings1 , result2 = final_user_ratings2 , result3 = final_user_ratings3)

    # return render_template('prediction_fe.html' , result1 = final_user_ratings1 , result2 = final_user_ratings2 , result3 = final_user_ratings3)

if __name__ == "__main__":
    app.run()
