# Movie-Recommendation

This application is divided in to four parts: 
Static contains the css code used , Templates contains the html pages , a python file containing all the code for the project and a database file containing all the json and bson files of the mongodb database.

In this project I 've used all the three types of collaborative filtering algorithms :

1.User- User Collaborative Filtering : 
-
  This algorithm finds the similarity between users who have rated similarly to a common movie and based on that extent of       similarity and rating given to the movie given by those users , the top movies are recommended to the user.
  
2.Item-Item Collaborative Filtering:
-
  This algorithm finds the similarity between movies and based on the similarity score obtained and the frequency of occurence of those movies in the K movies selected the best recommendations are given to the user.
  
3.Matrix Factorization:
-
This algorithm uses the function svd (single value decomposition) to factorize the matrix (say X) into two matrices (A , B)    with only few filled values to approximately guess how will user rate the movies he/she hasn't seen and recommends the top    movies therein.
  
The value of K that I chose was 15 because :
  The dataset that I chose approximately had 15-20 genres present in total and since every movie has 2-3 genres associated     with it so 10 would have been a somewhat good value of K but since I wanted to be on the safe side I chose K to be 15. Also   K could have been larger than 15 but since the movies dataset has only 500 movies I didn't want to overfit the dataset as     some genres would be repeting in most of them and might outweigh those present in less amount.

Database Files:
The mongoDB database used named movie_recommendations contains three collections named ratings_compressed , movies_compressed and user_ratings and the corresponding collections have been converted to bson and json files using mongodump and added to movie_recommendations folder.

Code:
The code that I used is contained in the python file mongo_and_collaborative_filtering.py file and the libraries I used with their description is below:
flask - Used to provide the backend of the server connecting python and html pages for the app to run.
wtforms - Helps to use the data from HTML in an easy manner.
numpy - It is used for solving the complex calculations easily with the help of arrays and matrices. Especially cosine similarity in the project is performed using numpy matrices.
pandas - It is used to derive data from mongoDB and use it for further analysis of data in an easy manner.
pymongo - It is used to connect to the mongo database present at the mongo atlas server.
