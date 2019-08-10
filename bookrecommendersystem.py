#Code for Book Recommendation System

# name : bookrecommendersystem.py
# input : user id
# output : isbn ad title
#    1) Content based filtering recommendations top 10
#    2) User based collaborative filtering recommendations top 10
#    3) Item based collaborative filtering recommendations top 10
#    4) Trending books recommendations top 10
#    5) Deep learning model based recommendations top10
#    6) Hybrid recommendation. Combine recommendations from all above approaches

import pickle
import nltk
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from keras.models import load_model
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.WARNING)
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

user_books_data = pickle.load(open('user_books_sorted.pkl','rb'))
some_valid_users = [4213258,4622890,597461,5253785,269235,45618,405390,614778,416390,17438949,1413439,3978225,68030,4159922,1713956,16254355,16731747,48483884,279256,1742824,3452652,2457095,5903843,8734459,8655084,14596170,10171516,1237196,2294090,3427339]


def is_valid_user_id(userid):
	if userid not in user_books_data.keys():
		print("Not valid user id. Some sample user ids : ")
		print(some_valid_users)
		sys.exit(0)

#get isbn from recommended ratings
def top10isbn(recommendations):
    top10_isbn = []
    for item in recommendations:
        item_isbn = item[1]
        top10_isbn.append(item_isbn)
    return top10_isbn

def contentbasedtop10(user,ratings_matrix,isbn_title_dict,all_isbn_list):
	allbooks = ratings_matrix.index
	allratings = ratings_matrix[user]
	ratings_ranking = []
	for i in range(len(allbooks)):
		ratingslist = []
		try:
			ratingslist.append(allratings[i])
			ratingslist.append(allbooks[i])
			ratings_ranking.append(ratingslist)
		except:
			ratingslist.append(int(0))
			ratingslist.append(allbooks[i])
			ratings_ranking.append(ratingslist)

	ratings_ranking = [item for item in ratings_ranking if item[1] in all_isbn_list]
	ratings_ranking.sort(key=lambda x: x[0])
	ratings_ranking.reverse()
	top10_ratings = ratings_ranking[:10]
	return top10_ratings

def get_content_based_recommendations(userid):
	ratings_matrix = pd.read_pickle('content_based_rating_matrix.pkl')
	isbn_title_dict = pickle.load(open("isbn_title_dict.pkl", "rb"))
	all_isbn_list = pickle.load(open("all_isbn_list.pkl", "rb"))
	recommendations = contentbasedtop10(userid,ratings_matrix,isbn_title_dict,all_isbn_list)
	return recommendations,isbn_title_dict

#Calculate pairwise distance similarity
def closest_users(user, ratings, n=10):
    model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model.fit(ratings)
    index = ratings.index.get_loc(user)
    distances, indices = model.kneighbors(ratings.iloc[index, :].values.reshape(1, -1), n_neighbors = n+1)
    distances = 1-distances.flatten()
    return distances,indices

#User based similarity
def predict_ratings_user(user, item, ratings, n=10):
    total=0
    distances, indices= closest_users(user, ratings, n)
    rmean = ratings.iloc[ratings.index.get_loc(user),:].mean()
    for x in range(0, len(indices.flatten())):
        if indices.flatten()[x] != ratings.columns.get_loc(item):
            diff = ratings.iloc[indices.flatten()[x],ratings.columns.get_loc(item)]-np.mean(ratings.iloc[indices.flatten()[x],:])
            delta = diff * (distances[x])
            total = total + delta
    return int(round(rmean + (total/(np.sum(distances)-1))))


# Get top user baed collborative filtering recommendations for user
def ubcftop10(user,ratings_matrix,isbn_title_dict,all_isbn_list):
	ratings_rankings = []
	for i in range(ratings_matrix.shape[1]):
		ratinglist = []
		if ratings_matrix[str(ratings_matrix.columns[i])][user] == 0:
			ratinglist.append(int(-1))
			ratinglist.append(str(ratings_matrix.columns[i]))
		else:
			ratinglist.append(int((predict_ratings_user(user, str(ratings_matrix.columns[i]), ratings_matrix))))
			ratinglist.append(str(ratings_matrix.columns[i]))
		ratings_rankings.append(ratinglist)

	ratings_rankings = [item for item in ratings_rankings if item[1] in all_isbn_list]
	ratings_rankings.sort(key=lambda x: x[0])
	ratings_rankings.reverse()
	ratings = ratings_rankings[:20]
	random.shuffle(ratings)
	top10_ratings = ratings[:10]
	return top10_ratings

def get_user_based_cf_recommendations(userid):
	ratings_matrix = pd.read_pickle('collaborative_ratings_matrix.pkl')
	isbn_title_dict = pickle.load(open("isbn_title_dict.pkl", "rb"))
	all_isbn_list = pickle.load(open("all_isbn_list.pkl", "rb"))
	recommendations = ubcftop10(userid,ratings_matrix,isbn_title_dict,all_isbn_list)
	return recommendations,isbn_title_dict


#Item based
def closest_items(item, ratings, n=10):
    ratings=ratings.T
    index = ratings.index.get_loc(item)
    model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model.fit(ratings)
    distances, indices = model.kneighbors(ratings.iloc[index, :].values.reshape(1, -1), n_neighbors = n+1)
    distances = 1-distances.flatten()
    return distances,indices


# predict ratings
def predict_ratings_item(user, item, ratings, n=10):
	total = 0
	distances, indices = closest_items(item, ratings)
	for x in range(0, len(indices.flatten())):
		if indices.flatten()[x] != ratings.columns.get_loc(item):
			factor = ratings.iloc[ratings.index.get_loc(user), indices.flatten()[x]] * (distances[x])
			total = total + factor
	try:
		item_rating = int(round(total / (np.sum(distances) - 1)))
	except:
		item_rating = 0

	return item_rating

# Get top item baed collborative filtering recommendations for user
def ibcftop10(user,ratings_matrix,isbn_title_dict,all_isbn_list):
	ratings_rankings = []
	for i in range(ratings_matrix.shape[1]):
		ratinglist = []
		if ratings_matrix[str(ratings_matrix.columns[i])][user] == 0:
			ratinglist.append(int(-1))
			ratinglist.append(str(ratings_matrix.columns[i]))
		else:
			ratinglist.append(int((predict_ratings_item(user, str(ratings_matrix.columns[i]), ratings_matrix))))
			ratinglist.append(str(ratings_matrix.columns[i]))
		ratings_rankings.append(ratinglist)

	ratings_rankings = [item for item in ratings_rankings if item[1] in all_isbn_list]
	ratings_rankings.sort(key=lambda x: x[0])
	ratings_rankings.reverse()
	ratings = ratings_rankings[:20]
	random.shuffle(ratings)
	top10_ratings = ratings[:10]
	return top10_ratings

def get_item_based_cf_recommendations(userid):
	ratings_matrix = pd.read_pickle('collaborative_ratings_matrix.pkl')
	isbn_title_dict = pickle.load(open("isbn_title_dict.pkl", "rb"))
	all_isbn_list = pickle.load(open("all_isbn_list.pkl", "rb"))
	recommendations = ibcftop10(userid,ratings_matrix,isbn_title_dict,all_isbn_list)
	return recommendations,isbn_title_dict


def top10deep(user,model,user_dict,isbn_title_dict,usersbooks,isbns):
	bookratings = []
	for book in isbns:
		ud = user_dict[user]
		if user not in usersbooks.keys():
			continue
		x = pd.Series([ud])
		y = pd.Series([book])
		predictions = model.predict([x, y])
		bookratings.append([predictions[0][0], isbns[book]])

	bookratings = [item for item in bookratings if item[1] in isbn_title_dict.keys()]
	bookratings = [item for item in bookratings if item[1] not in usersbooks[user]]
	bookratings.sort(key=lambda x: x[0], reverse=True)
	return bookratings[:10]

def get_deep_learning_recommendations(userid):
	reviews_dataset = pd.read_pickle('reviews.pkl')
	reviews_dataset.drop(columns=['user_name', 'user_review_text'], inplace=True)
	reviews_dataset.dropna(inplace=True)
	uniq_isbn = reviews_dataset.isbn.unique()
	isbn_dict = {uniq_isbn[i]: i + 1 for i in range(0, len(uniq_isbn))}
	uniq_user = reviews_dataset.user_id.unique()
	user_dict = {uniq_user[i]: i + 1 for i in range(0, len(uniq_user))}
	model = load_model('deeplenarningmodel')
	isbn_title_dict = pickle.load(open("isbn_title_dict.pkl", "rb"))
	usersbooks = pickle.load(open("user_books_sorted.pkl", "rb"))
	isbns = dict((v, k) for k, v in isbn_dict.items())
	deeplrecommended = top10deep(userid,model,user_dict,isbn_title_dict,usersbooks,isbns)
	return deeplrecommended,isbn_title_dict

def trending_similar(isbn,similarity,isbns):
    bookid = isbns[isbn]
    bookslist = list(enumerate(similarity[bookid]))
    bookslist.sort(key=lambda x: x[1], reverse=True)
    top10TrendingBooks = bookslist[1:11]
    return top10TrendingBooks

def trendingTop10(user,usersbooks,isbnlookup,similarity,isbns,combined_isbn_title_dict):
    booklist = usersbooks[user]
    ratingsbooks = []
    recommendedbooks = []
    for book in booklist:
        newbooks = trending_similar(book,similarity,isbns)
        ratingsbooks = ratingsbooks + newbooks
    ratingsbooks = list(set(ratingsbooks))
    for item in ratingsbooks:
        try :
            bookisbn = isbnlookup[item[0]]
            recommendedbooks.append([item[1],bookisbn])
        except:
            continue
    recommendedbooks = [item for item in recommendedbooks if item[1] in combined_isbn_title_dict.keys()]
    recommendedbooks.sort(key=lambda x: x[1], reverse=True)
    return recommendedbooks[:10]

def get_trending_recommendations(userid):
	combineddata = pd.read_pickle('booksdata_trendingdata_combined.pkl')
	combineddata.reset_index(drop=True, inplace=True)
	tfidfv = TfidfVectorizer(analyzer='word', stop_words=set(stopwords.words('english')))
	vocab = tfidfv.fit_transform(combineddata['description'])
	similarity = linear_kernel(vocab, vocab)
	isbns = pd.Series(combineddata['isbn']).to_dict()
	isbns = dict((v, k) for k, v in isbns.items())
	usersbooks = pickle.load(open("user_books_sorted.pkl", "rb"))
	combined_isbn_title_dict = pickle.load(open("combined_isbn_title_dict.pkl", "rb"))
	isbnlookup = dict((v, k) for k, v in isbns.items())
	trendingrecommended = trendingTop10(userid,usersbooks,isbnlookup,similarity,isbns,combined_isbn_title_dict)
	return trendingrecommended,combined_isbn_title_dict

def recommend_books():
	print("Welcome to Book Recommendation System :")
	userid = str(input("Enter user id : "))
	print("Entered userid is : ", userid)
	is_valid_user_id(str(userid))

	#Content based filtering
	content_results = []
	print("\nCalculating content based filtering books recommendations\n")
	recommendations_cb,isbn_title_dict_cb = get_content_based_recommendations(str(userid))
	isbn_recommendations_cb = top10isbn(recommendations_cb)
	print("Content based filtering recommendations : ")
	for isbnitem in isbn_recommendations_cb:
		print("ISBN :", str(isbnitem), "Title :", str(isbn_title_dict_cb[isbnitem]))
		content_results.append([str(isbnitem), str(isbn_title_dict_cb[isbnitem])])

	#User based collaborative filteringub_results = []
	print("\nCalculating user based collaborative filtering books recommendations\n")
	ub_results =[]
	recommendations_ub,isbn_title_dict_ub =  get_user_based_cf_recommendations(str(userid))
	isbn_recommendations_ub = top10isbn(recommendations_ub)
	print("User based collaborative filtering recommendations : ")
	for isbnitem in isbn_recommendations_ub:
		print("ISBN :", str(isbnitem), "Title :", str(isbn_title_dict_ub[isbnitem]))
		ub_results.append([str(isbnitem),str(isbn_title_dict_ub[isbnitem])])

	#Item based collaborative filtering
	print("\nCalculating item based collaborative filtering books recommendations\n")
	ib_results =[]
	recommendations_ib,isbn_title_dict_ib = get_item_based_cf_recommendations(str(userid))
	isbn_recommendations_ib = top10isbn(recommendations_ib)
	print("Item based collaborative filtering recommendations : ")
	for isbnitem in isbn_recommendations_ib:
		print("ISBN :", str(isbnitem), "Title :", str(isbn_title_dict_ib[isbnitem]))
		ib_results.append([str(isbnitem),str(isbn_title_dict_ib[isbnitem])])

	print("\nCalculating trending books recommendations\n")
	tr_results =[]
	trendingrecommended,combined_isbn_title_dict = get_trending_recommendations(str(userid))
	isbn_recommendations_tr = top10isbn(trendingrecommended)
	print("Recommendations based on trending books : ")
	for isbnitem in isbn_recommendations_tr:
		print("ISBN :", str(isbnitem), "Title :", str(combined_isbn_title_dict[isbnitem]))
		tr_results.append([str(isbnitem),str(combined_isbn_title_dict[isbnitem])])

	#Deep learning
	print("\nCalculating deep learning model based books recommendations\n")
	dl_results =[]
	deeplrecommended,isbn_title_dict_dl = get_deep_learning_recommendations(str(userid))
	isbn_recommendations_dl = top10isbn(deeplrecommended)
	print("Deep learning based model recommendations books : ")
	for isbnitem in isbn_recommendations_dl:
		print("ISBN :", str(isbnitem), "Title :", str(isbn_title_dict_dl[isbnitem]))
		dl_results.append([str(isbnitem),str(isbn_title_dict_dl[isbnitem])])

	# Deep learning
	print("\nHybrid approach based books recommendations\n")
	hybrid_recommendation = []
	if len(content_results) >= 2:
		hybrid_recommendation.append(content_results[:2])
	if len(ub_results) >= 2:
		hybrid_recommendation.append(ub_results[:2])
	if len(ib_results) >= 2:
		hybrid_recommendation.append(ib_results[:2])
	if len(tr_results) >= 2:
		hybrid_recommendation.append(tr_results[:2])
	if len(dl_results) >= 2:
		hybrid_recommendation.append(dl_results[:2])

	for bookdetails in hybrid_recommendation:
		for item in bookdetails:
			print("ISBN :", str(item[0]), "Title :", str(item[1]))

if __name__== "__main__":
	recommend_books()

