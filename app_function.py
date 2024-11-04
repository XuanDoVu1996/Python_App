import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from collections import Counter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import AlgoBase, PredictionImpossible, Dataset, Reader
import math
import pandas as pd
import numpy as np
import heapq
import csv
from collections import defaultdict

#Load the dataset
df = pd.read_csv("data/full_data_cleaned.csv")

#------------------------------------------------------------#
#function to filter out users with fewer than min_predictions
def filter_users_with_enough_predictions(df, min_predictions=2):
    # Count the number of ratings for each user
    user_counts = df['author_id'].value_counts()
    
    # Filter out users with fewer than min_predictions
    users_with_enough_predictions = user_counts[user_counts >= min_predictions].index
    
    # Filter the DataFrame to retain only the relevant users
    filtered_df = df[df['author_id'].isin(users_with_enough_predictions)]
    
    return filtered_df

#------------------------------------------------------------#

#function to vectorize the text data
def vectorize_text(df, col):
    vectorizer = TfidfVectorizer(min_df = 100, max_df = 0.7) #Vectorizing the description
    vectorized_col = vectorizer.fit_transform(df[col])
    vectorized_df = pd.DataFrame(vectorized_col.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)
    #concatenate the vectorized column with the original dataframe
    df = pd.concat([df, vectorized_df], axis=1)
    #drop the original column
    df.drop(col, axis=1, inplace=True)
    return df

#------------------------------------------------------------#
#create function to trim the variable
def trim_func(df, col):
    df[col] = df[col].str.lower() #convert to lower case
    df[col] = df[col].str.replace(r'\d+', '') #remove digits
    df[col] = df[col].str.replace(r'\W', ' ') #remove special characters
    df[col] = df[col].str.replace(r'\s+', ' ') #remove extra spaces
    df[col] = df[col].str.strip() #remove leading and trailing spaces
    return df

#------------------------------------------------------------#
#function to generate star rating
def generate_star_rating(rating, max_rating=5, filled_char="★", empty_char="☆"):
    filled_star = filled_char * int(round(rating))
    partially_filled_star = empty_char * int(max_rating - round(rating))
    return filled_star + partially_filled_star

#------------------------------------------------------------#
#build recommendation system based on products users have used
def user_profile_recommender(user_id, n, df_filtered, products, df):
    # Remove product out of stock:
    df = df[df['out_of_stock'] == 0]
    if len(df_filtered) < 5:
        return html.Div([
            html.P('Sorry, we can not find any recommendations based on your criteria!')
        ]) 
    else:
        #filtered products based on user input
        list = df_filtered.product_id.unique()
        filtered_products = products[products['product_id'].isin(list)]
        
        #User profile - filter products purchased by user
        user_lst = df[df['author_id']== user_id ].product_id.unique()
        user_products = products[products['product_id'].isin(user_lst)].set_index('product_id')
        user_prof = user_products.mean() # average value of the user's products

        #calculate the similarity between the user's products and all other products
        user_prof_similarity = cosine_similarity(user_prof.values.reshape(1, -1), filtered_products.set_index('product_id'))
        #convert the similarity to a dataframe
        user_prof_similarity_df = pd.DataFrame(user_prof_similarity.T, 
                                                index = filtered_products['product_id'],
                                                columns = ['similarity']).sort_values(by = 'similarity', ascending = False)
        #return the top n recommendations
        top_n = user_prof_similarity_df.index[:n]
        recommendations = []
        for product_id in top_n:
            product_info = df_filtered[df_filtered['product_id'] == product_id]
            recommendations.append({
                'product_name': product_info.iloc[0].product_name_x,
                'product_id': product_info.iloc[0].product_id,
                'brand': product_info.iloc[0].brand_name_x,
                'limited_edition': product_info.iloc[0].limited_edition,
                'price': product_info.iloc[0].price_usd_x,
                'rating': product_info.rating_x.mean()
            })
        recommendations = pd.DataFrame(recommendations).sort_values(by='rating', ascending=False).reset_index(drop=True)
        #add dollar sign to price
        recommendations['price'] = recommendations['price'].apply(lambda x: '$' + str(x))
        #change limited edition to yes or no
        recommendations['limited_edition'] = recommendations['limited_edition'].apply(lambda x: 'Yes' if x == 1 else 'No')
        #create empty list of rows
        rows = []
        # assign star rating to each recommendation
        for _,recommendation in recommendations.iterrows():
            recommendation['star_rating'] = generate_star_rating(recommendation['rating'])
            row = html.Tr([
                html.Td(_ + 1),
                html.Td(recommendation['product_name']),
                html.Td(recommendation['brand'], style={'text-align': 'center'}),
                html.Td(recommendation['price'], style={'text-align': 'center'}),
                html.Td(recommendation['star_rating'], style={'text-align': 'center'}),
                html.Td(recommendation['limited_edition'], style={'text-align': 'center'})
            ])
            rows.append(row)
        # Create the table
        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('#',style= {'width': '5%'}),
                    html.Th('Product Name',style= {'width': '15%'}),
                    html.Th('Brand', style= {'width': '5%', 'text-align': 'center'}),
                    html.Th('Price', style= {'width': '5%', 'text-align': 'center'}),
                    html.Th('Rating', style= {'width': '5%', 'text-align': 'center'}),
                    html.Th('Limited Edition', style= {'width': '5%', 'text-align': 'center'})
                ])
            ]),
            html.Tbody(rows)
        ])
        # Return the table
        return html.Div([
                html.H2('Best Products For You!'),
                table
                
            ], style = {
                    'width': '100%',
                    'maxHeight' : '450px',
                    'overflowY' : 'auto',
                    'margin': '20px auto 40px auto',
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'borderRadius': '8px',
                },
                )

#------------------------------------------------------------#
#function to get the top n recommendations for a user (CF Model)
def get_top_n_recommendations(model, user_id, antiset, n=5, dataset = df):
    # Filter based on the first value of each item
    filtered_list = [item for item in antiset if item[0] == user_id]
    predictions = model.test(filtered_list)

    # Convert to df
    predictions_df = pd.DataFrame(predictions)

    # Product info data
    product_info = df[['product_id', 'product_name_x', 'brand_name_x', 'price_usd_x',
                            'limited_edition','niche_product', 'out_of_stock']].drop_duplicates()
    
    # Merge data with product info
    predictions_df = predictions_df.merge(product_info, left_on = ['iid'], right_on = ['product_id'])
    
    # Filter out of stock products
    final_df = predictions_df[predictions_df['out_of_stock'] == 0].copy().sort_values(by='est', ascending=False)

    if len(final_df) > n:
        final_df_subset = final_df.head(n)  # Select the top n rows if the dataset has more than n rows
    else:
        final_df_subset = final_df  # Choose the whole dataset if it has n or fewer rows
    
    # Create a dataframe of top n recommendations with product information
    recommendations = []
    for product_id in final_df_subset['product_id']:
        product_info = df[df['product_id'] == product_id]
        recommendations.append({
            'product_name': product_info.iloc[0].product_name_x,
            'product_id': product_info.iloc[0].product_id,
            'brand': product_info.iloc[0].brand_name_x,
            'limited_edition': product_info.iloc[0].limited_edition,
            'price': product_info.iloc[0].price_usd_x,
            'rating': product_info.rating_x.mean(),
            'niche': product_info.iloc[0].niche_product
        })
    return(pd.DataFrame(recommendations).reset_index(drop=True))
    
def get_top_n_recommendations_final(recommendations):
    #change limited edition to yes or no
    recommendations['limited_edition'] = recommendations['limited_edition'].apply(lambda x: 'Yes' if x == 1 else 'No')
    #add dollar sign to price
    recommendations['price'] = recommendations['price'].apply(lambda x: '$' + str(x))
    #create empty list of rows
    rows = []
    # assign star rating to each recommendation
    for _,recommendation in recommendations.iterrows():
        recommendation['star_rating'] = generate_star_rating(recommendation['rating'])
        row = html.Tr([
            html.Td(_ + 1),
            html.Td(recommendation['product_name']),
            html.Td(recommendation['brand'], style={'text-align': 'center'}),
            html.Td(recommendation['price'], style={'text-align': 'center'}),
            html.Td(recommendation['star_rating'], style={'text-align': 'center'}),
            html.Td(recommendation['limited_edition'], style={'text-align': 'center'})
        ])
        rows.append(row)
    # Create the table
    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('#',style= {'width': '3%'}),
                html.Th('Product Name',style= {'width': '17%'}),
                html.Th('Brand', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Price', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Rating', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Limited Edition', style= {'width': '5%', 'text-align': 'center'})
            ])
        ]),
        html.Tbody(rows)
    ])
    # Return the table
    return html.Div([
        html.H2('Top 5 Recommendations For You ♡'),
        table
        
    ], style = {
            'width': '100%',
            'maxHeight' : '450px',
            'overflowY' : 'auto',
            'margin': '20px auto 40px auto',
            'padding': '20px',
            'border': '1px solid #ccc',
            'borderRadius': '8px',
        },
        )

#------------------------------------------------------------#
# Function to get top n trending products
def get_top_n_trending(n= 10 , dataset=df):

    # Remove product out of stock:
    filtered_data = dataset[dataset['out_of_stock'] == 0]
    
    # Group by 'product_id' and count the number of ratings
    rating_counts = filtered_data.groupby('product_id').size().reset_index(name='rating_count')
    
    # Sort in descending order by 'rating_count'
    rating_counts_sorted = rating_counts.sort_values(by='rating_count', ascending=False)
    
    # Get the top n products with the highest number of ratings
    top_n = rating_counts_sorted.head(n)
    
    # Create a dataframe of top n recommendations with product information
    recommendations = []
    for product_id in top_n['product_id']:
        product_info = dataset[dataset['product_id'] == product_id]
        recommendations.append({
            'product_name': product_info.iloc[0].product_name_x,
            'product_id': product_info.iloc[0].product_id,
            'brand': product_info.iloc[0].brand_name_x,
            'limited_edition': product_info.iloc[0].limited_edition,
            'price': product_info.iloc[0].price_usd_x,
            'rating': product_info.rating_x.mean()
        })
    recommendations = pd.DataFrame(recommendations).sort_values(by='rating', ascending=False).reset_index(drop=True)
    #add dollar sign to price
    recommendations['price'] = recommendations['price'].apply(lambda x: '$' + str(x))
    #change limited edition to yes or no
    recommendations['limited_edition'] = recommendations['limited_edition'].apply(lambda x: 'Yes' if x == 1 else 'No')
    #create empty list of rows
    rows = []
    # assign star rating to each recommendation
    for _,recommendation in recommendations.iterrows():
        recommendation['star_rating'] = generate_star_rating(recommendation['rating'])
        row = html.Tr([
            html.Td(_ + 1),
            html.Td(recommendation['product_name']),
            html.Td(recommendation['brand'], style={'text-align': 'center'}),
            html.Td(recommendation['price'], style={'text-align': 'center'}),
            html.Td(recommendation['star_rating'], style={'text-align': 'center'}),
            html.Td(recommendation['limited_edition'], style={'text-align': 'center'})
        ])
        rows.append(row)
    # Create the table
    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('#',style= {'width': '5%'}),
                html.Th('Product Name',style= {'width': '15%'}),
                html.Th('Brand', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Price', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Rating', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Limited Edition', style= {'width': '5%', 'text-align': 'center'})
            ])
        ]),
        html.Tbody(rows)
    ])
    # Return the table
    return html.Div([
        html.H2('Top Trending Products'),
        table
        
    ], style = {
            'width': '100%',
            'maxHeight' : '450px',
            'overflowY' : 'auto',
            'margin': '20px auto 40px auto',
            'padding': '20px',
            'border': '1px solid #ccc',
            'borderRadius': '8px',
        },
        )

#------------------------------------------------------------#
# Top rated products
def get_top_n_rated(n=10, dataset=df):

    # Remove product out of stock:
    filtered_data = dataset[dataset['out_of_stock'] == 0]
    
    # Group by 'product_id' and calculate average ratings
    rating_average = filtered_data.groupby('product_id')['rating_x'].mean().reset_index(name='average_rating')
    
    # Sort in descending order by 'rating_average'
    rating_average_sorted = rating_average.sort_values(by='average_rating', ascending=False)
    
    # Get the top n products with the highest number of ratings
    top_n = rating_average_sorted.head(n)
    
    # Create a dataframe of top n recommendations with product information
    recommendations = []
    for product_id in top_n['product_id']:
        product_info = dataset[dataset['product_id'] == product_id]
        recommendations.append({
            'product_name': product_info.iloc[0].product_name_x,
            'product_id': product_info.iloc[0].product_id,
            'brand': product_info.iloc[0].brand_name_x,
            'limited_edition': product_info.iloc[0].limited_edition,
            'price': product_info.iloc[0].price_usd_x,
            'rating': product_info.rating_x.mean()
        })
    recommendations = pd.DataFrame(recommendations).sort_values(by='rating', ascending=False).reset_index(drop=True)
    #add dollar sign to price
    recommendations['price'] = recommendations['price'].apply(lambda x: '$' + str(x))
    #change limited edition to yes or no
    recommendations['limited_edition'] = recommendations['limited_edition'].apply(lambda x: 'Yes' if x == 1 else 'No')
    #create empty list of rows
    rows = []
    # assign star rating to each recommendation
    for _,recommendation in recommendations.iterrows():
        recommendation['star_rating'] = generate_star_rating(recommendation['rating'])
        row = html.Tr([
            html.Td(_ + 1),
            html.Td(recommendation['product_name']),
            html.Td(recommendation['brand'], style={'text-align': 'center'}),
            html.Td(recommendation['price'], style={'text-align': 'center'}),
            html.Td(recommendation['star_rating'], style={'text-align': 'center'}),
            html.Td(recommendation['limited_edition'], style={'text-align': 'center'})
        ])
        rows.append(row)
    # Create the table
    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('#',style= {'width': '5%'}),
                html.Th('Product Name',style= {'width': '15%'}),
                html.Th('Brand', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Price', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Rating', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Limited Edition', style= {'width': '5%', 'text-align': 'center'})
            ])
        ]),
        html.Tbody(rows)
    ])
    # Return the table
    return html.Div([
        html.H2('Top Rated Products'),
        table
        
    ], style = {
            'width': '100%',
            'maxHeight' : '450px',
            'overflowY' : 'auto',
            'margin': '20px auto 40px auto',
            'padding': '20px',
            'border': '1px solid #ccc',
            'borderRadius': '8px',
        },
        )
#------------------------------------------------------------#
#function to get similar products based on products user added to cart
def get_similar_products(product, n, df, products):
    # Remove product out of stock:
    df = df[df['out_of_stock'] == 0]
    #filter products based on user input
    user_products = products[products['product_id'].isin(product)].set_index('product_id')
    user_prof = user_products.mean()

    #filter products purchased by other user
    other_user_lst = df[~df['product_id'].isin(product)].product_id.unique()
    other_user_prof = products[products['product_id'].isin(other_user_lst)]

    #calculate the similarity between the user's products and all other products
    similarity = cosine_similarity(user_prof.values.reshape(1, -1), other_user_prof.set_index('product_id'))
    #convert the similarity to a dataframe
    similarity = pd.DataFrame(similarity.T, index = other_user_prof['product_id'],
                                            columns = ['similarity']).sort_values(by = 'similarity', ascending = False)
    #return the top n recommendations
    top_n = similarity.index[:n]
    recommendations = []
    for product_id in top_n:
        product_info = df[df['product_id'] == product_id]
        recommendations.append({
            'product_name': product_info.iloc[0].product_name_x,
            'product_id': product_info.iloc[0].product_id,
            'brand': product_info.iloc[0].brand_name_x,
            'limited_edition': product_info.iloc[0].limited_edition,
            'price': product_info.iloc[0].price_usd_x,
            'rating': product_info.rating_x.mean()
        })
    recommendations = pd.DataFrame(recommendations).sort_values(by='rating', ascending=False).reset_index(drop=True)
    #add dollar sign to price
    recommendations['price'] = recommendations['price'].apply(lambda x: '$' + str(x))
    #change limited edition to yes or no
    recommendations['limited_edition'] = recommendations['limited_edition'].apply(lambda x: 'Yes' if x == 1 else 'No')
    #create empty list of rows
    rows = []
    # assign star rating to each recommendation
    for _,recommendation in recommendations.iterrows():
        recommendation['star_rating'] = generate_star_rating(recommendation['rating'])
        row = html.Tr([
            html.Td(_ + 1),
            html.Td(recommendation['product_name']),
            html.Td(recommendation['brand'], style={'text-align': 'center'}),
            html.Td(recommendation['price'], style={'text-align': 'center'}),
            html.Td(recommendation['star_rating'], style={'text-align': 'center'}),
            html.Td(recommendation['limited_edition'], style={'text-align': 'center'})
        ])
        rows.append(row)
    # Create the table
    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('#',style= {'width': '5%'}),
                html.Th('Product Name',style= {'width': '15%'}),
                html.Th('Brand', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Price', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Rating', style= {'width': '5%', 'text-align': 'center'}),
                html.Th('Limited Edition', style= {'width': '5%', 'text-align': 'center'})
            ])
        ]),
        html.Tbody(rows)
    ])
    # Return the table
    return html.Div([
            html.H2('Other Customers Bought'),
            table
            
        ], style = {
                'width': '100%',
                'maxHeight' : '450px',
                'overflowY' : 'auto',
                'margin': '20px auto 40px auto',
                'padding': '20px',
                'border': '1px solid #ccc',
                'borderRadius': '8px',
            },
            )

