from surprise import AlgoBase, PredictionImpossible, Dataset, Reader
import math
import numpy as np
import heapq
import csv
from collections import defaultdict

class ContentBasedFiltering(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
 
        data_loader = DataLoader()
        products = data_loader.get_products()
        
        print("Computing Product Similarity Matrix...")
            
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        for current_rating in range(self.trainset.n_items):
            for other_rating in range(current_rating+1, self.trainset.n_items):
                current_product_id = self.trainset.to_raw_iid(current_rating)
                other_product_id = self.trainset.to_raw_iid(other_rating)
                product_similarity = self.compute_product_similarity(current_product_id, other_product_id, products)
                self.similarities[current_rating, other_rating] = product_similarity 
                self.similarities[other_rating, current_rating] = self.similarities[current_rating, other_rating] 
 
        return self
    
    def compute_product_similarity(self, product1, product2, products):
        character1 = products[product1]
        character2 = products[product2]
        sum_of_squares_x, sum_of_products, sum_of_squares_y = 0, 0, 0
        for i in range(len(character1)):
            x = float(character1[i])
            y = float(character2[i])
            sum_of_squares_x += x * x
            sum_of_squares_y += y * y
            sum_of_products += x * y    
        return sum_of_products/math.sqrt(sum_of_squares_x*sum_of_squares_y)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User & item is not found.')
        
        neighbors = []
        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities[i,rating[0]]
            neighbors.append( (genre_similarity, rating[1]) )
        
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        total_similarity = weighted_sum = 0
        for (sim_score, rating) in k_neighbors:
            if (sim_score > 0):
                total_similarity += sim_score
                weighted_sum += sim_score * rating
            
        if (total_similarity == 0):
            raise PredictionImpossible('No neighbor found.')

        predicted_rating = weighted_sum / total_similarity

        return predicted_rating
    

class DataLoader:

    #product_path = 'products.csv'
    
    def get_products(self):
        products = defaultdict(list)
        with open('products.csv', newline='', encoding='ISO-8859-1') as csvfile:
            product_reader = csv.reader(csvfile)
            next(product_reader)  #Skip header line
            for row in product_reader:
                product_id = row[0]
                product_id_list = row[1:]
                products[product_id] = product_id_list          
        return products