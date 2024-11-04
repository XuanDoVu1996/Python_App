
# Python Project _ Interactive Recommendation System App _ Skincare Products

## :information_source: GENERAL INFORMATION
- :bulb: Purpose: launching a skincare-product review application to tap into the lucrative skincare market

- :file_folder: List of files:
  + data sources 
  + data cleaning and visualization
  + recommedation system models
  + python application

## :one: Methodology:

The methodoly for each recommender is as below:

**:arrow_right: Recommender 1**
- The first recommender, a shallow recommender using a simple filtering technique, is built to
provide users with general non-personalized recommendations, including top trending and top
rated products. These products will be presented on the front page of our platform so that all
users visiting our platform can view and refer to top trending and high-rated products weekly,
daily, or even in real time based on how often we update the underlying dataset.

**:arrow_right: Recommender 2**
- The second recommender, a personalized recommender, is built using content-based filtering,
with the purpose of suggesting relevant products for new users and for users who made less
than 3 ratings. For new users, who we have no data about their purchasing pattern or
preference and hence CF recommenders face extreme cold-start problems, we can either ask for their inputs directly, or use their browsing history (ie., searched items, viewed items) to
suggest related products. For users who made less than 3 ratings and thus CF systems work less
effectively, we can apply the same approaches as new users, or use purchased items to suggest
similar items

**:arrow_right: Recommender 3**
- The third recommender aims to suggest relevant products for users who have made at least 3
ratings, leveraging their purchasing and evaluating history. This personalized recommender is
chosen by evaluating and comparing various CF algorithms under the Surprise package, both
model-based and memory-based methods, as well as CBF.

<P style="page-break-before: always">

## :two: App Demo

The application includes four main components, each offering different
functionalities tailored to enhance the user experience. 

**:arrow_right: Function 1**
The first segment focuses on providing
recommendations for top rated products and top trending products. Top rated products are
suggested based on overall rating of products, ensuring users have access to high-quality items
with the highest rating. On the other hand, top-trending products are determined based on the
number of reviews products received, indicating their popularity among customers. 

<img src="images/recommender 1.jpg" alt="recommender 1" width="1200"/>

**:arrow_right: Function 2**

The second section of the web application offers users personalized recommendations for the
top five products, leveraging their past purchase history and review data. The personalized
recommendations are chosen based on the model 3. The model uses KNN-Basic algorithm
which predicts user’s preferences with highest accuracy, making it our top-performing model
among others.

<img src="images/recommender 3.jpg" alt="recommender 3" width="1200"/>

**:arrow_right: Function 3**

In the third part of the web application, users are allowed to refine their product selections
based on their preferred criteria by applying filters based on the user's characteristics such as
hair color, eye color, skin tone, skin type. Additionally, they can adjust other criteria such as
price range and niche products to align the search with their specific needs. This customization
ensures that users can find products that perfectly suit their individual preferences.

<img src="images/recommender 2_1.jpg" alt="recommender 2_1" width="1200"/>

<img src="images/recommender 2_2.jpg" alt="recommender 2_2" width="1200"/>

After users complete their journey of searching for relevant products and receiving
recommendations, the final segment of the web application encourages them to add their
favorite products to the cart. As users add products to their cart, we continue to suggest other
relevant products that other customers have bought. This enhances the user experience by
providing additional options they might also like. The final part, driven by CBF algorithm, aims to
further increase user engagement, and offer recommendations for products users may not have
previously considered.

<img src="images/recommender 2_3.jpg" alt="recommender 2_3" width="1200"/>




