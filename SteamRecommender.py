# Databricks notebook source
# MAGIC %md
# MAGIC # INTRODUCTION
# MAGIC In this task, I’m building a recommender system using the Steam dataset in Databricks. The dataset shows how users interact with video games, including their behaviour and how long they spend playing. Since there are no direct ratings, this data is treated as implicit feedback, meaning I have to infer user preferences based on what they do rather than what they explicitly rate.
# MAGIC
# MAGIC The main aim is to use the Alternating Least Squares (ALS) algorithm to create a collaborative filtering model. To do this, I go through the full process of loading the data, exploring it, preparing it for modelling, training the model, and then generating recommendations. Spark is used because it can handle large datasets efficiently, and MLflow is included to track experiments and compare model performance, making it easier to stay organised.

# COMMAND ----------

# MAGIC %md
# MAGIC # LOADING UP THE DATASET
# MAGIC I started by loading the Steam dataset from a CSV file into a Spark DataFrame. I used spark.read.csv() and set header=False because the file does not contain column names in the first row. I also enabled inferSchema=True so that Spark automatically detects the correct data types for each column instead of treating everything as strings. After loading the data, I renamed the columns using toDF() to make them more meaningful and easier to work with. I assigned the names user_id, game, behavior, and value, which represent the user identifier, the game name, the type of interaction (such as purchase or play), and the associated value (such as hours played). Finally, I used df.show() to display a sample of the DataFrame in Databricks so I can quickly verify that the data has been loaded and structured correctly.

# COMMAND ----------

from pyspark.sql.functions import *

# Load CSV
df = spark.read.csv(
    "/Volumes/teaching/datasets/assignment_2/task_2/steam-200k.csv",
    header=False,
    inferSchema=True
)

# Rename columns
df = df.toDF("user_id", "game", "behavior", "value")

df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # BASIC EXPLORATION of DATA (EDA)
# MAGIC ## VISUALISING THE TOP GAMES BY PLAYTIME
# MAGIC My first step was to explore user behaviour and game popularity within the dataset. I started by grouping the data by the behavior column and using count() to see how many records fall under each category, such as “play” or “purchase”. I then displayed the results using .show() to get a quick overview of how users interact with games. Next, I focused on identifying the most played games. I filtered the DataFrame to include only rows where the behavior is “play”, since I’m interested in gameplay activity rather than purchases. I then grouped the data by game and used the sum function on the value column to calculate the total number of hours played for each game. After that, I sorted the results in descending order based on total hours played so the most popular games appear first, and limited the output to the top 10. I displayed this result using display(top_games) so it’s easier to interpret in Databricks. To visualise the most played games, I created a bar chart using the top_games DataFrame. After running display(top_games), I switched to the chart view in Databricks and selected a bar chart. I set game as the y-axis so that each bar represents a different game, and I used total_hours as the x-axis to show the total playtime. I also renamed the x-axis label to Total Hours Played to make the chart clearer and more descriptive. This visualisation makes it easy to compare how much time users have spent on each of the top games at a glance. Finally, I calculated the number of unique users and unique games in the dataset. I did this by selecting the user_id and game columns separately, applying distinct() to remove duplicates, and then using count() to get the total number of unique values in each case. I printed these results to provide a quick summary of the dataset’s scale.

# COMMAND ----------

df.groupBy("behavior").count().show()

# Top played games
top_games = (
    df.filter(col("behavior") == "play") \
  .groupBy("game") \
  .agg(sum("value").alias("total_hours")) \
  .orderBy(desc("total_hours")) \
  .limit(10)
)

display(top_games)

# Number of unique users & games
print("Unique users:", df.select("user_id").distinct().count())
print("Unique games:", df.select("game").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## VISUALISING THE TOP 10 MOST ACTIVE USERS BY PLAYTIME
# MAGIC The next step I took was to visualise the top 10 most active users based on total playtime. I started by filtering the dataset to include only rows where the behavior is “play”, since I’m interested in actual gameplay rather than purchases. I then grouped the data by user_id and calculated the total number of hours played for each user using the sum function on the value column. After that, I sorted the results in descending order and limited it to the top 10 users with the highest playtime. To create the visualisation, I converted the Spark DataFrame into a Pandas DataFrame using toPandas() so that I could use Matplotlib for plotting. I also sorted the data in ascending order of total hours so that the bar chart appears neatly from lowest to highest. I then created a horizontal bar chart using Matplotlib, where each bar represents a user and its length corresponds to their total hours played. I applied a blue colour gradient using matplotlib.cm to make the chart more visually appealing and to subtly reflect differences in playtime. To improve readability, I added labels at the end of each bar showing the exact number of hours played for each user. Finally, I customised the chart by setting clear axis labels (Total Hours Played for the x-axis and User ID for the y-axis), adding a bold title, and slightly extending the x-axis limit to ensure the labels don’t overlap with the edges. I used plt.tight_layout() to improve spacing and plt.show() to display the final chart.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

top_users = (
    df.filter(col("behavior") == "play")
      .groupBy("user_id")
      .agg(sum("value").alias("total_hours"))
      .orderBy(desc("total_hours"))
      .limit(10)
)

pdf = top_users.toPandas().sort_values("total_hours")

# Color gradient based on hours
colors = cm.Blues(np.linspace(0.4, 0.9, len(pdf)))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(pdf["user_id"].astype(str), pdf["total_hours"], color=colors, edgecolor="white")

# Add value labels on each bar
for bar in bars:
    width = bar.get_width()
    ax.text(width + pdf["total_hours"].max()*0.01, bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f} hrs", va="center", fontsize=10)

ax.set_xlabel("Total Hours Played", fontsize=12)
ax.set_ylabel("User ID", fontsize=12)
ax.set_title("Top 10 Most Active Users", fontsize=14, fontweight="bold")
ax.set_xlim(0, pdf["total_hours"].max() * 1.12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # PREPARING DATA FOR MACHINE LEARNING
# MAGIC ## FILTERING THE DATA
# MAGIC The first step I took was to isolate the play behaviour into a new dataframe to use in training my recommender. I did this because there is no explicit feedback from the users in terms of their purchases other than the fact that they actually purchased the game. It was much easier to gather implicit feedback from the amount of play hours they accrued playing the games.

# COMMAND ----------

df_play = df.filter(col("behavior") == "play")

# COMMAND ----------

# MAGIC %md
# MAGIC ## USING STRINGINDEXER TO AUTOMATICALLY ASSIGN A UNIQUE NUMERIC ID TO EACH GAME
# MAGIC In this step, I prepared the data for machine learning by converting categorical variables into numerical indices. Since algorithms like ALS require numeric inputs, I used StringIndexer to transform both the game and user_id columns into numerical representations. I first created a StringIndexer for the game column, which converts each unique game into a corresponding numeric ID stored in a new column called game_id. I set handleInvalid="skip" to ensure that any unexpected or null values are ignored rather than causing errors during processing. I then created another StringIndexer for the user_id column, which maps each user to a unique numeric index stored in a new column called user_index, again handling invalid values by skipping them. After defining both indexers, I fitted and applied them sequentially to the dataset. I first transformed the data using the game indexer, and then applied the user indexer on the resulting DataFrame. The final output, df_indexed, now contains numerical representations of both users and games, making it suitable for use in machine learning models like ALS.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

game_indexer = StringIndexer(inputCol="game", outputCol="game_id", handleInvalid="skip")
user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid="skip")

df_indexed = game_indexer.fit(df_play).transform(df_play)
df_indexed = user_indexer.fit(df_indexed).transform(df_indexed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## PREPARING THE FINAL DATASET FOR THE RECOMMENDATION MODEL BY RENAMING RELEVANT COLUMNS
# MAGIC The final thing I did to prepare my data was to select and rename the relevant columns. I extracted the user_index, game_id, and value columns from the indexed DataFrame. Since machine learning models like ALS require integer IDs, I explicitly cast both user_index and game_id to integers and renamed them to user and item respectively. I also renamed the value column to rating, as it represents the number of hours played and will be used as the implicit rating in the model. The resulting DataFrame, final_df, now has the correct structure (user, item, rating) required for training a recommendation system.

# COMMAND ----------

final_df = df_indexed.select(
    col("user_index").cast("int").alias("user"),
    col("game_id").cast("int").alias("item"),
    col("value").alias("rating")   # hours played
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SPLITTING MY MODEL INTO TRAINING AND TESTING DATA TO EVALUATE THE PERFORMANCE OF MY RECOMMENDATION MODEL
# MAGIC My next step was to split the dataset into training and test sets so I can evaluate the performance of my recommendation model. I used the randomSplit() function to divide the data into 80% for training and 20% for testing. I also set a seed value of 42 to ensure that the split is reproducible, meaning I’ll get the same results each time I run the code. After splitting the data, I used count() to check how many records are in each dataset. I then printed the number of rows in both the training and test sets to confirm that the split worked as expected and to get a sense of the data distribution.

# COMMAND ----------

train, test = final_df.randomSplit([0.8, 0.2], seed=42)

print("Training data count:", train.count())
print("Test data count:", test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # BUILDING AND TRAINING MY RECOMMENDATION MODEL USING ALS (ALTERNATING LEAST SQUARES)
# MAGIC I built and trained a recommendation model using the ALS (Alternating Least Squares) algorithm from PySpark. I started by importing the ALS class and configuring the model with the appropriate parameters. I specified user, item, and rating as the input columns, which represent the user IDs, game IDs, and hours played respectively. I set implicitPrefs=True because this dataset represents implicit feedback (such as playtime) rather than explicit ratings. I also configured key hyperparameters, including rank=20 to control the number of latent factors, maxIter=15 to define how many iterations the model should run, and regParam=0.1 to help prevent overfitting through regularisation. Additionally, I set coldStartStrategy="drop" to remove any users or items in the predictions that the model has not seen during training. Finally, I trained the model by fitting it on the training dataset using als.fit(train), which produced the trained recommendation model stored in the variable model.

# COMMAND ----------

from pyspark.ml.recommendation import ALS

als = ALS(
    userCol="user",
    itemCol="item",
    ratingCol="rating",
    implicitPrefs=True,
    rank=20,
    maxIter=15,
    regParam=0.1,
    coldStartStrategy="drop"
)

model = als.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EVALUATING THE PERFORMANCE OF MY RECOMMENDATION MODEL
# MAGIC Next step was to evaluate the performance of my recommendation model. I started by generating predictions on the test dataset using model.transform(test), which applies the trained ALS model to estimate ratings (or playtime) for each user-item pair in the test set. I then created a RegressionEvaluator to measure how accurate these predictions are. I set the evaluation metric to RMSE (Root Mean Squared Error), which calculates the average difference between the predicted values and the actual values. I also specified rating as the true label column and prediction as the model’s output column. After setting up the evaluator, I used it to compute the RMSE by passing in the predictions DataFrame. Finally, I printed the result, which gives me a single value indicating how well the model is performing—the lower the RMSE, the better the model’s predictions align with the actual data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

predictions = model.transform(test)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GENERATING PERSONALISED RECOMMENDATIONS FOR EVERY USER IN THE DATASET
# MAGIC My next step was to generate personalised recommendations for every user in the dataset. I used the recommendForAllUsers(5) function on the trained model to retrieve the top 5 recommended items (games) for each user based on their predicted preferences. This function returns a DataFrame where each row represents a user, along with a list of recommended items and their predicted ratings. I then used .show(truncate=False) to display the full results without cutting off the recommendation lists, so I can clearly see all the suggested games and their associated scores for each user.

# COMMAND ----------

user_recs = model.recommendForAllUsers(5)
user_recs.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GENERATING RECOMMENDATIONS FOR EACH GAME
# MAGIC I also made sure to generate recommendations from the perspective of each game. I used the recommendForAllItems(5) function on the trained model to find the top 5 users who are most likely to interact with each game based on the model’s predictions. This produces a DataFrame where each row represents a game, along with a list of recommended users and their predicted ratings (or expected level of interest). I then used .show(truncate=False) to display the full results without truncating the recommendation lists, so I can clearly see all the suggested users and their associated scores for each game.

# COMMAND ----------

game_recs = model.recommendForAllItems(5)
game_recs.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # CONVERTING OUR RECOMMENDATIONS BACK INTO UNDERSTANDABLE GAME NAMES
# MAGIC ## EXTRACTING THE MAPPED GAME NAMES FROM THE INDEXED GAME IDs
# MAGIC After our recommendations, it was important to convert these ids back into names that users can see and understand. I extracted the mapping between the indexed game IDs and their original game names. I started by fitting the game_indexer on the df_play DataFrame, which creates a model that assigns a unique numeric index to each game. After fitting the indexer, I accessed the .labels attribute from the fitted model. This returns a list of all the unique game names in the exact order they were assigned their numeric indices. This mapping is useful because it allows me to interpret the model’s output by converting the numeric game_id values back into their original game names.

# COMMAND ----------

game_model = game_indexer.fit(df_play)
game_labels = game_model.labels

# COMMAND ----------

# MAGIC %md
# MAGIC ## CONVERTING THE RECOMMENDATIONS BACK INTO UNDERSTANDABLE NAMES
# MAGIC The next step was to take the user_recs DataFrame, which contains a list of recommendations for each user, and use the explode function to break down the nested recommendations column so that each recommended item appears on its own row. This gave me one row per user–recommendation pair. After that, I extracted the relevant fields from the nested structure. I selected the user, the recommended item (which I renamed to game_id), and the predicted rating (which I renamed to score) to create a cleaner DataFrame. I then converted this Spark DataFrame into a Pandas DataFrame using toPandas() so I could work with it more easily for analysis and display. To make the results more meaningful, I mapped the numeric game_id values back to their original game names using the game_labels list I created earlier. This allows me to see actual game titles instead of numeric IDs. Finally, I used head(10) to display the first 10 rows of the cleaned recommendations, giving me a quick preview of the results in a clear and readable format.

# COMMAND ----------

from pyspark.sql.functions import explode, col

# Explode recommendations (one row per recommendation)
user_recs_exp = user_recs.select(
    col("user"),
    explode("recommendations").alias("rec")
)

# Extract item and score
user_recs_clean = user_recs_exp.select(
    col("user"),
    col("rec.item").alias("game_id"),
    col("rec.rating").alias("score")
)

pdf_user_recs = user_recs_clean.toPandas()

# Map game_id → game name
pdf_user_recs["game"] = pdf_user_recs["game_id"].apply(lambda x: game_labels[x])

pdf_user_recs.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, I transformed the item-based recommendations into a more readable format. I started with the game_recs DataFrame, which contains a list of recommended users for each game, and used the explode function to break down the nested recommendations column so that each recommendation appears on its own row. This gives me one row per game–user recommendation pair. After that, I selected and renamed the relevant fields to make the data clearer. I kept the item column and renamed it to game_id, and then extracted the user and rating from the nested structure, renaming them to recommended_user and score respectively. I then converted the Spark DataFrame into a Pandas DataFrame using toPandas() so I could work with it more easily. To make the output more meaningful, I mapped the numeric game_id values back to their original game names using the game_labels list I created earlier. Finally, I used head(10) to display the first 10 rows, giving me a quick and readable preview of which users are most likely to engage with each game based on the model’s predictions.

# COMMAND ----------

game_recs_exp = game_recs.select(
    col("item"),
    explode("recommendations").alias("rec")
)

game_recs_clean = game_recs_exp.select(
    col("item").alias("game_id"),
    col("rec.user").alias("recommended_user"),
    col("rec.rating").alias("score")
)

pdf_game_recs = game_recs_clean.toPandas()

pdf_game_recs["game"] = pdf_game_recs["game_id"].apply(lambda x: game_labels[x])

pdf_game_recs.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MAKING RECOMMENDATIONS FOR THE TOP USER AND THE TOP GAME
# MAGIC Finally, I also made sure to make recommendations for the top user, and the top game. I filtered the pdf_user_recs DataFrame to focus on a single user (user 0). I then sorted their recommendations by score in descending order so that the highest-ranked recommendations appear at the top. This allows me to clearly see which games the model predicts this user is most likely to engage with. Next, I looked at recommendations from the game perspective. I filtered the pdf_game_recs DataFrame to focus specifically on the game Dota 2, and then sorted the results by score in descending order. This shows me which users are most likely to play or engage with Dota 2 according to the model. Finally, I used display() to present both filtered results in a table format within Databricks, making it easy to interpret and compare the recommendations.

# COMMAND ----------

display(pdf_user_recs[pdf_user_recs["user"] == 0] \
    .sort_values("score", ascending=False))

display(pdf_game_recs[pdf_game_recs["game"] == "Dota 2"] \
    .sort_values("score", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## TRACKING MY EXPERIMENT USING ML FLOW AND RECORDING THE PERFORMANCE OF THE RECOMMENDATION MODEL
# MAGIC The final step in my experiment is to use MLflow to track my experiment and record the performance of the recommendation model. I started by importing MLflow and setting the experiment path using mlflow.set_experiment(), which ensures that all runs are logged under a specific workspace location for easy organisation and comparison. I then started a new MLflow run using mlflow.start_run() so that everything executed within this block is tracked. Inside the run, I defined the ALS model again with the same parameters, including the user, item, and rating columns, as well as key hyperparameters like rank, number of iterations, and regularisation. I then trained the model using the training dataset and generated predictions on the test dataset. After generating predictions, I printed the total number of predictions to confirm that the model is producing outputs as expected. I then evaluated the model using RMSE to measure how accurate the predictions are, and printed the result. Finally, I logged the key model parameters (rank, max iterations, and regularisation) along with the RMSE metric to MLflow. This allows me to keep track of different experiments, compare model performance, and easily revisit the best-performing configurations later.

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Shared/steam_recommender")

with mlflow.start_run():
    als = ALS(
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        implicitPrefs=True,
        rank=20,
        maxIter=10,
        regParam=0.1,
        coldStartStrategy="drop"
    )

    model = als.fit(train)
    predictions = model.transform(test)

    print("Predictions count:", predictions.count())

    rmse = evaluator.evaluate(predictions)
    print("RMSE:", rmse)

    if rmse is not None:
        mlflow.log_param("rank", 20)
        mlflow.log_param("maxIter", 10)
        mlflow.log_param("regParam", 0.1)
        mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# DBTITLE 1,Ranking-Based Evaluation Metrics
# MAGIC %md
# MAGIC ## EVALUATING RECOMMENDATION QUALITY USING RANKING-BASED METRICS
# MAGIC While RMSE measures the accuracy of predicted ratings, it does not capture how well the model ranks items for each user. For recommendation systems, ranking-based metrics are more meaningful because they evaluate whether the model surfaces relevant items in the top positions. I calculated three key metrics:
# MAGIC - **Precision@K**: Of the top K recommended games for each user, what proportion were actually played by the user in the test set? A higher precision means the model is recommending more relevant games.
# MAGIC - **Recall@K**: Of all the games a user actually played in the test set, what proportion appeared in the top K recommendations? A higher recall means the model is capturing more of the user's true preferences.
# MAGIC - **MAP (Mean Average Precision)**: The mean of the average precision scores across all users. Average precision rewards models that place relevant items higher in the recommendation list, making MAP a comprehensive measure of ranking quality.
# MAGIC
# MAGIC I evaluated these metrics at K=5 and K=10 to understand how recommendation quality changes with the number of suggestions.

# COMMAND ----------

# DBTITLE 1,Calculate Precision@K, Recall@K, and MAP
from pyspark.sql.functions import collect_set, col, explode, array, lit, udf, avg
from pyspark.sql.types import FloatType, ArrayType, IntegerType
import numpy as np

# Step 1: Get actual items per user from the test set
actual_items = test.groupBy("user").agg(
    collect_set("item").alias("actual_items")
)

# Step 2: Get top-K recommendations per user
def evaluate_at_k(model, actual_items_df, k):
    """Calculate precision@k, recall@k, and MAP@k."""
    
    # Get top-K recommendations for all users
    user_recs = model.recommendForAllUsers(k)
    
    # Explode recommendations and collect as list of item IDs
    recs_flat = user_recs.select(
        col("user"),
        explode("recommendations").alias("rec")
    ).select(
        col("user"),
        col("rec.item").alias("rec_item")
    )
    
    rec_items = recs_flat.groupBy("user").agg(
        collect_set("rec_item").alias("rec_items")
    )
    
    # Also collect as ordered list for MAP calculation
    from pyspark.sql.functions import collect_list, struct, sort_array
    recs_ordered = user_recs.select(
        col("user"),
        col("recommendations.item").alias("rec_items_ordered")
    )
    
    # Join actual and recommended items
    joined = actual_items_df.join(rec_items, on="user", how="inner")
    joined = joined.join(recs_ordered, on="user", how="inner")
    
    # Convert to pandas for metric calculation
    pdf = joined.toPandas()
    
    precisions = []
    recalls = []
    avg_precisions = []
    
    for _, row in pdf.iterrows():
        actual = set(row["actual_items"])
        recommended = set(row["rec_items"])
        rec_ordered = list(row["rec_items_ordered"])
        
        if len(actual) == 0:
            continue
        
        # Precision@K: relevant items in top-K / K
        hits = len(actual & recommended)
        precision = hits / k
        precisions.append(precision)
        
        # Recall@K: relevant items in top-K / total relevant
        recall = hits / len(actual)
        recalls.append(recall)
        
        # Average Precision: rewards relevant items ranked higher
        ap = 0.0
        num_hits = 0
        for i, item in enumerate(rec_ordered):
            if item in actual:
                num_hits += 1
                ap += num_hits / (i + 1)
        if num_hits > 0:
            ap /= (len(actual) if len(actual) < k else k)
        avg_precisions.append(ap)
    
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    map_score = np.mean(avg_precisions)
    
    return mean_precision, mean_recall, map_score

# Step 3: Evaluate at K=5 and K=10
print("=" * 60)
print("RANKING-BASED EVALUATION METRICS")
print("=" * 60)

for k in [5, 10]:
    precision, recall, map_score = evaluate_at_k(model, actual_items, k)
    print(f"\n--- K = {k} ---")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}:    {recall:.4f}")
    print(f"MAP@{k}:       {map_score:.4f}")

print("\n" + "=" * 60)

# COMMAND ----------

# DBTITLE 1,Scaling with Hyperparameter Tuning
# MAGIC %md
# MAGIC ## SCALING THE RECOMMENDER SYSTEM WITH HYPERPARAMETER TUNING
# MAGIC To find the best-performing ALS model, I performed a systematic hyperparameter grid search across three key parameters:
# MAGIC - **rank** (number of latent factors): [10, 20, 50] — controls the complexity of user/item representations
# MAGIC - **regParam** (regularisation): [0.01, 0.1, 0.5] — prevents overfitting by penalising large factor values
# MAGIC - **maxIter** (training iterations): [10, 15, 20] — controls how long the model trains
# MAGIC
# MAGIC This produces 27 different configurations. For each one, I train an ALS model, evaluate it using RMSE and Precision@10, and log all parameters and metrics to MLflow. After all runs complete, I identify the best configuration and visualise the results to compare performance across all experiments.

# COMMAND ----------

# DBTITLE 1,Hyperparameter Grid Search with MLflow
import mlflow
import itertools
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

mlflow.set_experiment("/Shared/steam_recommender")

# Define hyperparameter grid
ranks = [10, 20, 50]
reg_params = [0.01, 0.1, 0.5]
max_iters = [10, 15, 20]

results = []
total = len(ranks) * len(reg_params) * len(max_iters)
run_num = 0

for rank_val, reg, iters in itertools.product(ranks, reg_params, max_iters):
    run_num += 1
    with mlflow.start_run(run_name=f"ALS_r{rank_val}_reg{reg}_iter{iters}"):
        als_tuned = ALS(
            userCol="user",
            itemCol="item",
            ratingCol="rating",
            implicitPrefs=True,
            rank=rank_val,
            maxIter=iters,
            regParam=reg,
            coldStartStrategy="drop"
        )

        tuned_model = als_tuned.fit(train)
        preds = tuned_model.transform(test)

        eval_rmse = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        rmse_val = eval_rmse.evaluate(preds)

        # Calculate Precision@10 for ranking quality
        p10, r10, map10 = evaluate_at_k(tuned_model, actual_items, 10)

        # Log to MLflow
        mlflow.log_param("rank", rank_val)
        mlflow.log_param("maxIter", iters)
        mlflow.log_param("regParam", reg)
        mlflow.log_metric("rmse", rmse_val)
        mlflow.log_metric("precision_at_10", p10)
        mlflow.log_metric("recall_at_10", r10)
        mlflow.log_metric("map_at_10", map10)

        results.append({
            "rank": rank_val, "regParam": reg, "maxIter": iters,
            "rmse": rmse_val, "precision@10": p10, "recall@10": r10, "MAP@10": map10
        })
        print(f"[{run_num}/{total}] rank={rank_val}, reg={reg}, iter={iters} "
              f"→ RMSE={rmse_val:.2f}, P@10={p10:.4f}, R@10={r10:.4f}, MAP@10={map10:.4f}")

print(f"\nCompleted {len(results)} experiments.")

# COMMAND ----------

# DBTITLE 1,Best Model Selection and Results Summary
import pandas as pd
import matplotlib.pyplot as plt

# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by MAP@10 (best ranking quality)
results_by_map = results_df.sort_values("MAP@10", ascending=False)

print("=" * 70)
print("TOP 5 CONFIGURATIONS BY MAP@10 (RANKING QUALITY)")
print("=" * 70)
display(results_by_map.head(5))

best = results_by_map.iloc[0]
print(f"\nBest model: rank={int(best['rank'])}, regParam={best['regParam']}, maxIter={int(best['maxIter'])}")
print(f"  RMSE:        {best['rmse']:.2f}")
print(f"  Precision@10: {best['precision@10']:.4f}")
print(f"  Recall@10:    {best['recall@10']:.4f}")
print(f"  MAP@10:       {best['MAP@10']:.4f}")

# Visualise: scatter plot of RMSE vs MAP@10, coloured by regParam
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: RMSE vs MAP@10
for reg in reg_params:
    subset = results_df[results_df["regParam"] == reg]
    axes[0].scatter(subset["rmse"], subset["MAP@10"],
                    label=f"regParam={reg}", s=80, alpha=0.8)
axes[0].set_xlabel("RMSE", fontsize=12)
axes[0].set_ylabel("MAP@10", fontsize=12)
axes[0].set_title("RMSE vs MAP@10 by Regularisation", fontsize=13, fontweight="bold")
axes[0].legend()

# Right: Precision@10 by rank, grouped by maxIter
for iters in max_iters:
    subset = results_df[results_df["maxIter"] == iters]
    avg_by_rank = subset.groupby("rank")["precision@10"].mean()
    axes[1].plot(avg_by_rank.index, avg_by_rank.values,
                 marker="o", label=f"maxIter={iters}", linewidth=2)
axes[1].set_xlabel("Rank (Latent Factors)", fontsize=12)
axes[1].set_ylabel("Avg Precision@10", fontsize=12)
axes[1].set_title("Precision@10 by Rank and Iterations", fontsize=13, fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## CONCLUSION
# MAGIC In this project, I built a personalised game recommendation system using the Steam dataset and PySpark's ALS (Alternating Least Squares) algorithm within Databricks. The goal was to leverage implicit feedback — specifically, the number of hours users spent playing each game — to predict which games a user is most likely to enjoy.
# MAGIC
# MAGIC ### Key Findings from Exploratory Data Analysis
# MAGIC - The dataset contained **200,000 records** across **12,393 unique users** and **5,155 unique games**, with two behaviour types: purchase (129,511) and play (70,489).
# MAGIC - **Dota 2** dominated total playtime with over **981,000 hours**, followed by Counter-Strike Global Offensive (322,772 hours) and Team Fortress 2 (173,673 hours), showing a highly skewed distribution of engagement.
# MAGIC - The **top 10 most active users** each accumulated between 8,137 and 11,754 total hours of playtime, indicating a small group of power users driving significant engagement.
# MAGIC
# MAGIC ### Baseline Model Performance
# MAGIC - The data was split into **80% training (56,518 records)** and **20% testing (13,971 records)** to evaluate the model.
# MAGIC - The initial ALS model was configured with `rank=20`, `maxIter=10`, `regParam=0.1`, and `implicitPrefs=True`, producing **12,239 predictions** on the test set.
# MAGIC - The baseline model achieved an **RMSE of 231.68**. While this value appears high, it is important to note that the rating values represent raw hours played, which range from 0 to over 10,000 hours. The heavy-tailed distribution of playtime means that a small number of extreme values significantly inflate the RMSE.
# MAGIC
# MAGIC ### Ranking-Based Evaluation
# MAGIC To better assess recommendation quality beyond RMSE, I calculated ranking-based metrics on the baseline model:
# MAGIC - **Precision@5: 0.0666** and **Precision@10: 0.0512** — indicating approximately 1 in 15 top-5 recommendations was a game the user actually played. While this may seem low, compared to a random recommender (which would achieve roughly 0.1% precision across 5,155 games), the model performs **significantly better than random**.
# MAGIC - **Recall@10: 0.2296** — the model captures approximately 23% of the games a user played in just 10 recommendations.
# MAGIC - **MAP@10: 0.0927** — showing that relevant items are placed reasonably well within the top positions of the recommendation list.
# MAGIC
# MAGIC ### Hyperparameter Tuning
# MAGIC To improve the model, I performed a systematic grid search across **27 configurations**, varying rank [10, 20, 50], regParam [0.01, 0.1, 0.5], and maxIter [10, 15, 20]. Each configuration was trained, evaluated using both RMSE and ranking metrics (Precision@10, Recall@10, MAP@10), and logged to MLflow for comparison. The key findings were:
# MAGIC - **Best configuration: `rank=10, regParam=0.01, maxIter=20`**, achieving a MAP@10 of **0.0979**, Precision@10 of **0.0548**, and Recall@10 of **0.2320**.
# MAGIC - **Lower rank values (10-20) consistently outperformed rank=50**, suggesting the user-item interaction space does not require a high number of latent factors.
# MAGIC - **Lower regularisation (0.01) yielded the best results**, likely because the implicit feedback signal is already sparse and does not require strong penalisation.
# MAGIC - **More training iterations (20) generally improved performance**, giving the algorithm sufficient time to converge.
# MAGIC
# MAGIC ### Recommendation Results
# MAGIC - **User-based recommendations**: For example, user 0 was recommended games such as *Orcs Must Die! 2*, *Just Cause 2*, *Dragon Age Origins*, *Alien Swarm*, and *Counter-Strike Source* — all action and adventure titles consistent with typical gaming preferences.
# MAGIC - **Game-based recommendations**: For *Dota 2*, the model identified users 3181, 1594, 2401, 3894, and 3374 as the most likely to engage, demonstrating the model's ability to match games with suitable audiences.
# MAGIC - All numeric IDs were successfully mapped back to readable game names using the StringIndexer labels, making the output interpretable and actionable.
# MAGIC
# MAGIC ### Experiment Tracking
# MAGIC - All model parameters and performance metrics — including RMSE, Precision@10, Recall@10, and MAP@10 — were logged to **MLflow** under the `/Shared/steam_recommender` experiment across **28 total runs** (1 baseline + 27 grid search), enabling systematic comparison and full reproducibility of results.
# MAGIC
# MAGIC ### Summary
# MAGIC Overall, this project demonstrated how distributed computing with PySpark and Databricks can be used to build scalable recommendation systems from implicit feedback data. The ALS algorithm proved effective at capturing user preferences from playtime patterns, and the hyperparameter grid search confirmed that a simpler model (`rank=10`) with low regularisation (`regParam=0.01`) and sufficient training iterations (`maxIter=20`) delivers the best ranking quality. MLflow provided a robust framework for tracking and comparing all 28 experiments. Future improvements could include log-transforming playtime values to reduce the impact of extreme outliers, incorporating purchase behaviour as an additional signal, and exploring content-based or hybrid approaches to further enhance recommendation diversity and accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC # DISCLAIMER
# MAGIC AI tools were used in the making of this code to aid understanding and to make the code more comprehensive.