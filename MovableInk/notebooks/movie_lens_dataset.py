#!/usr/bin/env python

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, count, desc, row_number, col, length
from pyspark.sql.window import Window
from tabulate import tabulate

spark = (
    SparkSession
    .builder
    .appName("MovieLens Analysis") 
    .config("spark.sql.adaptive.enabled", "true") 
    .config("spark.sql.shuffle.partitions", "8") 
    .getOrCreate()
)


# Read ratings.csv, tags.csv, and movies.csv
ratings_df = spark.read.csv("../input_data/ratings.csv", header=True)
tags_df = spark.read.csv("../input_data/tags.csv", header=True)
movies_df = spark.read.csv("../input_data/movies.csv", header=True)

# Function to print null value counts for a DataFrame
def print_null_info(df, name):
    null_info = []
    for col_name in df.columns:
        null_count = df.where(col(col_name).isNull()).count()
        null_info.append((col_name, null_count))
    
    print(f"Null values in {name} DataFrame:")
    print(tabulate(null_info, headers=["Column", "Null Count"], tablefmt="pretty"))


print_null_info(ratings_df, "ratings")

print_null_info(tags_df, "tags")

print_null_info(movies_df, "movies")

ratings_df.show(5)

tags_df.show(5)

movies_df.show(5)

def most_common_tag_for_movie_title(tags_df, movies_df):
    """
    Function finds the most common tag for a movie title.

    Args:
    - tags_df: DataFrame containing tags data
    - movies_df: DataFrame containing movies data

    Returns:
    - DataFrame containing the most common tag for the movie title
    """
    movie_tag_df = movies_df.join(tags_df, "movieId", "left")
    most_common_tag_df = (
        movie_tag_df
        .groupBy("title", "tag")
        .agg(count("*").alias("tag_count"))
        .orderBy(desc("tag_count"))
    )

    # Get the row with the highest tag count for each movie
    # Title Memento (2000) had multiple tags (nonlinear, twist ending). Handle it
    window_spec = Window.partitionBy("title").orderBy(desc("tag_count"))
    most_common_tag_df = (
        most_common_tag_df
        .withColumn("rank", row_number().over(window_spec))
        .filter(col("rank") == 1).drop("rank")
        .orderBy(desc('tag_count'))
    )
    
    return most_common_tag_df

most_common_tag_df = most_common_tag_for_movie_title(tags_df, movies_df)

def most_common_genre_rated_by_user(ratings_df, movies_df):
    """
    Function finds the most common genre rated by a user.

    Args:
    - ratings_df: DataFrame containing ratings data
    - movies_df: DataFrame containing movies data

    Returns:
    - DataFrame containing the most common genre rated by a user
    """
    user_rating_df = ratings_df.join(movies_df, "movieId", "left")
    most_common_genre_df = (
        user_rating_df
        .withColumn("genre", explode(split("genres", "\\|")))
        .groupBy("userId", "genre")
        .agg(count("*").alias("genre_count"))
        .orderBy(desc("genre_count"))
    )
    # Get the row with the most common genre for each user
    # userId 104 has multiple (19). Handle it
    window_spec = Window.partitionBy("userId").orderBy(desc("genre_count"))
    most_common_genre_df = (
        most_common_genre_df
        .withColumn("rank", row_number().over(window_spec))
        .filter(col("rank") == 1).drop("rank")
        .orderBy(desc('genre_count'))
    )
    return most_common_genre_df

most_common_genre_df = most_common_genre_rated_by_user(ratings_df, movies_df)
