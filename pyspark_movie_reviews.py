
# PySpark Movie Reviews Processing
# IMDb Dataset of 50K Movie Reviews

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("Movie Review Analysis") \
    .getOrCreate()

print("Spark session created successfully!")

# 2. Load the IMDb Dataset
df = spark.read.csv(
    "IMDB Dataset.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

print("Dataset loaded successfully!")

# 3. Display Dataset Information
df.printSchema()

print("First five records:")
df.show(5, truncate=False)

print("Original number of reviews:", df.count())

# 4. Clean the Dataset
cleaned_df = df.filter(
    col("review").isNotNull() &
    (trim(col("review")) != "") &
    col("sentiment").isNotNull()
)

cleaned_df = cleaned_df.withColumn(
    "review", trim(col("review"))
).withColumn(
    "sentiment", trim(col("sentiment"))
)

print("Data cleaning completed!")

print("Number of reviews after cleaning:", cleaned_df.count())

# 5. Analyze Sentiment Distribution
print("Positive and Negative Review Counts:")

cleaned_df.groupBy("sentiment").count().show()

# 6. Calculate Review Length
transformed_df = cleaned_df.withColumn(
    "review_length",
    length(col("review"))
)

print("Review length column created successfully!")

transformed_df.select(
    "review",
    "sentiment",
    "review_length"
).show(5, truncate=False)

# 7. Filter Reviews Longer Than 500 Characters
long_reviews = transformed_df.filter(
    col("review_length") > 500
)

print("Reviews longer than 500 characters:")

long_reviews.show(10, truncate=False)

print(
    "Number of long reviews:",
    long_reviews.count()
)

# 8. Save Processed Dataset
transformed_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("quoteAll", "true") \
    .csv("final_movie_reviews")

print("Processed dataset saved successfully!")

# 9. Stop Spark Session
spark.stop()

print("Movie review processing completed successfully!")
