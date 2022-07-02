import argparse

from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import count, split, avg, percentile_approx

parser = argparse.ArgumentParser()

parser.add_argument('--crimes_file', help='Crimes file', required=True)
parser.add_argument('--codes_file', help='File with codes', required=True)
parser.add_argument('--output_folder', help='Output folder', default='output')

args = parser.parse_args()

spark = SparkSession.builder.getOrCreate()

sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)
codesDataFrame = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .option('inferSchema', 'true') \
    .load(args.codes_file)

crimesDataFrame = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .option('inferSchema', 'true') \
    .load(args.crimes_file)

codesDataFrame \
    .orderBy(codesDataFrame.CODE.asc()) \
    .dropDuplicates(['CODE']) \
    .withColumn("CRIME_TYPE", split(codesDataFrame.NAME, ' - ')[0]) \
    .createOrReplaceTempView("offense_codes")

crimesDataFrame.createOrReplaceTempView("crimes")

crimesDataFrame.distinct().groupBy(crimesDataFrame.DISTRICT).agg(
    count("*").alias("crimes_total"),
    avg(crimesDataFrame.Lat).alias('lat'),
    avg(crimesDataFrame.Long).alias('lng'),
).createOrReplaceTempView("crimes_total")

crimesDataFrame.distinct().groupBy([crimesDataFrame.DISTRICT, crimesDataFrame.YEAR, crimesDataFrame.MONTH]).agg(
    count("*").alias("crimes_total")).groupBy(crimesDataFrame.DISTRICT).agg(
    percentile_approx("crimes_total", 0.5).alias("crimes_monthly")
).createOrReplaceTempView("crimes_median")

spark.sql("""SELECT frequentCodes.district, CONCAT_WS(',', collect_list(CRIME_TYPE)) AS frequent_crime_types
  FROM (SELECT DISTRICT,
               offense_codes.CRIME_TYPE,
               ROW_NUMBER() OVER (PARTITION BY DISTRICT ORDER BY COUNT(*) DESC) AS RANK
          FROM crimes LEFT JOIN offense_codes
            ON crimes.OFFENSE_CODE = offense_codes.code
         GROUP BY DISTRICT, offense_codes.CRIME_TYPE) AS frequentCodes
 WHERE rank <= 3
 GROUP BY DISTRICT
          """).createOrReplaceTempView("frequent_crimes")

spark.sql(
    """SELECT total.district,
       total.crimes_total,
       crimes_median.crimes_monthly,
       total.lat,
       total.lng,
       frequent_crimes.frequent_crime_types
  FROM crimes_total AS total
           LEFT JOIN crimes_median
                     ON total.DISTRICT = crimes_median.DISTRICT
           LEFT JOIN frequent_crimes
                     ON total.DISTRICT = frequent_crimes.DISTRICT
                     WHERE total.DISTRICT IS NOT NULL""") \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .parquet(args.output_folder)
