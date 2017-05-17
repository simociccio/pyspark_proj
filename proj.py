


from pyspark import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql import functions as fn
from nltk.corpus import stopwords
from pyspark.ml.feature import Tokenizer
import requests
import pandas as pd
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import treetaggerwrapper

sql_c = SQLContext(sc)
d0 = sc.textFile('*.tsv',use_unicode=False)
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1], score = v[2]))
df = sql_c.createDataFrame(d1)
df2=df.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))

pdf=df2.toPandas()
pdf['score']=pd.to_numeric(pdf['score'], errors='coerce').fillna(0)
df3 = sql_c.createDataFrame(pdf)


tokenizer = Tokenizer().setInputCol('tweet').setOutputCol('token')


stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()

sw_filter = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("token")\
  .setOutputCol("filtered")


# we will remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)\
  .setInputCol("filtered")\
  .setOutputCol("tf")


idf = IDF().\
    setInputCol('tf').\
    setOutputCol('tfidf')



cv_pipeline = Pipeline(stages=[tokenizer, sw_filter,cv]).fit(df3)		
cv_pipeline.transform(df3).show(5)  
idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(df3)
idf_pipeline.transform(df3).show(5)
tfidf_df = idf_pipeline.transform(df3)

nb = NaiveBayes(smoothing=1.0, modelType="multinomial").setLabelCol('score').setFeaturesCol('tfidf')



training_df=df3
en_nb_pipeline = Pipeline(stages=[idf_pipeline, nb]).fit(training_df)



d0 = sc.textFile('/home/saimon/Scaricati/gold_data/Cereal_Songs.tsv')
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1]))
df3 = sql_c.createDataFrame(d1)
df3=df3.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))
en_nb_pipeline1 = Pipeline(stages=[idf_pipeline, nb]).fit(training_df).transform(df3).select('id','tweet','prediction').toPandas().to_csv('Cereal_Songs_PREDICT.tsv','\t', index=False, header=False)


d0 = sc.textFile('/home/saimon/Scaricati/gold_data/Bad_Job_In_5_Words.tsv')
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1]))
df4 = sql_c.createDataFrame(d1)
df4=df4.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))
en_nb_pipeline2 = Pipeline(stages=[idf_pipeline, nb]).fit(training_df).transform(df4).select('id','tweet','prediction').toPandas().to_csv('Bad_Job_In_5_Words_PREDICT.tsv','\t', index=False, header=False)

d0 = sc.textFile('/home/saimon/Scaricati/gold_data/Break_Up_In_5_Words.tsv')
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1]))
df5 = sql_c.createDataFrame(d1)
df5=df5.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))
en_nb_pipeline3 = Pipeline(stages=[idf_pipeline, nb]).fit(training_df).transform(df5).select('id','tweet','prediction').toPandas().to_csv('Break_Up_In_5_Words_PREDICT.tsv','\t', index=False, header=False)

d0 = sc.textFile('/home/saimon/Scaricati/gold_data/Broadway_A_Celeb.tsv')
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1]))
df6 = sql_c.createDataFrame(d1)
df6=df6.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))
en_nb_pipeline4 = Pipeline(stages=[idf_pipeline, nb]).fit(training_df).transform(df6).select('id','tweet','prediction').toPandas().to_csv('Broadway_A_Celeb_PREDICT.tsv','\t', index=False, header=False)

d0 = sc.textFile('/home/saimon/Scaricati/gold_data/Modern_Shakespeare.tsv')
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1]))
df7 = sql_c.createDataFrame(d1)
df7=df7.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))
en_nb_pipeline5 = Pipeline(stages=[idf_pipeline, nb]).fit(training_df).transform(df7).select('id','tweet','prediction').toPandas().to_csv('Modern_Shakespeare_PREDICT.tsv','\t', index=False, header=False)

d0 = sc.textFile('/home/saimon/Scaricati/gold_data/Ruin_A_Christmas_Movie.tsv')
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda v : Row(id = v[0], tweet = v[1]))
df8 = sql_c.createDataFrame(d1)
df8=df8.withColumn('tweet',regexp_replace('tweet','@\w+|\.|\s\s+',''))
en_nb_pipeline6 = Pipeline(stages=[idf_pipeline, nb]).fit(training_df).transform(df8).select('id','tweet','prediction').toPandas().to_csv('Ruin_A_Christmas_Movie_PREDICT.tsv','\t', index=False, header=False)




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

dfp=df2.toPandas()

def postag_cell(pandas_cell):
    import pprint   # For proper print of sequences.
    import treetaggerwrapper
    tagger = treetaggerwrapper.TreeTagger(TAGDIR='/home/saimon/Scaricati',TAGLANG='en')
    #2) tag your text.
    y = [i.decode('UTF-8') if isinstance(i, basestring) else i for i in [pandas_cell]]
    tags = tagger.tag_text(y)
    #3) use the tags list... (list of string output from TreeTagger).
    return tags


dfp['POS-tagged_opinions'] = (dfp['tweet'].apply(postag_cell))

from pyspark import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql import functions as fn
from nltk.corpus import stopwords
from pyspark.ml.feature import Tokenizer
import requests
import pandas as pd
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import treetaggerwrapper

sql_c = SQLContext(sc)

//////////////Parso il file con i tweet lemmatizati////////////////////////////////////

d0 = sc.textFile('/home/saimon/Scaricati/train_dir/output1.csv',use_unicode=False)
d1 = d0.map(lambda x: x.lower().split('\t'))
d1 = d1.map(lambda x: Row(id = x[0],score = x[1],tweet = x[2],tweetLemma = x[3]))
df = sql_c.createDataFrame(d1)

tokenizer = Tokenizer().setInputCol('tweetLemma').setOutputCol('token')


stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()

sw_filter = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("token")\
  .setOutputCol("filtered")

df=df.withColumn('tweetLemma',regexp_replace('tweetLemma','@\w+|\.|\s\s+|@|\"|#',''))

filter_pipeline = Pipeline(stages=[tokenizer, sw_filter]).fit(df)
filter_pipeline.transform(df).show(5)  

pdf=filter_pipeline.transform(df).toPandas()
pdf['score']=pd.to_numeric(pdf['score'], errors='coerce').fillna(0)
df3 = sql_c.createDataFrame(pdf)


///////////////Estraggo i lemmi con i relativi pesi dal file DepecheMood_tf_idf///////////////////

d0 = sc.textFile('/home/saimon/Scaricati/DepecheMood_V1.0/DepecheMood_V1.0/DepecheMood_tfidf.txt')
d1 = d0.map(lambda x: x.lower().split('\t'))
header = d1.first()
d2=d1.filter(lambda line : line !=header)
df_dep = sql_c.createDataFrame(d2,header)
df_dep=df_dep.withColumn('lemma#pos',regexp_replace('lemma#pos','#.*',''))


/////////////////////seleziono solo i tweet con 2.0 di score////////////////////////////////

df4=df3.select('*').where(df3.score==2.0)
//seleziono parola per parola dalle filtered
pdf_lemma=df4.toPandas()
s = pdf_lemma.apply(lambda x: pd.Series(x['filtered']),axis=1).stack().reset_index(level=1, drop=True)
s.name='word'
pdf_lemma=pdf_lemma.drop('filtered', axis=1).join(s)
pdf_dep=df_dep.toPandas()
pdf_lemma.index.names = ['ind']
pdf_dep.index.names = ['ind']
pdf_merge=pdf_dep.merge(pdf_lemma,left_on='lemma#pos',right_on='word')
pdf_prova=pdf_merge.groupby(['id'])['afraid']['amused'].sum()
pdf_merge['afraid']=pd.to_numeric(pdf_merge['afraid'], errors='coerce').fillna(0)
pdf_merge['amused']=pd.to_numeric(pdf_merge['amused'], errors='coerce').fillna(0)
pdf_merge['angry']=pd.to_numeric(pdf_merge['angry'], errors='coerce').fillna(0)
pdf_merge['annoyed']=pd.to_numeric(pdf_merge['annoyed'], errors='coerce').fillna(0)
pdf_merge['dont_care']=pd.to_numeric(pdf_merge['dont_care'], errors='coerce').fillna(0)
pdf_merge['happy']=pd.to_numeric(pdf_merge['happy'], errors='coerce').fillna(0)
pdf_merge['inspired']=pd.to_numeric(pdf_merge['inspired'], errors='coerce').fillna(0)
pdf_merge['sad']=pd.to_numeric(pdf_merge['sad'], errors='coerce').fillna(0)


pdf_fin=pdf_merge.groupby(['id'], as_index=False).sum()
df_fin = sql_c.createDataFrame(pdf_fin)
sf_fin.show()







