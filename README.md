# Recom_Sys_CS282
This is a recommendation system for cs282 project. Recommender systems are pervasive in industries from e-commerce to entertainment. They provide a way to shield a customer from over-choice. As companies accumulate more and more data, the more important it becomes to have efficient and effective recommender systems. The attention mechanism, has led to a revolution in machine performance in natural language tasks. Given this success, recent work has applied the attention mechanism to recommender systems, which takes advantage of the abundance of user review data. In this project, we extend this work by proposing two new attention-based algorithms for the recommendation, inspired by document retrieval systems. We also investigate the effect of using pre-trained BERT word embeddings instead of traditional, non-contextual word2vec and GloVe embeddings on both our recommender systems as well as other systems that use user review data for the recommendation.

# How to run
All the configuration are in conf.py, if you have specific requirements, like change embedding, remember to modify it.

## download data and install the environment
`cd data`
`sh download.sh All_Beauty` 
This will download the dataset we need.

`sh download_glove.sh`
`sh download_google300d.sh`
This will download the glove, word2vec embedding needed by the project.

After that, all the required module can be installed by:
`pip install -r requirement.txt`

## Start training
`cd src && python train.py`

## Testing model
Although start a training will automatically test the model. You can still test a model manually by
`cd src && python test.py <model_path>`