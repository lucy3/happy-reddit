# Gilding on Reddit

## Code

The contents of **data**, **logs**, **results** contain the inputs and outputs of these files: 

- **Log Odds.ipynb**: for visualizing log odds plot
- **Trends and Statistics.ipynb**: upvote popularity of gilded vs non-gilded comments, pearson correlation between topic similarity and user similarity graphs, and counting deleted gilds. 
- **access\_comment\_id.py**: reformatting comment into jsons
- **access\_post\_id.py**: reformatting posts into jsons
- **classifier.py**: classifier/s for predicting gilding and comment rank
- **classifier\_labels.py**: creates labels for classifier
- **classifier\_results.py**: reformats results of classifier
- **detect_communities.py**: community detection on subreddit network
- **get\_deleted\_authors.py**: count deleted gilds
- **get\_post\_IDs.py**: extracts all post IDs from May 2015
- **get\_social\_features.py**: social features for classifier
- **get\_subreddit\_counts.py**: number of comments on subreddits, long-tail plot
- **get\_total\_comments.py**: number of comments and posts
- **lexical\_features.py**: lexical features for classifier
- **liwc\_features.py**: liwc features for classifier
- **log\_odds.py**: calculating and clustering log odds
- **preprocess\_feature\_data.py**: preprocessing for lexical features
- **social\_temp\_features.py**: empty
- **subreddit\_graph.py**: generate user and text networks