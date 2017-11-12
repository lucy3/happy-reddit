"""
Contains a function that returns 
a dictionary of comment id
to an array of features. 
"""

POST_IDs = "../data/post_IDs.txt"
INPUT = "/dfs/dataset/infolab/Reddit/comments/2015/RC_2015-05"

def get_text():
    '''
    @input
        - 
    @return
        - 
    '''
    with open(POST_IDS, 'r') as 
    with open(INPUT, 'r') as input_file: 
        for line in input_file: 
            comment = json.loads(line)

def get_lexical_features():
    '''
    @input
        - 
    @return
        - 
    '''
    pass

def main():
    get_text()
    get_lexical_features()

if __name__ == "__main__":
    main()