# with open(r'examples/example_tweets.txt', 'r') as f:
#     print(f.readlines())

my_tweets = ["Hi! this is my tweet", "it is kinda bad probably idk"]
with open (r'examples/make_examples.txt', 'w') as f:
    for tweet in my_tweets:
        f.write(str(tweet+'\n'))

f.close()