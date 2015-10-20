import json

reviewsRead = open('yelp_academic_dataset_review.json')
reviews = []
n = 0
for line in reviewsRead:
    review = json.loads(line)
    print line
    reviews.append(review)
    n = n+1
    if n == 10:
        break
for review in reviews:
    print review["text"]
