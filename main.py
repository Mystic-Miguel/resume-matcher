import argparse, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def main():
    p = argparse.ArgumentParser(description="Simple TF-IDF resume matcher")
    p.add_argument("--resume", required=True, help="txt file")
    p.add_argument("--jd", required=True, help="txt file")
    args = p.parse_args()
    docs = [open(args.resume, encoding="utf-8").read(), open(args.jd, encoding="utf-8").read()]
    tf = TfidfVectorizer(stop_words="english")
    X = tf.fit_transform(docs)
    sim = cosine_similarity(X[0], X[1])[0][0]
    print(f"Match score: {sim:.3f}")
if __name__ == "__main__":
    sys.exit(main())
