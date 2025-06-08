import io
import spacy

nlp = spacy.load("en_core_web_sm")

# pip install spacy
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz

def filter_keywords(keywords):
    filtered_keywords = []
    
    for phrase in keywords:
        doc = nlp(phrase)
        if len(doc) == 1:
            if doc[0].pos_ in ["NOUN", "ADJ", "VERB"] and phrase != 'video':
                filtered_keywords.append(phrase)
        else:
            is_valid = False
            if len(doc) == 2 and (
                (doc[0].pos_ == "ADJ" and doc[1].pos_ in ["NOUN", "PROPN"]) or
                (doc[0].pos_ in ["NOUN", "PROPN"] and doc[1].pos_ in ["NOUN", "PROPN"])
            ):
                is_valid = True
            elif len(doc) == 3 and doc[0].pos_ == "ADJ" and doc[1].pos_ in ["NOUN", "PROPN"] and doc[2].pos_ in ["NOUN", "PROPN"]:
                is_valid = True
            elif len(doc) == 2 and doc[0].pos_ == "VERB" and doc[1].pos_ in ["NOUN", "PROPN"]:
                is_valid = True
            
            if is_valid and phrase != 'video':
                filtered_keywords.append(phrase)
    
    return filtered_keywords