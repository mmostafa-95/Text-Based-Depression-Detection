#!/usr/bin/env python
# coding: utf-8

# ## Tweets
# * 
# ### some issue 
# * Hashtag may have more importance ❌
# * image if can get it and describe in it ❌
# * grammer may expresses as not same lang ❌ not go throw 
# * remove names ❌ can do that for english name in space
# 
# 
# ### Done
# * some words like lol , thx , ...... and reduce typing ✔️
# * Emoji ✔️
# * may scrapping with @user name like in our data ✔️ not need 
# * words spelling ✔️
# * stop words issue like not and negative of words ,, ✔️ create custom stop words 
# * correct issue >> retain mad to make ✔️ as followed with . so remove . before  
# * lemma retain best to good may cause issue ✔️ will use it as helpfull more

# ## Good point in spacy 
# * can know some words type as name or organization .....  using NLTK for that as issue in MODIN




import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import re
import demoji
from unidecode import unidecode

import os
import modin.pandas as mpd
import ray


import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
    
# nltk.download("punkt")





EMOTICONS = {
    u":‑\)":"Happy face",u":\)":"Happy face",    u":-\]":"Happy face",u":\]":"Happy face",    u":-3":"Happy face",    u":3":"Happy face",    u":->":"Happy face",u":>":"Happy face",    u"8-\)":"Happy face",    u":o\)":"Happy face",    u":-\}":"Happy face",u":\}":"Happy face",    u":-\)":"Happy face",    u":c\)":"Happy face",    u":\^\)":"Happy face",u"=\]":"Happy face",    u"=\)":"Happy face",    u":‑D":"Laughing",u":D":"Laughing",    u"8‑D":"Laughing",u"8D":"Laughing",    u"X‑D":"Laughing",u"XD":"Laughing",    u"=D":"Laughing",u"=3":"Laughing",    u"B\^D":"Laughing",u":-\)\)":"Very happy",    u":‑\(":"sad",u":-\(":"sad",    u":\(":"sad",u":‑c":"sad",    u":c":"sad",u":‑<":"sad",    u":<":"sad",u":‑\[":"sad",    u":\[":"sad",u":-\|\|":"sad",    u">:\[":"sad",u":\{":"sad",    u":@":"sad",u">:\(":"sad",    u":'‑\(":"Crying",u":'\(":"Crying",    u":'‑\)":"Tears of happiness",u":'\)":"Tears of happiness",    u"D‑':":"Horror",u"D:<":"Disgust",    u"D:":"Sadness",u"D8":"Great dismay",    u"D;":"Great dismay",    u"D=":"Great dismay",    u"DX":"Great dismay",u":‑O":"Surprise",    u":O":"Surprise",    u":‑o":"Surprise",    u":o":"Surprise",u":-0":"Shock",    u"8‑0":"Yawn",    u">:O":"Yawn",    u":-\*":"Kiss",u":\*":"Kiss",    u":X":"Kiss",    u";‑\)":"Wink",u";\)":"Wink",    u"\*-\)":"Wink",u"\*\)":"Wink",    u";‑\]":"Wink",u";\]":"Wink",    u";\^\)":"Wink",u":‑,":"Wink",    u";D":"Wink",u":‑P":"Tongue sticking out",u":P":"Tongue sticking out",u"X‑P":"Tongue sticking out",u"XP":"Tongue sticking out",u":‑Þ":"Tongue sticking out",u":Þ":"Tongue sticking out",u":b":"Tongue sticking out",u"d:":"Tongue sticking out",u"=p":"Tongue sticking out",u">:P":"Tongue sticking out",u":‑/":"annoyed",u":/":"annoyed",u":-[.]":"annoyed",u">:[(\\\)]":"annoyed",u">:/":"annoyed",u":[(\\\)]":"annoyed",u"=/":"annoyed",u"=[(\\\)]":"annoyed",u":L":"annoyed",u"=L":"annoyed",u":S":"annoyed",u":‑\|":"Straight face",u":\|":"Straight face",u":$":"Embarrassed",u":‑x":"tongue-tied",u":x":"tongue-tied",u":‑#":"tongue-tied",u":#":"tongue-tied",u":‑&":"tongue-tied",u":&":"tongue-tied",u"O:‑\)":"innocent",u"O:\)":"innocent",u"0:‑3":"innocent",u"0:3":"innocent",u"0:‑\)":"innocent",u"0:\)":"innocent",u":‑b":"Tongue sticking out",u"0;\^\)":"innocent",u">:‑\)":"Evil",u">:\)":"Evil",u"\}:‑\)":"Evil",u"\}:\)":"Evil",u"3:‑\)":"Evil",u"3:\)":"Evil",u">;\)":"Evil",u"\|;‑\)":"Cool",u"\|‑O":"Bored",u":‑J":"Tongue-in-cheek",u"#‑\)":"Party all night",u"%‑\)":"confused",u"%\)":"confused",u":-###..":"Being sick",u":###..":"Being sick",u"<:‑\|":"Dump",u"\(>_<\)":"Troubled",u"\(>_<\)>":"Troubled",u"\(';'\)":"Baby",u"\(\^\^>``":"Nervous",u"\(\^_\^;\)":"Nervous",u"\(-_-;\)":"Nervous",u"\(~_~;\) \(・\.・;\)":"Nervous",u"\(-_-\)zzz":"Sleeping",u"\(\^_-\)":"Wink",u"\(\(\+_\+\)\)":"Confused",u"\(\+o\+\)":"Confused",u"\(o\|o\)":"Ultraman",u"\^_\^":"Joyful",u"\(\^_\^\)/":"Joyful",u"\(\^O\^\)／":"Joyful",u"\(\^o\^\)／":"Joyful",u"\(__\)":"respect",u"_\(\._\.\)_":"respect",u"<\(_ _\)>":"respect",u"<m\(__\)m>":"respect",u"m\(__\)m":"respect",u"m\(_ _\)m":"respect",u"\('_'\)":"Sad",u"\(/_;\)":"Sad",u"\(T_T\) \(;_;\)":"Sad",u"\(;_;":"Sad of Crying",u"\(;_:\)":"Sad",u"\(;O;\)":"Sad",u"\(:_;\)":"Sad",u"\(ToT\)":"Sad",u";_;":"Sad",u";-;":"Sad",u";n;":"Sad",u";;":"Sad",u"Q\.Q":"Sad",u"T\.T":"Sad",u"QQ":"Sad",u"Q_Q":"Sad",u"\(-\.-\)":"Shame",u"\(-_-\)":"Shame",u"\(一一\)":"Shame",u"\(；一_一\)":"Shame",u"\(=_=\)":"Tired",u"\(=\^\·\^=\)":"cat",u"\(=\^\·\·\^=\)":"cat",u"=_\^=":"cat",u"\(\.\.\)":"Looking down",u"\(\._\.\)":"Looking down",u"\^m\^":"Giggling with hand covering mouth",u"\(\・\・?":"Confusion",u">\^_\^<":"Normal Laugh",u"<\^!\^>":"Normal Laugh",u"\^/\^":"Normal Laugh",u"\（\*\^_\^\*）" :"Normal Laugh",u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",u"\(^\^\)":"Normal Laugh",u"\(\^\.\^\)":"Normal Laugh",u"\(\^_\^\.\)":"Normal Laugh",u"\(\^_\^\)":"Normal Laugh",u"\(\^\^\)":"Normal Laugh",u"\(\^J\^\)":"Normal Laugh",u"\(\*\^\.\^\*\)":"Normal Laugh",u"\(\^—\^\）":"Normal Laugh",u"\(#\^\.\^#\)":"Normal Laugh",u"\（\^—\^\）":"Waving",u"\(;_;\)/~~~":"Waving",u"\(\^\.\^\)/~~~":"Waving",u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",u"\(T_T\)/~~~":"Waving",u"\(ToT\)/~~~":"Waving",u"\(\*\^0\^\*\)":"Excited",u"\(\*_\*\)":"Amazed",u"\(\*_\*;":"Amazed",u"\(\+_\+\) \(@_@\)":"Amazed",u"\(\*\^\^\)v":"Laughing",u"\(\^_\^\)v":"Laughing",u"\(\(d[-_-]b\)\)":"Listening to music",u'\(-"-\)':"Worried",u"\(ーー;\)":"Worried",u"\(\^0_0\^\)":"Eyeglasses",u"\(\＾ｖ\＾\)":"Happy",u"\(\＾ｕ\＾\)":"Happy",u"\(\^\)o\(\^\)":"Happy",u"\(\^O\^\)":"Happy",u"\(\^o\^\)":"Happy",u"\)\^o\^\(":"Happy",u":O o_O":"Surprised",u"o_0":"Surprised",u"o\.O":"Surpised",u"\(o\.o\)":"Surprised",u"oO":"Surprised",u"\(\*￣m￣\)":"Dissatisfied",u"\(‘A`\)":"Snubbed"
}


# # remove unneeded words 

# ### Reduce Words (diseases , drugs , feelingwords ) getting from dir dictionaries




file ='dictionaries/diseases.txt'
All_diseases = []
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        if len(line) > 0 :
            All_diseases.append(line)





file ='dictionaries/drugs.txt'
All_drugs = {}
with open(file) as f:
    drugs_name = "Stimulant notable stimulants"
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        if len(line) > 0 and line[0]!= '#':
            All_drugs[line] = drugs_name
        
        elif len(line) > 0 :
            text = line.split('/')    
            drugs_name = text[-1]





file ='dictionaries/feelingwords_mapping.txt'
All_feelings = {}
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        text = line.split('\t')
        if len(line) > 0 :
                All_feelings[text[-1]] = text[0]





file ='dictionaries/meds.txt'
All_meds = []
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        if len(line) > 0 :
            All_meds.append(line)





file ='dictionaries/feelingwords_mapping.txt'
All_feelings = {}
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n')
        text = line.split('\t')
        if len(line) > 0 :
                All_feelings[text[-1]] = text[0]





## there is spaces in them not Handeled 
def reduce(text):
    T = text.split()
    my_doc_cleaned = [ 'diseases' if word in All_diseases else word for word in T ]
    my_doc_cleaned = [ 'meds' if word in All_meds else word for word in my_doc_cleaned ]
    my_doc_cleaned = [ All_feelings.get(word) if word in All_feelings else word for word in my_doc_cleaned ]
    my_doc_cleaned = [ All_drugs.get(word) if word in All_drugs else word for word in my_doc_cleaned ]
    
    return " ".join(my_doc_cleaned)
# reduce("i am annoyed nialamide can't")


# ### Replace Social shortcut 




file ='dictionaries/ShortCutsSocial.txt'
ShortCutsSocial = {}
with open(file) as f:
    line = f.readline()
    while line:
        line = f.readline().strip('\n').lower()
        text = line.split('â€“')
        if len(line) > 0 :
                ShortCutsSocial[text[0].strip()] =  text[-1].strip()
def Replace_ShortCut_Social(text):
    T = text.lower().split()
    my_doc_cleaned = [ ShortCutsSocial.get(word) if word in ShortCutsSocial else word for word in T ]    
    return " ".join(my_doc_cleaned)


# ### custom stop word

# * refer to "https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html "

# ![img95.png](attachment:img95.png)




custom_stop_word_list=['a','an','are','as','at','by','for','from','if','her','i','me', 'he','him','she', 'himself','they',
                       'you','yours', 'yourselves','themselves','their', 'hereupon','wherein', 'upon',
                       'in','on','onto', 'it','its','of','on','that','the','to','this', 'with','thereafter', 'thence','these',
                       'there', 'sometime','here',  'ourselves', 'when','where','what','whoever',  'whom', 'while','why',
                       'whose', 'whatever','whereas','whenever',  'with', 'who', 'how', 'whither', 'does', 'due',
                       'wherever', 'across', 'somewhere', 'my','mine',  'though', 'itself', 'whence', 'might', 'might', 'we',
                       'as','per', 'whereby', 'since', 'during', 'would', 'such', 'those','which', 'thereby', 'amount', 'at',
                       'into', 'otherwise', 'whether','somehow', 'hence', 'something', 'because', 'meanwhile', 'should', 
                       'still', 'also', 'and','else', 'along', 'another','thru',  'via', 'so', 'after', 
                       'before','may',  'about', 'namely', 'seeming', 'hereby', 'then', 'thereupon','whereafter', 'of', 'to',
                       
                       ## May effect 
                       #'have', 'becoming', 
                        'is','am','be', 'were','was','be','could','being', 'has','are', 
                       'been', 'his',  'us', 'herself',  'do', 'doing', 'both','did', 'had', 
                       ]


# # emoji




# need esmael clear using 
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)


# ### replace Emoji




def replace_emoji(tweet):
    emoji = demoji.findall(tweet)
    
    result = ''
    result1 = ''
    for emot in EMOTICONS:
        tweet = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").split()), tweet)    
    tweet = tweet.replace(u"\u2122", '')  # remove ™
    tweet = tweet.replace(u"\u20ac", '')   # remove €
    
    for char in tweet:        
        if (len(emoji.get(char, char)) > 1):
            result +=' ' +emoji.get(char, char).replace(" ", " ")
        else :
            result += emoji.get(char, char)
    result = emoji_pattern.sub(r'', result)     #remove emojis escapped from tweet    
    result = result.replace("-"," ")
    return result





# ### Remove Emoji




def remove_emoji(tweet):
    emoji = demoji.findall(tweet)
    
    result = ''
    result1 = ''
    for emot in EMOTICONS:
        tweet = re.sub(u'('+emot+')', " ", tweet)    
    for char in tweet:        
        if (len(emoji.get(char, char)) <= 1):
            result += emoji.get(char, char)

    result = emoji_pattern.sub(r'', result)     #remove emojis escapped from tweet    
    

    return result
# print(remove_emoji("If you're a programmer😂 and blocks of text are needed😭  😀😀😀  :L ;D"))


# ### transfer html code (&lt;) to < , &amp; to & and .......




def Replace_HTML_codes(text):
    from xml.sax import saxutils as su
    result = su.unescape(text)
    return result


# ### SpellChecker
# #### There is some Library 
# * SpellChecker
# * textblob
# * JamSpell can't use there is issue in install
# * SpellChecker correct any word has . as know it's abridgement and after that remove .




from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    text = text.replace("."," ");
    text = unidecode(text)  # Replcae unascii
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# ### Remove URL (http://ssdfsd ) and @username and change & to and




def Remove_Url_UserName(text): 
    text = re.sub(r"http\S+", "", text , re.IGNORECASE)         #remove url links
    text = re.sub("www.[A-Za-z0-9./]+", ' ', text,re.IGNORECASE)        #remove url links
    text = re.sub('@[^\s]+', ' ', text)     #remove user name
    text = text.replace("&", " and ")       # change & to meaning of sentence
#     text = text.replace("%", " percentage ")       # change & to persentage 
    
    text = re.sub('\n', ' ', text)          #convert to one line only 
    text = re.sub(' +', ' ', text)          #convert two or more spaces into one space
    return text


# ### Transfer 4 > four and numbers to its string 




def replace_numbers_with_string(string):
    import inflect
    items = string.split()
    Trans = inflect.engine()

    for idx, item in enumerate(items):
        try:
            repl = False
            nf = float(item)
            ni = int(nf)  
            repl = Trans.number_to_words(ni)
            items[idx] = str(repl)
        except ValueError:
            if repl != False:
                items[idx] = str(repl)  # when we reach here, item is float
    return " ".join(items)


# # Remove 
# * Names 
# * lemmatization 
# * StopWords
# * punct

# # Working with Modin




contractions = {
"ain't": "are not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how has","i'd": "I would","i'd've": "I would have","i'll": "I will","i'll've": "I will have","i'm": "I am","i've": "I have","isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that had","that'd've": "that would have","that's": "that is","there'd": "there would","there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who has","who've": "who have","why's": "why is","why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had / you would","you'd've": "you would have","you'll": "you shall / you will","you'll've": "you shall have / you will have","you're": "you are","you've": "you have"
}





lemmatizer = WordNetLemmatizer()   
def lemma_StopWords_punct(text):
    
    for cont in contractions:
        text = re.sub(u'('+cont+')', " ".join(contractions[cont].replace(",","").split()), text) 
        
    texts = "Issue in Process "
    text = unidecode(text)  # Replcae unascii
    text = re.sub(' +', ' ', text)          #convert two or more spaces into one space
                                            #     text_ = Remove_Url_UserName_digits_ConnectLines_ReduceSpaces(text)   
    lemmatized_sentence = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text_ = lemmatized_sentence

    texts = text
    texts = re.sub(' +', ' ', texts)          #convert spaces to one space as names removed 
    words = nltk.word_tokenize(texts)
    new_words= [word for word in words if word.isalnum()]
    

    my_doc_cleaned= [word for word in new_words if word not in custom_stop_word_list]
    
    
    return my_doc_cleaned





def StopWords_punct(text):
    # Remove stop words , punct 
    texts = "Issue in Process "
    text = unidecode(text)  # Replcae unascii
    text = re.sub(' +', ' ', text)          #convert two or more spaces into one space
                                            #     text_ = Remove_Url_UserName_digits_ConnectLines_ReduceSpaces(text)
    texts = text
    texts = re.sub(' +', ' ', texts)          #convert spaces to one space as names removed 
    
    words = nltk.word_tokenize(texts)
    new_words= [word for word in words if word.isalnum()]


    my_doc_cleaned= [word for word in new_words if word not in custom_stop_word_list]
     
    return my_doc_cleaned


# ## Merge Functions

# # Steps
# 


def preprocess_text(text):
    text = Remove_Url_UserName(text)
    t2 = Replace_HTML_codes(text)
    t3 = replace_emoji(t2)
    t4 = Replace_ShortCut_Social(t3)
    t5 = reduce(t4)    
    t6 = correct_spellings(t5)
    t7 = replace_numbers_with_string(t6)
    t8 = lemma_StopWords_punct(t7)

    text = ' '.join(str(v) for v in t8)
    return text
