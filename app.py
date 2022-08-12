import string
import docx2txt
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
import streamlit as st
from multi_rake import Rake
rake = Rake()
import pandas as pd 


def convert_to_set(txt): 
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    listofwords = (word_tokenize(txt))
    lemmantized_list = [lemmatizer.lemmatize(i) for i in listofwords]
    return set(lemmantized_list)

st.title("Resume Optimizer")

st.write("This application compares the resume and job description input \
    and provides an indicative match strength (using Cosine similarity) to \
        provide feedbacks for candidates to optimize their resume. Delta is \
            compared against a benchmark to achieve 80% similarity")


with st.expander("Upload or input resume content manually"):
    resume_upload = st.file_uploader("Upload Resume")
    if resume_upload != None:
        resume_text = docx2txt.process(resume_upload)
        resume_text = str(resume_text.translate(str.maketrans('', '', string.punctuation)))
        resume_text = resume_text.lower()
        resume_text = remove_stopwords(resume_text)
        resume_txt = st.text_area('Resume Text', resume_text)

    else:
        resume_txt = st.text_area('Resume Text')

    rcol1, rcol2 = st.columns(2)
    rcol1.metric(label="Number of words", value=len(resume_txt))
    rcol2.metric(label="Unique words", value=len(convert_to_set(resume_txt)))

with st.expander("Upload or input job Description manually"):
    jd_upload = st.file_uploader("Upload Job Description")
    if jd_upload != None:
        jd_text2 = docx2txt.process(jd_upload)
        jd_text = docx2txt.process(jd_upload)
        jd_text = str(jd_text.translate(str.maketrans('', '', string.punctuation)))
        jd_text = jd_text.lower()
        jd_text = remove_stopwords(jd_text)
        jd_txt = st.text_area('Job Description Text', jd_text)
    else:
        jd_txt = st.text_area('Job Description Text')
    
    jcol1, jcol2 = st.columns(2)
    jcol1.metric(label="Number of words", value=len(jd_txt))
    jcol2.metric(label="Unique words", value=len(convert_to_set(jd_txt)))


from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from sklearn.feature_extraction. text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if resume_txt != "" and jd_txt != "":
    text_list = [resume_txt, jd_txt]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100

    st.header('Cosine Similarity (benchmark at 80%)')
    st.metric(label ='Match Perscentage' ,value=round(matchPercentage,1), delta=round(matchPercentage-80,1))

else:
    st.write("")


if resume_txt != "" and jd_txt != "":

    s1 = convert_to_set(resume_txt)
    s2 = convert_to_set(jd_txt)
    s = s2.difference(s1)
    word_found = len(s2) - len(s)

    st.header("Recommendations")
    st.write(f"Out of {len(s2)} unique words found in Job Description, {word_found} words are found in the resume. \
        You may consider optimizing your resume by including words below:")
    
    st.write(s)

    jd_keywords = rake.apply(jd_text2)
    st.write("Significance of words / phrases in job description using RAKE method")
    df = pd.DataFrame(jd_keywords)
    df.columns = ['Phrases', 'Importance']

    st.dataframe(df[:10])