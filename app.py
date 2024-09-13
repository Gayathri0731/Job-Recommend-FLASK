from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and process the job data
job_data = pd.read_csv('Data_job1.csv')
job_data['Index'] = job_data.index
sel_features = ['Company Name','Job Description','Location','Position','Required Courses']
combined_features = job_data['Company Name']+' '+job_data['Job Description']+' '+job_data['Location']+' '+job_data['Position']+' '+job_data['Required Courses']
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vector)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    job_name = request.form['job_title']
    list_of_all_titles = job_data['Job Title'].tolist()

    find_close_match = difflib.get_close_matches(job_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_job = job_data[job_data['Job Title'] == close_match]['Index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_job]))
        sorted_similar_job = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        recommended_jobs = []
        i = 1
        for jobs in sorted_similar_job:
            index = jobs[0]
            job_row = job_data.iloc[index]
            if i < 11:
                recommended_jobs.append(job_row)
                i += 1
        
        return render_template('index.html', recommendations=recommended_jobs, input_job=job_name)
    else:
        return render_template('index.html', recommendations=[], input_job=job_name)

if __name__ == '__main__':
    app.run(debug=True)
