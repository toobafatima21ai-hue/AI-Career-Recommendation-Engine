# AI-Career-Recommendation-Engine
AI Career Path Recommendation System

An intelligent content-based recommendation system that suggests suitable career paths based on user-provided technical skills and interests. The system leverages Natural Language Processing concepts, Vectorization, and Cosine Similarity to compute similarity between user profiles and predefined career profiles.

🚀 Features
 -AI-based career matching using Cosine Similarity
 -Content-based filtering approach (no external dataset required)
 -Personalized career recommendations based on:
Technical skills
Domain interests
Experience level
📚 Built-in career knowledge base (AI, Data Science, DevOps, Software Engineering)
💻 Interactive command-line interface
🔁 Multi-session support (run multiple recommendations in one session)
🛠️ Tech Stack
Python 
Scikit-learn
CountVectorizer (Vectorization)
Cosine Similarity
Object-based recommendation logic
# How It Works
User enters their skills (e.g., Python, SQL, Machine Learning)
System converts skills into a vector representation
Each career profile is also vectorized
Cosine similarity is used to compute match scores
Top matching careers are displayed in ranked order
Example Output
#1 Data Scientist → 92.5% match
#2 Machine Learning Engineer → 87.3% match
#3 AI Engineer → 81.6% match
🎯 Use Case
Helps students explore suitable tech careers
Useful for AI/ML beginners building recommendation systems
Demonstrates real-world application of NLP + ML fundamentals
graph TD;
HOW IT WORKS:
User Skills
     ↓
Text Vectorization (CountVectorizer)
     ↓
Cosine Similarity Calculation
     ↓
Similarity Scoring Engine
     ↓
Ranked Career Recommendations


# =============================================================================
# DecodeLabs | Project 3
# AI Career Path Recommendation System
# Method: Content-Based Filtering + Cosine Similarity
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# CAREER DATABASE
# =============================================================================

CAREERS = [

{
    "career":"Data Scientist",
    "skills":"python sql statistics machine_learning data_analysis"
},

{
    "career":"Machine Learning Engineer",
    "skills":"python machine_learning tensorflow pytorch deep_learning"
},

{
    "career":"AI Engineer",
    "skills":"python ai deep_learning nlp tensorflow"
},

{
    "career":"Computer Vision Engineer",
    "skills":"python opencv cnn deep_learning image_processing"
},

{
    "career":"NLP Engineer",
    "skills":"python nlp transformers llm deep_learning"
},

{
    "career":"Data Analyst",
    "skills":"sql excel power_bi python analytics"
},

{
    "career":"Backend Developer",
    "skills":"python django sql api backend"
},

{
    "career":"DevOps Engineer",
    "skills":"linux docker kubernetes aws cloud"
}

]

# =============================================================================
# INPUT
# =============================================================================

def collect_user_profile():

    print("\n" + "="*60)
    print("AI CAREER RECOMMENDATION SYSTEM")
    print("="*60)

    # ---------------------------------------------------------
    # Career Profile Setup Message
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CAREER RECOMMENDATION PROFILE SETUP")
    print("=" * 60)

    print("""
Provide the technical skills you are familiar with.
These skills will be analyzed using Content-Based
Filtering and Cosine Similarity to identify career
paths that best match your profile.

Supported Domains:

• Artificial Intelligence & Machine Learning
• Data Science & Analytics
• Software Development
• Cloud Computing & DevOps

Sample Skills:

AI & Machine Learning:
  python, machine learning, deep learning,
  tensorflow, pytorch, nlp, transformers,
  computer vision, opencv, cnn

Data Science:
  sql, statistics, pandas, numpy,
  data analysis, power bi, excel

Software Development:
  python, java, c++, django,
  flask, api, backend

Cloud & DevOps:
  linux, docker, kubernetes,
  aws, cloud, automation

Example Input:
python, machine learning, sql
""")

    user_skills = input(
        "\nEnter your skills (comma separated): "
    ).lower()

    skills = [
        s.strip().replace(" ", "_")
        for s in user_skills.split(",")
        if s.strip()
    ]

    return " ".join(skills)
# =============================================================================
# PROCESS
# =============================================================================

def recommend_careers(user_profile):

    career_names = []

    documents = []

    documents.append(user_profile)

    for career in CAREERS:

        documents.append(
            career["skills"]
        )

        career_names.append(
            career["career"]
        )

    # Vector Mapping
    vectorizer = CountVectorizer()

    vectors = vectorizer.fit_transform(
        documents
    )

    # Cosine Similarity
    similarities = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    )[0]

    results = []

    for i, score in enumerate(similarities):

        results.append(
            (
                career_names[i],
                score
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results

# =============================================================================
# OUTPUT
# =============================================================================

def display_results(results):

    print("\n" + "="*60)
    print("TOP CAREER RECOMMENDATIONS")
    print("="*60)

    for rank, (career, score) in enumerate(results[:5], start=1):

        percentage = score * 100

        bar = "█" * int(percentage / 5)

        print(
            f"\n#{rank} {career}"
        )

        print(
            f"Match: {percentage:.2f}%"
        )

        print(
            f"[{bar}]"
        )

# =============================================================================
# MAIN
# =============================================================================

def run():

    while True:

        profile = collect_user_profile()

        results = recommend_careers(profile)

        display_results(results)

        again = input(
            "\nWould you like another career recommendation? (yes/no): "
        ).strip().lower()

        if again in ["yes", "y"]:
            continue

        print("\nThank you for using the AI Career Recommendation System.")
        print("=" * 60)
        break

if __name__ == "__main__":
    run()
