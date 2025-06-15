from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import numpy as np

# 1. Essay and query
essay_sentences = [
    "Dogs are among the most loyal companions humans have ever known.",
    "They have been domesticated for thousands of years.",
    "Different breeds of dogs serve various purposes, from guarding homes to herding livestock.",
    "Golden Retrievers are gentle and great with children.",
    "Dogs are loyal animals that help humans emotionally and reduce stress.",
    "German Shepherds are intelligent and often work with police forces.",
    "Pomeranians are small but full of energy and personality.",
    "Dogs communicate through barking, wagging tails, and body posture.",
    "They have an excellent sense of smell, much better than humans.",
    "Many dogs are trained as service animals to assist people with disabilities.",
    "Some dogs also work in search and rescue operations during disasters.",
    "Dogs are social animals and thrive on attention and affection.",
    "They enjoy playing games like fetch or tug-of-war with their owners.",
    "Regular walks and physical activity are important for a dog’s health.",
    "Dogs can suffer from separation anxiety when left alone for long periods.",
    "Proper training helps dogs become well-behaved and confident.",
    "They respond well to positive reinforcement and routine.",
    "Feeding dogs a balanced diet is essential for their growth and energy.",
    "Grooming keeps their coats clean and reduces the risk of skin infections.",
    "Veterinary care and vaccinations protect them from common diseases.",
    "In return for care and love, dogs offer loyalty, protection, and unconditional friendship."
]

query = "How do pets help with anxiety?"

# 2. Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
all_texts = essay_sentences + [query]
embeddings = model.encode(all_texts)

# Normalize + PCA
normalized_embeddings = normalize(embeddings)
pca = PCA(n_components=3)
reduced = pca.fit_transform(normalized_embeddings)

# Components
x, y, z = reduced[:, 0], reduced[:, 1], reduced[:, 2]
query_vec = reduced[-1]
sentence_vecs = reduced[:-1]

# 3. Cosine similarity
similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]
max_index = np.argmax(similarities)
max_sim_score = similarities[max_index]
max_sim_vec = sentence_vecs[max_index]

# 4. Plot (use sentence text as legend)
fig = go.Figure()

for i, sentence in enumerate(essay_sentences):
    fig.add_trace(go.Scatter3d(
        x=[x[i]], y=[y[i]], z=[z[i]],
        mode='markers',
        name=sentence if len(sentence) < 50 else sentence[:47] + "...",
        marker=dict(size=5)
    ))

# Query vector marker
fig.add_trace(go.Scatter3d(
    x=[query_vec[0]], y=[query_vec[1]], z=[query_vec[2]],
    mode='markers',
    name="User Query",
    marker=dict(size=8, color='red', symbol='diamond')
))

# Line from origin to query
fig.add_trace(go.Scatter3d(
    x=[0, query_vec[0]],
    y=[0, query_vec[1]],
    z=[0, query_vec[2]],
    mode="lines",
    name="Origin → Query",
    line=dict(color="red", width=2, dash="dot")
))

# Line from origin to most similar sentence
fig.add_trace(go.Scatter3d(
    x=[0, max_sim_vec[0]],
    y=[0, max_sim_vec[1]],
    z=[0, max_sim_vec[2]],
    mode="lines",
    name="Origin → Closest Sentence",
    line=dict(color="green", width=2, dash="dash")
))

# Cosine similarity score annotation
fig.add_trace(go.Scatter3d(
    x=[(query_vec[0] + max_sim_vec[0]) / 2],
    y=[(query_vec[1] + max_sim_vec[1]) / 2],
    z=[(query_vec[2] + max_sim_vec[2]) / 2],
    mode="text",
    text=[f"CosSim: {max_sim_score:.2f}"],
    showlegend=False
))

# Layout
fig.update_layout(
    title="3D Embedding Plot of Dog Essay vs User Query",
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    ),
    legend_title="Legend",
    margin=dict(l=10, r=10, t=50, b=10)
)

# Save and show
fig.write_html("dog_embeddings_query_plot.html")
fig.show()
