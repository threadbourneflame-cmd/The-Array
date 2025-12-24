import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 1. Load the Brain
model = SentenceTransformer('all-MiniLM-L6-v2')

def analyze_trinity(text_a, label_a, text_b, label_b, text_c, label_c):
    # FALLBACK LABELS (If user leaves them blank)
    if not label_a.strip(): label_a = "Input A"
    if not label_b.strip(): label_b = "Input B"
    if not label_c.strip(): label_c = "Input C"
    
    clean_labels = [label_a, label_b, label_c]
    texts = [text_a.strip(), text_b.strip(), text_c.strip()]
    
    # Validation
    if not any(texts):
        return pd.DataFrame(), "Waiting for signals...", "Waiting for signals..."

    # --- PART 1: THE ALIGNMENT MATRIX (With Custom Names) ---
    # We use embeddings to calculate how 'close' each text is to the others
    embeddings = model.encode(texts)
    matrix = cosine_similarity(embeddings)
    
    # Create the DataFrame with YOUR custom names
    df_matrix = pd.DataFrame(matrix, columns=clean_labels, index=clean_labels)
    df_matrix = df_matrix.round(3)

    # --- PART 2: FINGERPRINTS (Unique Vibe) ---
    # TF-IDF to find words unique to each specific input
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        
        signatures = ""
        for i, label in enumerate(clean_labels):
            if not texts[i]: continue # Skip empty boxes
            
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-5:][::-1]
            top_words = feature_names[top_indices]
            
            # Only keep words with actual weight
            valid_words = [w for w, idx in zip(top_words, top_indices) if row[idx] > 0]
            
            signatures += f"🔹 {label.upper()}: {', '.join(valid_words)}\n"
            
    except:
        signatures = "Not enough data for fingerprinting."

    # --- PART 3: THE CONSENSUS (Universal vs Majority) ---
    vectorizer = CountVectorizer(stop_words='english')
    try:
        dtm = vectorizer.fit_transform(texts)
        vocab = vectorizer.get_feature_names_out()
        presence = (dtm.toarray() > 0).astype(int)
        
        # UNIVERSAL: Present in all 3 (Sum = 3)
        univ_indices = np.where(presence.sum(axis=0) == 3)[0]
        univ_words = vocab[univ_indices]
        
        # MAJORITY: Present in 2 (Sum = 2)
        maj_indices = np.where(presence.sum(axis=0) == 2)[0]
        maj_words = vocab[maj_indices]
        
        consensus_text = ""
        
        if len(univ_words) > 0:
            consensus_text += f"🔥 UNIVERSAL TRUTH (3/3): {', '.join(univ_words)}\n\n"
        else:
            consensus_text += "❌ NO UNIVERSAL TRUTH FOUND.\n\n"
            
        if len(maj_words) > 0:
            consensus_text += f"⚠️ MAJORITY REPORT (2/3): {', '.join(maj_words)}"
        else:
            consensus_text += "⚠️ NO PARTIAL ALIGNMENT FOUND."
            
    except:
        consensus_text = "Waiting for more data..."

    return df_matrix, signatures, consensus_text

# --- UI BUILD (Flexible Inputs) ---
with gr.Blocks(theme=gr.themes.Glass()) as app:
    gr.Markdown("# 🕸️ THE ARRAY v0.2")
    gr.Markdown("### FlameTeam Multi-Model Alignment System")
    
    with gr.Row():
        # COLUMN 1
        with gr.Column():
            lbl_a = gr.Textbox(label="Label 1", value="Me (Prompt)", placeholder="Name this input...")
            box_a = gr.TextArea(show_label=False, placeholder="Paste text here...", lines=4)
        # COLUMN 2
        with gr.Column():
            lbl_b = gr.Textbox(label="Label 2", value="Model A", placeholder="Name this input...")
            box_b = gr.TextArea(show_label=False, placeholder="Paste text here...", lines=4)
        # COLUMN 3
        with gr.Column():
            lbl_c = gr.Textbox(label="Label 3", value="Model B", placeholder="Name this input...")
            box_c = gr.TextArea(show_label=False, placeholder="Paste text here...", lines=4)
        
    btn = gr.Button("RUN THE ARRAY", variant="primary")
    
    # RESULTS
    gr.Markdown("### 1. The Alignment Matrix")
    out_matrix = gr.Dataframe(label="Cosine Similarity Grid")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 2. Unique Fingerprints")
            out_signatures = gr.Textbox(label="Distinctive Terms (TF-IDF)", lines=5)
        with gr.Column():
            gr.Markdown("### 3. The Consensus")
            out_consensus = gr.Textbox(label="Shared Concepts", lines=5)

    btn.click(analyze_trinity, inputs=[box_a, lbl_a, box_b, lbl_b, box_c, lbl_c], outputs=[out_matrix, out_signatures, out_consensus])

app.launch()
