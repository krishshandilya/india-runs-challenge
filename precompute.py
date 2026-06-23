import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

def make_candidate_text(c):
    profile = c.get('profile', {})
    skills = c.get('skills', [])
    
    parts = []
    
    # 1. Headline and Title
    headline = profile.get('headline', '')
    title = profile.get('current_title', '')
    if title:
        parts.append(title)
    if headline:
        parts.append(headline)
        
    # 2. Skills
    skills_strs = [s.get('name', '') for s in skills if s.get('name')]
    if skills_strs:
        parts.append("Skills: " + ", ".join(skills_strs))
        
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings and save models locally.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--models-dir", default="./models", help="Directory to save local models")
    parser.add_argument("--out-embeddings", default="./embeddings.npy", help="Path to save candidate embeddings")
    parser.add_argument("--out-ids", default="./candidate_ids.json", help="Path to save candidate IDs mapping")
    args = parser.parse_args()
    
    # Create directories if they do not exist
    os.makedirs(args.models_dir, exist_ok=True)
    bi_encoder_path = os.path.join(args.models_dir, "bi_encoder")
    cross_encoder_path = os.path.join(args.models_dir, "cross_encoder")
    
    print("Downloading and saving Bi-Encoder model...")
    bi_model = SentenceTransformer('all-MiniLM-L6-v2')
    bi_model.save(bi_encoder_path)
    
    print("Downloading and saving Cross-Encoder model...")
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_model.save(cross_encoder_path)
    # Safe secondary save
    cross_model.model.save_pretrained(cross_encoder_path)
    cross_model.tokenizer.save_pretrained(cross_encoder_path)
    
    candidate_ids = []
    candidate_texts = []
    
    print(f"Reading candidates from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            c = json.loads(line)
            candidate_ids.append(c['candidate_id'])
            candidate_texts.append(make_candidate_text(c))
            
    print(f"Computing embeddings for {len(candidate_texts)} candidates...")
    # Using a larger batch size for faster embedding on CPU / GPU if available
    embeddings = bi_model.encode(
        candidate_texts, 
        batch_size=256, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    print(f"Saving embeddings to {args.out_embeddings}...")
    np.save(args.out_embeddings, embeddings)
    
    print(f"Saving candidate IDs to {args.out_ids}...")
    with open(args.out_ids, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
        
    print("Precomputation completed successfully!")

if __name__ == "__main__":
    main()
