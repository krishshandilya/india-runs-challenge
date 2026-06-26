import os
import json
import argparse
import csv
import sys
from datetime import datetime
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def is_honeypot(c):
    profile = c.get('profile') or {}
    history = c.get('career_history') or []
    skills = c.get('skills') or []
    
    # 1. Job duration mismatch (> 6 months)
    curr_date = datetime(2026, 6, 15)
    for h_idx, h in enumerate(history):
        start = parse_date(h.get('start_date'))
        end = parse_date(h.get('end_date')) if h.get('end_date') else curr_date
        stated_months = h.get('duration_months')
        if stated_months is not None:
            try:
                stated_months = int(stated_months)
            except (ValueError, TypeError):
                stated_months = None
        
        if start and end and stated_months is not None:
            calc_months = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(calc_months - stated_months) > 6:
                return True, f"Job {h_idx} duration mismatch"
                
    # 2. Experience exceeds span significantly (> 3.0 years)
    if history:
        starts = [parse_date(h.get('start_date')) for h in history if h.get('start_date')]
        ends = [parse_date(h.get('end_date')) if h.get('end_date') else curr_date for h in history]
        starts = [d for d in starts if d]
        if starts:
            earliest_start = min(starts)
            latest_end = max(ends)
            total_span_months = (latest_end.year - earliest_start.year) * 12 + (latest_end.month - earliest_start.month)
            total_span_years = total_span_months / 12.0
            
            exp = profile.get('years_of_experience')
            try:
                exp = float(exp) if exp is not None else 0.0
            except (ValueError, TypeError):
                exp = 0.0
            
            if exp > total_span_years + 3.0:
                return True, "Experience exceeds span"

    # 3. Expert/advanced skills with 0 months duration (>= 3 skills)
    expert_zero_dur = 0
    for s in skills:
        prof = s.get('proficiency')
        try:
            dur = int(s.get('duration_months') or 0)
        except (ValueError, TypeError):
            dur = 0
        if prof in ['advanced', 'expert'] and dur == 0:
            expert_zero_dur += 1
    if expert_zero_dur >= 3:
        return True, "Expert skills with 0 duration"
        
    # 4. Company age vs job duration in description
    for h_idx, h in enumerate(history):
        desc = str(h.get('description') or '').lower()
        m = re.search(r'(\d+)\s*year\s*old\s*startup', desc)
        if m:
            startup_age = int(m.group(1))
            stated_months = h.get('duration_months') or 0
            try:
                stated_months = int(stated_months)
            except (ValueError, TypeError):
                stated_months = 0
            if stated_months / 12.0 > startup_age + 2:
                return True, f"Job {h_idx} age mismatch"
                
    return False, ""

def compute_heuristics(c, target_title_emb, text_to_emb):
    profile = c.get('profile') or {}
    history = c.get('career_history') or []
    signals = c.get('redrob_signals') or {}
    
    title = profile.get('current_title') or ''
    headline = profile.get('headline') or ''
    
    # 1. Experience score (Ideal: 5-9 years)
    exp = profile.get('years_of_experience')
    try:
        exp = float(exp) if exp is not None else 0.0
    except (ValueError, TypeError):
        exp = 0.0
    exp_score = 0.1
    if 5.0 <= exp <= 9.0:
        exp_score = 1.0
    elif 4.0 <= exp < 5.0:
        exp_score = 0.85
    elif 9.0 < exp <= 12.0:
        exp_score = 0.8
    elif 3.0 <= exp < 4.0:
        exp_score = 0.6
    elif 12.0 < exp <= 15.0:
        exp_score = 0.5
    elif exp < 3.0:
        exp_score = 0.2
        
    # 2. Location score (Noida/Pune preferred)
    loc = str(profile.get('location') or '').lower()
    country = str(profile.get('country') or '').lower()
    willing_relocate = signals.get('willing_to_relocate', False)
    
    is_pune_noida = any(city in loc for city in ["pune", "noida", "delhi", "gurgaon", "ghaziabad", "faridabad"])
    tier1_india = any(city in loc for city in ["hyderabad", "bangalore", "mumbai", "chennai", "kolkata", "coimbatore"])
    
    location_score = 0.0
    if is_pune_noida:
        location_score = 1.0
    elif "india" in country or tier1_india:
        if willing_relocate:
            location_score = 0.8
        else:
            location_score = 0.3
    else:
        if willing_relocate:
            location_score = 0.4
            
    # 3. Product vs Consulting score (Disqualify/penalize TCS/Infosys etc. if entire career)
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy", "wipro technologies", "infosys limited"]
    all_companies = [str(h.get('company') or '').lower() for h in history if h.get('company')]
    
    company_score = 0.5
    if all_companies:
        is_only_consulting = all(any(cf in comp for cf in consulting_firms) for comp in all_companies)
        if is_only_consulting:
            company_score = 0.0
        else:
            # Check if they have product company experience
            has_product = any(not any(cf in comp for cf in consulting_firms) for comp in all_companies)
            if has_product:
                company_score = 1.0
                
    # 4. Semantic Title Match Heuristic
    t_emb = text_to_emb[title] if title in text_to_emb else np.zeros(384)
    h_emb = text_to_emb[headline] if headline in text_to_emb else np.zeros(384)
    
    sim_title = np.dot(t_emb, target_title_emb) if title else 0.0
    sim_headline = np.dot(h_emb, target_title_emb) if headline else 0.0
    
    # Bounded in [0, 1] representing continuous semantic similarity
    title_score = max(0.0, max(sim_title, sim_headline))
        
    heuristic_score = 0.35 * exp_score + 0.35 * location_score + 0.20 * company_score + 0.10 * title_score
    return heuristic_score

def compute_behavior_multiplier(c):
    signals = c.get('redrob_signals') or {}
    
    response_rate = signals.get('recruiter_response_rate')
    try:
        response_rate = float(response_rate) if response_rate is not None else 0.5
    except (ValueError, TypeError):
        response_rate = 0.5
        
    open_to_work = signals.get('open_to_work_flag', False)
    
    notice_period = signals.get('notice_period_days')
    try:
        notice_period = int(notice_period) if notice_period is not None else 60
    except (ValueError, TypeError):
        notice_period = 60
        
    last_act = parse_date(signals.get('last_active_date'))
    curr_date = datetime(2026, 6, 15)
    
    active_mult = 1.0
    if last_act:
        act_months = (curr_date.year - last_act.year) * 12 + (curr_date.month - last_act.month)
        if act_months <= 1:
            active_mult = 1.0
        elif act_months <= 3:
            active_mult = 0.9
        elif act_months <= 6:
            active_mult = 0.7
        else:
            active_mult = 0.7  # Relaxed from 0.3 to 0.7
            
    resp_mult = 0.5 + 0.5 * response_rate
    if response_rate < 0.10:
        resp_mult = 0.5  # Relaxed from 0.2 to 0.5
        
    otw_mult = 1.0 if open_to_work else 0.85
    
    notice_mult = 1.0
    if notice_period <= 30:
        notice_mult = 1.0
    elif notice_period <= 90:
        notice_mult = 0.9
    else:
        notice_mult = 0.75
        
    return active_mult * resp_mult * otw_mult * notice_mult


def make_candidate_text(c):
    profile = c.get('profile') or {}
    skills = c.get('skills') or []
    history = c.get('career_history') or []
    
    parts = []
    
    title = profile.get('current_title') or ''
    headline = profile.get('headline') or ''
    if title:
        parts.append(f"Title: {title}")
    if headline:
        parts.append(f"Headline: {headline}")
    
    summary = profile.get('summary') or ''
    if summary:
        parts.append(f"Summary: {summary}")
        
    skills_strs = []
    for s in skills[:15]:
        name = s.get('name') or ''
        prof = s.get('proficiency') or ''
        dur = s.get('duration_months')
        try:
            dur = int(dur) if dur is not None else 0
        except (ValueError, TypeError):
            dur = 0
        if name:
            skills_strs.append(f"{name} ({prof}, {dur}m)")
    if skills_strs:
        parts.append("Skills: " + ", ".join(skills_strs))
        
    hist_strs = []
    for h in history[:3]:
        h_title = h.get('title') or ''
        h_comp = h.get('company') or ''
        dur = h.get('duration_months')
        try:
            dur = int(dur) if dur is not None else 0
        except (ValueError, TypeError):
            dur = 0
        h_desc = str(h.get('description') or '')
        hist_strs.append(f"Role: {h_title} at {h_comp} ({dur}m). Description: {h_desc[:150]}")
    if hist_strs:
        parts.append("Experience:\n" + "\n".join(hist_strs))
        
    return "\n".join(parts)

def generate_reasoning(c, rank, score):
    profile = c.get('profile') or {}
    skills = c.get('skills') or []
    signals = c.get('redrob_signals') or {}
    history = c.get('career_history') or []
    
    title = profile.get('current_title') or 'Engineer'
    exp = profile.get('years_of_experience')
    try:
        exp = float(exp) if exp is not None else 0.0
    except (ValueError, TypeError):
        exp = 0.0
        
    location = profile.get('location') or 'India'
    
    notice = signals.get('notice_period_days')
    try:
        notice = int(notice) if notice is not None else 60
    except (ValueError, TypeError):
        notice = 60
        
    resp_rate = signals.get('recruiter_response_rate')
    try:
        resp_rate = float(resp_rate) if resp_rate is not None else 0.5
    except (ValueError, TypeError):
        resp_rate = 0.5
    resp = int(resp_rate * 100)
    
    open_to_work = signals.get('open_to_work_flag', False)
    
    # 1. Company background
    all_companies = [str(h.get('company') or '').lower() for h in history if h.get('company')]
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy", "wipro technologies", "infosys limited"]
    is_only_consulting = all_companies and all(any(cf in comp for cf in consulting_firms) for comp in all_companies)
    company_desc = "consulting background" if is_only_consulting else "product background"
    
    # 2. Title similarity evaluation
    relevant_keywords = ["ai", "ml", "machine learning", "deep learning", "data scien", "nlp", "search", "ranking", "retrieval", "applied scientist"]
    title_lower = title.lower()
    has_high_sim = any(kw in title_lower for kw in relevant_keywords)
    title_desc = f"high title similarity ({title})" if has_high_sim else f"semantic alignment to target profile"
    
    # 3. Experience range description
    exp_desc = f"ideal {exp} years of experience" if 5.0 <= exp <= 9.0 else f"{exp} years of experience"
    
    s1 = f"Strong overall score driven by {title_desc}, {exp_desc}, and a {company_desc}."
    
    # 4. Location evaluations
    loc_lower = str(location).lower()
    is_local = any(city in loc_lower for city in ["pune", "noida", "delhi", "gurgaon", "ghaziabad", "faridabad"])
    willing_relocate = signals.get('willing_to_relocate', False)
    
    # 5. Behavioral signals
    behavior_parts = []
    if open_to_work:
        behavior_parts.append("active open-to-work status")
    else:
        behavior_parts.append("passive search status")
        
    if notice <= 30:
        behavior_parts.append(f"prompt {notice}-day notice")
    else:
        behavior_parts.append(f"{notice}-day notice period")
        
    if resp_rate >= 0.8:
        behavior_parts.append(f"high {resp}% response rate")
    elif resp_rate < 0.4:
        behavior_parts.append(f"low response rate ({resp}%)")
    else:
        behavior_parts.append(f"moderate {resp}% response rate")
        
    behavior_str = ", ".join(behavior_parts)
    
    if is_local:
        s2 = f"Placement is supported by a local {location} location boost and favorable behavioral signals ({behavior_str})."
    else:
        if willing_relocate:
            s2 = f"Placement is adjusted by a relocation requirement from {location}, balanced by favorable behavioral signals ({behavior_str})."
        else:
            s2 = f"Placement is constrained by a relocation requirement from {location} and behavioral signals ({behavior_str})."
            
    return f"{s1} {s2}"


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write ranked submission CSV")
    parser.add_argument("--models-dir", default="./models", help="Directory containing local models")
    parser.add_argument("--embeddings", default="./embeddings.npy", help="Path to precomputed embeddings")
    parser.add_argument("--candidate-ids", default="./candidate_ids.json", help="Path to candidate IDs JSON mapping")
    args = parser.parse_args()
    
    bi_encoder_path = os.path.join(args.models_dir, "bi_encoder")
    cross_encoder_path = os.path.join(args.models_dir, "cross_encoder")
    
    if not os.path.exists(args.embeddings) or not os.path.exists(args.candidate_ids):
        print(f"Error: Precomputed embeddings ({args.embeddings}) or mapping ({args.candidate_ids}) not found.")
        print("Please run precompute.py first.")
        sys.exit(1)
        
    print("Loading local Bi-Encoder...")
    bi_model = SentenceTransformer(bi_encoder_path)
    
    print("Loading local Cross-Encoder...")
    cross_model = CrossEncoder(cross_encoder_path)
    
    print("Loading precomputed candidate mappings...")
    with open(args.candidate_ids, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
        
    id_to_idx = {cand_id: idx for idx, cand_id in enumerate(candidate_ids)}
    
    print("Loading candidate embeddings...")
    embeddings = np.load(args.embeddings)
    
    jd_text = """
    Job Description: Senior AI Engineer — Founding Team
    Company: Redrob AI (Series A AI-native talent intelligence platform)
    Location: Pune/Noida, India (Hybrid — flexible cadence)
    Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
    Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
    Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar.
    Strong Python.
    Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.
    """
    
    print("Embedding Job Description...")
    jd_embedding = bi_model.encode(jd_text, normalize_embeddings=True)
    
    print("Computing bi-encoder similarity scores...")
    similarities = np.dot(embeddings, jd_embedding)
    
    target_title_str = "Senior AI Engineer"
    print(f"Pre-encoding target title for heuristics: '{target_title_str}'")
    target_title_emb = bi_model.encode(target_title_str, normalize_embeddings=True)
    
    print("Collecting unique candidate titles and headlines...")
    unique_texts = set()
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            is_hp, _ = is_honeypot(c)
            if is_hp:
                continue
            cand_id = c['candidate_id']
            if cand_id not in id_to_idx:
                continue
            profile = c.get('profile') or {}
            title = profile.get('current_title') or ''
            headline = profile.get('headline') or ''
            if title:
                unique_texts.add(title)
            if headline:
                unique_texts.add(headline)
                
    if unique_texts:
        print(f"Embedding {len(unique_texts)} unique titles and headlines...")
        unique_list = list(unique_texts)
        unique_embs = bi_model.encode(unique_list, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        text_to_emb = {text: unique_embs[idx] for idx, text in enumerate(unique_list)}
    else:
        text_to_emb = {}

    valid_candidates = []
    skipped_honeypots = 0
    
    print("Parsing candidate metadata and running heuristic filters...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            
            # Skip honeypots immediately
            is_hp, _ = is_honeypot(c)
            if is_hp:
                skipped_honeypots += 1
                continue
                
            cand_id = c['candidate_id']
            if cand_id not in id_to_idx:
                continue
                
            idx = id_to_idx[cand_id]
            bi_sim = similarities[idx]
            
            heuristic_score = compute_heuristics(c, target_title_emb, text_to_emb)
            behavior_mult = compute_behavior_multiplier(c)
            
            valid_candidates.append({
                "candidate": c,
                "bi_similarity": bi_sim,
                "heuristic_score": heuristic_score,
                "behavior_mult": behavior_mult
            })
            
    print(f"Skipped {skipped_honeypots} honeypot profiles.")
    print(f"Processing {len(valid_candidates)} valid candidates.")
    
    # Calculate a baseline retrieval score using multiplicative heuristics
    for item in valid_candidates:
        bi_sim_norm = max(0.0, item["bi_similarity"])
        item["preliminary_score"] = bi_sim_norm * (0.7 + 0.3 * item["heuristic_score"])
        
    # Retrieve top 500 candidates based on preliminary score for re-ranking
    valid_candidates.sort(key=lambda x: x["preliminary_score"], reverse=True)
    top_candidates = valid_candidates[:500]
    
    print("Running Cross-Encoder re-ranking on top 500 candidates...")
    if not top_candidates:
        print("Error: No valid candidates found to re-rank.")
        sys.exit(1)
    pairs = [(jd_text, make_candidate_text(item["candidate"])) for item in top_candidates]
    cross_scores = cross_model.predict(pairs, batch_size=32)
    
    # Sigmoid normalization of raw logits (clipped for numerical stability)
    semantic_scores = 1.0 / (1.0 + np.exp(-np.clip(cross_scores, -20.0, 20.0)))
    
    # Calculate final scores for the top 500 using multiplicative heuristics
    ranked_results = []
    for item, sem_score in zip(top_candidates, semantic_scores):
        final_score = sem_score * (0.7 + 0.3 * item["heuristic_score"]) * item["behavior_mult"]
        ranked_results.append((item["candidate"], final_score))

        
    # Sort top candidates by final score descending, and then candidate_id ascending for deterministic tie-breaker
    ranked_results.sort(key=lambda x: (-x[1], x[0]['candidate_id']))
    
    # Select the final top 100
    top_100 = ranked_results[:100]
    
    # Write to target CSV using a temporary file fallback to handle OS file locks
    temp_out = args.out + ".tmp"
    print(f"Writing top 100 candidates to temporary file {temp_out}...")
    with open(temp_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, (c, score) in enumerate(top_100):
            rank = idx + 1
            reasoning = generate_reasoning(c, rank, score)
            writer.writerow([c['candidate_id'], rank, score, reasoning])
            
    try:
        if os.path.exists(args.out):
            os.remove(args.out)
        os.rename(temp_out, args.out)
        print(f"Successfully ranked candidates and wrote output to {args.out}")
    except PermissionError as e:
        print(f"Error: Permission denied writing to '{args.out}'. Details: {e}", file=sys.stderr)
        print(f"Temporary output has been saved to: '{temp_out}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
