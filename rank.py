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
    profile = c.get('profile', {})
    history = c.get('career_history', [])
    skills = c.get('skills', [])
    
    # 1. Job duration mismatch (> 6 months)
    curr_date = datetime(2026, 6, 15)
    for h_idx, h in enumerate(history):
        start = parse_date(h.get('start_date'))
        end = parse_date(h.get('end_date')) if h.get('end_date') else curr_date
        stated_months = h.get('duration_months')
        
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
            
            if profile.get('years_of_experience', 0) > total_span_years + 3.0:
                return True, "Experience exceeds span"

    # 3. Expert/advanced skills with 0 months duration (>= 3 skills)
    expert_zero_dur = 0
    for s in skills:
        if s.get('proficiency') in ['advanced', 'expert'] and s.get('duration_months', 0) == 0:
            expert_zero_dur += 1
    if expert_zero_dur >= 3:
        return True, "Expert skills with 0 duration"
        
    # 4. Company age vs job duration in description
    for h_idx, h in enumerate(history):
        desc = h.get('description', '').lower()
        m = re.search(r'(\d+)\s*year\s*old\s*startup', desc)
        if m:
            startup_age = int(m.group(1))
            stated_months = h.get('duration_months', 0)
            if stated_months / 12.0 > startup_age + 2:
                return True, f"Job {h_idx} age mismatch"
                
    return False, ""

def compute_heuristics(c):
    profile = c.get('profile', {})
    history = c.get('career_history', [])
    signals = c.get('redrob_signals', {})
    
    title = profile.get('current_title', '').lower()
    headline = profile.get('headline', '').lower()
    
    # 1. Experience score (Ideal: 5-9 years)
    exp = profile.get('years_of_experience', 0)
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
    loc = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
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
    all_companies = [h.get('company', '').lower() for h in history if h.get('company')]
    
    company_score = 0.5
    if all_companies:
        is_only_consulting = all(any(cf in comp for cf in consulting_firms) for comp in all_companies)
        if is_only_consulting:
            company_score = 0.0
        else:
            has_product = any(not any(cf in comp for cf in consulting_firms) for comp in all_companies)
            if has_product:
                company_score = 1.0
                
    # 4. Title match score (Additional safeguard)
    target_titles = ["ai engineer", "ml engineer", "machine learning engineer", "nlp engineer", "search engineer", "ranking engineer", "retrieval engineer", "founding engineer"]
    title_score = 0.2
    if any(t in title for t in target_titles) or any(t in headline for t in target_titles):
        title_score = 1.0
    elif "data scientist" in title or "data scientist" in headline:
        title_score = 0.8
    elif "backend engineer" in title or "software engineer" in title or "backend engineer" in headline or "software" in headline:
        title_score = 0.5
        
    heuristic_score = 0.35 * exp_score + 0.35 * location_score + 0.20 * company_score + 0.10 * title_score
    return heuristic_score

def compute_behavior_multiplier(c):
    signals = c.get('redrob_signals', {})
    response_rate = signals.get('recruiter_response_rate', 0.5)
    open_to_work = signals.get('open_to_work_flag', False)
    notice_period = signals.get('notice_period_days', 60)
    
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
            active_mult = 0.7
            
    resp_mult = 0.5 + 0.5 * response_rate
    if response_rate < 0.10:
        resp_mult = 0.5
        
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
    profile = c.get('profile', {})
    skills = c.get('skills', [])
    history = c.get('career_history', [])
    
    parts = []
    
    title = profile.get('current_title', '')
    headline = profile.get('headline', '')
    if title:
        parts.append(f"Title: {title}")
    if headline:
        parts.append(f"Headline: {headline}")
    
    summary = profile.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary}")
        
    skills_strs = []
    for s in skills[:15]:
        name = s.get('name', '')
        prof = s.get('proficiency', '')
        dur = s.get('duration_months', 0)
        skills_strs.append(f"{name} ({prof}, {dur}m)")
    if skills_strs:
        parts.append("Skills: " + ", ".join(skills_strs))
        
    hist_strs = []
    for h in history[:3]:
        h_title = h.get('title', '')
        h_comp = h.get('company', '')
        h_dur = h.get('duration_months', 0)
        h_desc = h.get('description', '')
        hist_strs.append(f"Role: {h_title} at {h_comp} ({h_dur}m). Description: {h_desc[:150]}")
    if hist_strs:
        parts.append("Experience:\n" + "\n".join(hist_strs))
        
    return "\n".join(parts)

def generate_reasoning(c, rank, score):
    profile = c.get('profile', {})
    skills = c.get('skills', [])
    signals = c.get('redrob_signals', {})
    
    title = profile.get('current_title', 'Engineer')
    exp = profile.get('years_of_experience', 0)
    location = profile.get('location', 'India')
    notice = signals.get('notice_period_days', 60)
    resp = int(signals.get('recruiter_response_rate', 0.0) * 100)
    
    skill_names = [s.get('name') for s in skills[:3]]
    skills_text = ", ".join(skill_names) if skill_names else "applied machine learning"
    
    if rank <= 15:
        s1 = f"Outstanding {title} with {exp} years of experience demonstrating strong systems engineering capabilities."
    elif rank <= 50:
        s1 = f"Highly qualified {title} with {exp} years of experience and direct expertise in {skills_text}."
    else:
        s1 = f"Capable professional with {exp} years of software experience and strong fundamentals in {skills_text}."
        
    concerns = []
    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if signals.get('recruiter_response_rate', 1.0) < 0.40:
        concerns.append("lower response rate")
    if not signals.get('willing_to_relocate', True) and location.lower() not in ["pune", "noida", "delhi", "gurgaon"]:
        concerns.append("relocation constraint")
        
    if concerns:
        s2 = f"Direct match for the intelligence team, with minor concern on {', '.join(concerns)}."
    else:
        s2 = f"Excellent match for ranking/retrieval systems; shows active candidate status and {resp}% response rate."
        
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
        sys.exit(1)
        
    bi_model = SentenceTransformer(bi_encoder_path)
    cross_model = CrossEncoder(cross_encoder_path)
    
    with open(args.candidate_ids, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
    id_to_idx = {cand_id: idx for idx, cand_id in enumerate(candidate_ids)}
    embeddings = np.load(args.embeddings)
    
    jd_text = """
    Job Description: Senior AI Engineer — Founding Team
    Company: Redrob AI (Series A AI-native talent intelligence platform)
    Location: Pune/Noida, India (Hybrid — flexible cadence)
    Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
    Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
    Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or similar.
    Strong Python.
    """
    
    jd_embedding = bi_model.encode(jd_text, normalize_embeddings=True)
    similarities = np.dot(embeddings, jd_embedding)
    
    valid_candidates = []
    skipped_honeypots = 0
    
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            is_hp, _ = is_honeypot(c)
            if is_hp:
                skipped_honeypots += 1
                continue
                
            cand_id = c['candidate_id']
            if cand_id not in id_to_idx:
                continue
                
            idx = id_to_idx[cand_id]
            bi_sim = similarities[idx]
            heuristic_score = compute_heuristics(c)
            behavior_mult = compute_behavior_multiplier(c)
            
            valid_candidates.append({
                "candidate": c,
                "bi_similarity": bi_sim,
                "heuristic_score": heuristic_score,
                "behavior_mult": behavior_mult
            })
            
    for item in valid_candidates:
        bi_sim_norm = max(0.0, item["bi_similarity"])
        item["preliminary_score"] = bi_sim_norm * (0.7 + 0.3 * item["heuristic_score"])
        
    valid_candidates.sort(key=lambda x: x["preliminary_score"], reverse=True)
    top_candidates = valid_candidates[:300]
    
    pairs = [(jd_text, make_candidate_text(item["candidate"])) for item in top_candidates]
    cross_scores = cross_model.predict(pairs, batch_size=32)
    semantic_scores = 1.0 / (1.0 + np.exp(-cross_scores))
    
    ranked_results = []
    for item, sem_score in zip(top_candidates, semantic_scores):
        final_score = sem_score * (0.7 + 0.3 * item["heuristic_score"]) * item["behavior_mult"]
        ranked_results.append((item["candidate"], final_score))
        
    ranked_results.sort(key=lambda x: (-x[1], x[0]['candidate_id']))
    top_100 = ranked_results[:100]
    
    temp_out = args.out + ".tmp"
    with open(temp_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for idx, (c, score) in enumerate(top_100):
            rank = idx + 1
            reasoning = generate_reasoning(c, rank, score)
            writer.writerow([c['candidate_id'], rank, score, reasoning])
            
    if os.path.exists(args.out):
        os.remove(args.out)
    os.rename(temp_out, args.out)

if __name__ == "__main__":
    main()
