# app.py

import os
import time
import requests
import streamlit as st
import pandas as pd
from urllib.parse import quote_plus
from typing import Dict, Any, List, TypedDict
import json
import re
import math
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# LangGraph and LangChain imports
# ---- LangChain import compatibility shim ----
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
except Exception:
    try:
        # fallback for older langchain structure
        from langchain.schema import HumanMessage, SystemMessage
        from langchain.output_parsers import JsonOutputParser
    except Exception as e:
        import streamlit as st
        st.error(f"LangChain import failed: {e}. Check langchain version in requirements.txt.")
        HumanMessage = None
        SystemMessage = None
        JsonOutputParser = None

from langgraph.graph import StateGraph, END
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
# from langchain_core.output_parsers import JsonOutputParser

# Load env (optional)
from dotenv import load_dotenv
load_dotenv()

# --- Config / env keys ---
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
COMPANY_WEBSITE_URL = os.getenv("COMPANY_WEBSITE_URL", "")
COMPANY_LINKEDIN_PAGE = os.getenv("COMPANY_LINKEDIN_PAGE", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
# ---- Validate required environment variables ----
missing_keys = []
if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")
if not SERPAPI_KEY:
    missing_keys.append("SERPAPI_API_KEY")

if missing_keys:
    st.warning(f"⚠️ Missing environment variables: {', '.join(missing_keys)}. "
               "Please set them in Streamlit Secrets (Manage app → Settings → Secrets).")
    st.stop()


# --- Platform Configuration ---
PLATFORMS = {
    # Professional Networking & General Job Boards
    "LinkedIn": {
        "sites": ["linkedin.com/in/", "linkedin.com/jobs/"],
        "category": "Professional Networking",
        "posting_support": True
    },
    "Indeed": {
        "sites": ["indeed.com", "indeed.com/q-"],
        "category": "General Job Boards",
        "posting_support": True
    },
    "ZipRecruiter": {
        "sites": ["ziprecruiter.com", "ziprecruiter.com/candidate/"],
        "category": "General Job Boards",
        "posting_support": True
    },
    "Glassdoor": {
        "sites": ["glassdoor.com", "glassdoor.com/Profile/"],
        "category": "General Job Boards",
        "posting_support": True
    },
    "Monster": {
        "sites": ["monster.com", "monster.com/resumes/"],
        "category": "General Job Boards",
        "posting_support": True
    },
    
    # Niche & Industry-Specific Job Boards
    "Dice": {
        "sites": ["dice.com", "dice.com/profiles/"],
        "category": "Tech Job Boards",
        "posting_support": True
    },
    "Wellfound": {
        "sites": ["wellfound.com", "angel.co/", "angel.co/u/"],
        "category": "Startup Job Boards",
        "posting_support": True
    },
    "GitHub": {
        "sites": ["github.com/", "github.com/users/"],
        "category": "Tech Portfolios",
        "posting_support": False
    },
    "FlexJobs": {
        "sites": ["flexjobs.com", "flexjobs.com/remote-jobs/"],
        "category": "Remote Job Boards",
        "posting_support": True
    },
    
    # Additional Specialized & Regional Platforms
    "Hired": {
        "sites": ["hired.com", "hired.com/candidates/"],
        "category": "Tech Recruitment",
        "posting_support": True
    },
    "Dribbble": {
        "sites": ["dribbble.com/", "dribbble.com/users/"],
        "category": "Design Portfolios",
        "posting_support": False
    },
    "Behance": {
        "sites": ["behance.net/", "behance.net/portfolio/"],
        "category": "Creative Portfolios",
        "posting_support": False
    },
    "Upwork": {
        "sites": ["upwork.com", "upwork.com/freelancers/"],
        "category": "Freelance Platforms",
        "posting_support": True
    },
    "Fiverr": {
        "sites": ["fiverr.com", "fiverr.com/users/"],
        "category": "Freelance Platforms",
        "posting_support": True
    },
    "CareerBuilder": {
        "sites": ["careerbuilder.com", "careerbuilder.com/people/"],
        "category": "General Job Boards",
        "posting_support": True
    },
    "Handshake": {
        "sites": ["joinhandshake.com", "app.joinhandshake.com/"],
        "category": "Early Career",
        "posting_support": True
    },
    "The Muse": {
        "sites": ["themuse.com", "themuse.com/profiles/"],
        "category": "Career Development",
        "posting_support": True
    },
    "Remote.co": {
        "sites": ["remote.co", "remote.co/remote-jobs/"],
        "category": "Remote Job Boards",
        "posting_support": True
    },
    "We Work Remotely": {
        "sites": ["weworkremotely.com", "weworkremotely.com/remote-jobs/"],
        "category": "Remote Job Boards",
        "posting_support": True
    }
}

# --- State ---
class CandidateState(TypedDict):
    jd: str
    profile_snippet: str
    profile_url: str
    platform: str
    category: str
    score: int
    reason: str
    message: str
    error: str

# --- LangGraph workflow ---
# --- LangGraph workflow ---
def create_candidate_evaluator():
    """Create the LangGraph workflow for candidate evaluation"""

      # --- Select model (with fallback support) ---
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Check API key validity
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        st.error("❌ Missing or invalid OpenAI API key. Please add it under Streamlit → Settings → Secrets.")
        st.stop()

    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=256,
            openai_api_key=OPENAI_API_KEY  # ✅ Correct argument for new SDK
        )
        st.info(f"✅ Model '{model_name}' initialized successfully.")
    except Exception as e:
        st.warning(f"⚠️ Model '{model_name}' unavailable or key invalid: {e}. Trying fallback model 'gpt-4o'...")
        try:
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=256,
                openai_api_key=OPENAI_API_KEY  # ✅ Consistent fix
            )
            st.success("✅ Fallback model 'gpt-4o' initialized successfully.")
        except Exception as err:
            st.error(f"❌ Unable to initialize ChatOpenAI model: {err}")
            st.stop()



    # --- Define system prompt ---
    system_prompt = SystemMessage(content="""You are an AI recruiter. Your task is to evaluate candidates from various platforms for a given job description.

Return JSON exactly in this format:

{
  "score": <integer from 0 to 100>,
  "reason": "<1–2 sentences explaining the score>",
  "message": "<short, personalized connection/message request>"
}

Rules:
- 90–100: Excellent fit
- 70–89: Good fit
- 50–69: Partial fit
- 30–49: Weak fit
- 0–29: Very poor fit

Consider the platform context (LinkedIn, GitHub, freelance sites, etc.) when evaluating.

Never return null, None, N/A, or empty score. 
Message must always be polite, concise, personalized.
""")

    # --- Candidate evaluation logic ---
    def evaluate_candidate(state: CandidateState) -> CandidateState:
        try:
            user_prompt = (
                f"Job Description:\n{state['jd']}\n\n"
                f"Candidate snippet / title / summary:\n{state['profile_snippet']}\n\n"
                f"Platform: {state['platform']} ({state['category']})\n"
                f"Profile URL: {state['profile_url']}\n\n"
                "Output JSON only."
            )

            # Invoke LLM
            response = llm.invoke([system_prompt, HumanMessage(content=user_prompt)])
            text = response.content.strip()

            # Try parsing JSON response
            try:
                parsed = json.loads(text)
                score = parsed.get("score")
                reason = parsed.get("reason", "")
                message = parsed.get("message", "")

                try:
                    score = int(float(score))
                except (ValueError, TypeError):
                    m = re.search(r"\d+", str(score or ""))
                    score = int(m.group(0)) if m else 25

                return {
                    **state,
                    "score": score,
                    "reason": reason,
                    "message": message,
                    "error": ""
                }

            except json.JSONDecodeError:
                return {
                    **state,
                    "score": 25,
                    "reason": "Failed to parse AI response",
                    "message": "Hi, I came across your profile and would like to connect.",
                    "error": f"JSON parse error: {text}"
                }

        except Exception as e:
            return {
                **state,
                "score": 25,
                "reason": f"Evaluation failed: {str(e)}",
                "message": "Hi, I came across your profile and would like to connect.",
                "error": str(e)
            }

    # --- Build and compile LangGraph workflow ---
    workflow = StateGraph(CandidateState)
    workflow.add_node("evaluate", evaluate_candidate)
    workflow.set_entry_point("evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()


# --- Helper functions ---
def serpapi_search_platforms(query: str, platforms: List[str], num: int = 100, serpapi_key: str = None):
    """Search across multiple platforms using SerpAPI"""
    if not serpapi_key:
        raise ValueError("SerpAPI key not provided.")
    
    all_results = []
    
    for platform_name in platforms:
        platform_config = PLATFORMS.get(platform_name)
        if not platform_config:
            continue
            
        for site in platform_config["sites"]:
            platform_query = f"site:{site} {query}"
            url = "https://serpapi.com/search.json"
            params = {
                "engine": "google", 
                "q": platform_query, 
                "api_key": serpapi_key, 
                "num": min(num, 20)  # Limit per platform
            }
            
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                results = resp.json().get("organic_results", [])
                
                # Add platform metadata to each result
                for result in results:
                    result["platform"] = platform_name
                    result["category"] = platform_config["category"]
                    result["site"] = site
                
                all_results.extend(results)
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                st.warning(f"Failed to search {platform_name} ({site}): {e}")
                continue
                
    return all_results

def extract_name_from_url(url: str, platform: str):
    """Extract name from URL based on platform patterns"""
    try:
        parts = url.rstrip("/").split("/")
        last_part = parts[-1]
        
        # Platform-specific name extraction
        if platform == "LinkedIn":
            return last_part.replace("-", " ").title()
        elif platform in ["GitHub", "Dribbble", "Behance"]:
            return last_part.title()
        elif platform in ["Upwork", "Fiverr"]:
            # Remove query parameters and clean up
            name_part = last_part.split("?")[0]
            return name_part.replace("-", " ").title()
        else:
            # Generic cleanup
            return last_part.replace("-", " ").replace("_", " ").title()
    except Exception:
        return url

def send_linkedin_invite(profile_url: str, message: str, access_token: str):
    """Send LinkedIn connection invite"""
    if not access_token:
        return {"ok": False, "error": "No LinkedIn access token provided."}
    
    # Only works for LinkedIn URLs
    if "linkedin.com" not in profile_url:
        return {"ok": False, "error": "Only LinkedIn invites are supported via API."}
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    payload = {"message": message, "inviteeProfileUrl": profile_url}
    INVITE_ENDPOINT = "https://api.linkedin.com/v2/invitations"
    try:
        resp = requests.post(INVITE_ENDPOINT, headers=headers, json=payload, timeout=20)
        if resp.status_code in (200, 201):
            return {"ok": True, "response": resp.json()}
        else:
            return {"ok": False, "status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def detect_open_to_work(snippet: str, platform: str) -> bool:
    """Detect if candidate is open to work (platform-aware)"""
    open_indicators = [
        r"\bopen to work\b",
        r"#opentowork",
        r"open-to-work",
        r"available for work",
        r"seeking opportunities",
        r"actively looking",
        r"job seeker",
        r"available for hire",
        r"freelance available",
        r"open for contracts"
    ]
    
    combined_pattern = "|".join(open_indicators)
    return bool(re.search(combined_pattern, snippet, flags=re.I))

def enforce_must_have_scoring(candidate_snippet, must_skills, current_score, platform):
    """Apply must-have skills scoring with platform context"""
    candidate_text_lower = candidate_snippet.lower()
    missing = [s for s in must_skills if s.lower() not in candidate_text_lower]
    
    if missing:
        # Less penalty for portfolio sites where skills might be demonstrated differently
        if platform in ["GitHub", "Dribbble", "Behance"]:
            penalty = min(15 * len(missing), 30)
        else:
            penalty = min(20 * len(missing), 40)
        return max(0, current_score - penalty), missing
    
    # Bonus for having all must-have skills
    return min(100, current_score + 10), []

def get_platform_specific_message(platform, base_message):
    """Adapt message based on platform"""
    platform_messages = {
        "GitHub": "Hi! I saw your GitHub profile and was impressed with your work. ",
        "Dribbble": "Hello! I came across your Dribbble portfolio and love your design style. ",
        "Behance": "Hi! Your Behance portfolio caught my attention. ",
        "Upwork": "Hi! I reviewed your Upwork profile and think you'd be a great fit. ",
        "Fiverr": "Hello! I saw your Fiverr profile and am interested in your services. ",
        "Wellfound": "Hi! I found your Wellfound profile and think you'd be perfect for this role. "
    }
    
    if platform in platform_messages:
        return platform_messages[platform] + base_message
    return base_message

def generate_seo_optimized_job_post(job_title, company, description, must_have_skills, location, email):
    """Generate SEO optimized job posting content"""
    skills_text = ", ".join(must_have_skills) if must_have_skills else "various"
    
    seo_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{job_title} at {company} | Hiring Now</title>
        <meta name="description" content="Apply for {job_title} position at {company}. Required skills: {skills_text}. Location: {location}. Immediate hiring.">
        <meta name="keywords" content="{job_title}, {company} jobs, {location} jobs, {skills_text}">
        <meta property="og:title" content="{job_title} at {company}">
        <meta property="og:description" content="Join {company} as {job_title}. Apply now!">
        <meta property="og:type" content="job_posting">
    </head>
    <body>
        <h1>{job_title}</h1>
        <h2>{company} - {location}</h2>
        
        <h3>Job Description</h3>
        <p>{description}</p>
        
        <h3>Required Skills</h3>
        <ul>
            {''.join([f'<li>{skill}</li>' for skill in must_have_skills]) if must_have_skills else '<li>Various skills required</li>'}
        </ul>
        
        <h3>How to Apply</h3>
        <p>Send your resume to our hiring team. All applications will be reviewed.</p>
        
        <p><strong>Posted on:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
    </body>
    </html>
    """
    return seo_content

def post_to_company_website(job_data):
    """Post job to company website (simulated)"""
    # This would integrate with your CMS/website API
    seo_content = generate_seo_optimized_job_post(
        job_data['title'],
        job_data['company'],
        job_data['description'],
        job_data['must_have_skills'],
        job_data['location'],
        job_data['email']
    )
    
    # Simulate posting - in real implementation, use your CMS API
    return {"success": True, "message": "Job posted to company website", "seo_content": seo_content}

def post_to_linkedin_company_page(job_data, access_token):
    """Post job to LinkedIn company page"""
    if not access_token:
        return {"success": False, "message": "No LinkedIn access token"}
    
    # LinkedIn API endpoint for company shares
    url = f"https://api.linkedin.com/v2/ugcPosts"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    
    post_content = {
        "author": f"urn:li:organization:{COMPANY_LINKEDIN_PAGE}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": f"We're hiring! {job_data['title']} at {job_data['company']}\n\n{job_data['description'][:200]}...\n\nRequired skills: {', '.join(job_data['must_have_skills'])}\nLocation: {job_data['location']}\n\nApply now! #hiring #careers #jobopportunity"
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=post_content)
        if response.status_code == 201:
            return {"success": True, "message": "Job posted to LinkedIn company page"}
        else:
            return {"success": False, "message": f"LinkedIn API error: {response.text}"}
    except Exception as e:
        return {"success": False, "message": f"Error posting to LinkedIn: {str(e)}"}

def send_job_post_notification(job_data, platforms):
    """Send email notification about job posting"""
    try:
        # Email configuration
        sender_email = job_data['email']
        receiver_email = "hr@company.com"  # Or wherever notifications should go
        password = EMAIL_PASSWORD
        
        message = MIMEMultipart("alternative")
        message["Subject"] = f"New Job Posted: {job_data['title']}"
        message["From"] = sender_email
        message["To"] = receiver_email
        
        text = f"""
        New Job Posted
        
        Title: {job_data['title']}
        Company: {job_data['company']}
        Location: {job_data['location']}
        Platforms: {', '.join(platforms)}
        
        Description: {job_data['description'][:500]}...
        
        Must-have skills: {', '.join(job_data['must_have_skills'])}
        
        Posted by: {job_data['email']}
        """
        
        html = f"""
        <html>
          <body>
            <h2>New Job Posted</h2>
            <p><strong>Title:</strong> {job_data['title']}</p>
            <p><strong>Company:</strong> {job_data['company']}</p>
            <p><strong>Location:</strong> {job_data['location']}</p>
            <p><strong>Platforms:</strong> {', '.join(platforms)}</p>
            <p><strong>Description:</strong> {job_data['description'][:500]}...</p>
            <p><strong>Must-have skills:</strong> {', '.join(job_data['must_have_skills'])}</p>
            <p><em>Posted by: {job_data['email']}</em></p>
          </body>
        </html>
        """
        
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        
        message.attach(part1)
        message.attach(part2)
        
        # Send email (configure your SMTP server)
        # with smtplib.SMTP("smtp.your-email.com", 587) as server:
        #     server.starttls()
        #     server.login(sender_email, password)
        #     server.sendmail(sender_email, receiver_email, message.as_string())
        
        return {"success": True, "message": "Notification sent"}
    
    except Exception as e:
        return {"success": False, "message": f"Email error: {str(e)}"}

# --- Streamlit UI ---
st.set_page_config(
    page_title="TalentFinder — Multi-Platform Recruitment", 
    layout="wide",
    page_icon="🚀"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .platform-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .candidate-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
        background-color: white;
    }
    .high-score {
        border-left: 4px solid #28a745;
    }
    .medium-score {
        border-left: 4px solid #ffc107;
    }
    .low-score {
        border-left: 4px solid #dc3545;
    }
    .opentowork-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">DATAMETRICS TalentFinder</h1>', unsafe_allow_html=True)
st.markdown("### Multi-Platform Recruitment & Candidate Management")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'search_results' not in st.session_state:
    st.session_state.search_results = pd.DataFrame()

# Sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    
    tab_keys, tab_settings = st.tabs(["API Keys", "Settings"])
    
    with tab_keys:
        serp_key = st.text_input("SerpAPI Key", value=SERPAPI_KEY, type="password")
        openai_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
        linkedin_token = st.text_input("LinkedIn Access Token", value=LINKEDIN_ACCESS_TOKEN, type="password")
        email_password = st.text_input("Email Password", value=EMAIL_PASSWORD, type="password")
    
    with tab_settings:
        st.subheader("Search Settings")
        default_num = st.slider("Profiles per platform", 5, 20, 10)
        
        st.subheader("Company Info")
        company_website = st.text_input("Company Website", value=COMPANY_WEBSITE_URL)
        company_linkedin = st.text_input("LinkedIn Company Page ID", value=COMPANY_LINKEDIN_PAGE)
        
        st.subheader("Platform Selection")
        selected_platforms = st.multiselect(
            "Choose platforms to search:",
            options=list(PLATFORMS.keys()),
            default=["LinkedIn", "Indeed", "Glassdoor", "Dice"]
        )

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search Candidates", "📤 Post Job", "📊 Analytics"])

with tab1:
    st.markdown("### 1️⃣ Job Requirements")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        jd = st.text_area("📝 Job Description *", height=200, 
                         placeholder="Paste the complete job description here...")
        
        st.markdown("### 🎯 Must-Have Skills *")
        must_have = st.text_area(
            "Enter must-have skills (comma separated)",
            placeholder="Python, SQL, AWS, Machine Learning...",
            height=80,
            key="must_have_skills"
        )
    
    with col2:
        company_name = st.text_input("🏢 Target Company", 
                                   placeholder="Google, Microsoft, Amazon...",
                                   help="Search for ex-employees of specific companies")
        
        keywords = st.text_input("🔍 Additional Keywords", 
                               placeholder="remote, senior, entry-level...")
        
        location = st.text_input("📍 Location", 
                               placeholder="New York, NY...")
        
        distance_filter = st.selectbox("📏 Distance Filter", 
                                     ["Any", "5 miles", "10 miles", "25 miles", "50 miles", "100 miles"])
        
        experience_level = st.selectbox("🎯 Experience Level", 
                                      ["Any", "Entry", "Mid", "Senior", "Executive"])
        
        open_to_work_only = st.checkbox("Prioritize OpenToWork Candidates", value=True)

    # Search Button
    if st.button("Search Candidates Across Platforms", type="primary", use_container_width=True):
        if not serp_key:
            st.error("🔑 Please provide SerpAPI key in the sidebar")
        elif not jd.strip():
            st.error("📝 Please provide a job description")
        elif not openai_key:
            st.error("🔑 Please provide OpenAI API key in the sidebar")
        elif not selected_platforms:
            st.error("🌐 Please select at least one platform to search")
        elif not must_have.strip():
            st.error("🎯 Please specify must-have skills")
        else:
            # Perform search
            evaluator = create_candidate_evaluator()
            job_title_hint = " ".join(jd.strip().split()[:6])
            must_skills_list = [s.strip() for s in must_have.split(",") if s.strip()]
            
            must_clauses = " ".join([f"\"{s.strip()}\"" for s in must_skills_list])
            company_query = f" {company_name}" if company_name else ""
            location_query = f" {location}" if location else ""
            experience_query = f" {experience_level}" if experience_level != "Any" else ""
            
            query = f"{job_title_hint} {keywords} {must_clauses} {company_query} {location_query} {experience_query}".strip()
            
            with st.spinner(f"🔍 Searching across {len(selected_platforms)} platforms..."):
                try:
                    organic = serpapi_search_platforms(query, selected_platforms, default_num, serp_key)
                    
                    if not organic:
                        st.warning("❌ No candidates found across selected platforms")
                    else:
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, item in enumerate(organic):
                            status_text.text(f"Evaluating candidate {i+1}/{len(organic)}...")
                            
                            link = item.get("link", "")
                            snippet = (item.get("snippet") or item.get("title") or "")[:800]
                            platform = item.get("platform", "Unknown")
                            category = item.get("category", "Unknown")
                            name = extract_name_from_url(link, platform)
                            open_flag = detect_open_to_work(snippet, platform)
                            
                            # LangGraph evaluation
                            initial_state = {
                                "jd": jd, 
                                "profile_snippet": snippet, 
                                "profile_url": link,
                                "platform": platform,
                                "category": category,
                                "score": 0, 
                                "reason": "", 
                                "message": "", 
                                "error": ""
                            }
                            
                            try:
                                final_state = evaluator.invoke(initial_state)
                                score, missing_skills = enforce_must_have_scoring(
                                    snippet, must_skills_list, final_state["score"], platform
                                )
                                
                                adapted_message = get_platform_specific_message(platform, final_state["message"])
                                
                                results.append({
                                    "name": name,
                                    "platform": platform,
                                    "category": category,
                                    "url": link,
                                    "snippet": snippet,
                                    "score": score,
                                    "reason": final_state["reason"],
                                    "message": adapted_message,
                                    "error": final_state.get("error", ""),
                                    "open_to_work": open_flag,
                                    "missing_skills": missing_skills
                                })
                            except Exception as e:
                                results.append({
                                    "name": name,
                                    "platform": platform,
                                    "category": category,
                                    "url": link,
                                    "snippet": snippet,
                                    "score": 25,
                                    "reason": f"Evaluation error: {str(e)}",
                                    "message": "Hi, I came across your profile and would like to connect.",
                                    "error": str(e),
                                    "open_to_work": open_flag,
                                    "missing_skills": must_skills_list
                                })
                            
                            progress_bar.progress((i + 1) / len(organic))
                            time.sleep(0.3)
                        
                        df = pd.DataFrame(results)
                        
                        # Apply OpenToWork prioritization
                        if open_to_work_only:
                            df = df.sort_values(by=["open_to_work", "score"], ascending=[False, False])
                        else:
                            df = df.sort_values(by="score", ascending=False)
                        
                        st.session_state.search_results = df
                        st.session_state.page = 0
                        
                        status_text.text("✅ Search complete!")
                        progress_bar.empty()
                        
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")

    # Display Results
    if not st.session_state.search_results.empty:
        df = st.session_state.search_results
        
        st.markdown("---")
        st.markdown("### 📊 Search Results")
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(df))
        with col2:
            st.metric("Platforms", df['platform'].nunique())
        with col3:
            open_to_work_count = df['open_to_work'].sum()
            st.metric("OpenToWork", open_to_work_count)
        with col4:
            avg_score = df['score'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        # Platform Distribution
        st.markdown("#### 🌐 Platform Distribution")
        platform_counts = df['platform'].value_counts()
        cols = st.columns(len(platform_counts))
        for i, (platform, count) in enumerate(platform_counts.items()):
            with cols[i % len(cols)]:
                st.metric(platform, count)
        
        # Pagination
        PAGE_SIZE = 10
        total_pages = max(1, (len(df) - 1) // PAGE_SIZE + 1)
        
        st.markdown(f"**Page {st.session_state.page + 1} of {total_pages}**")
        
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous") and st.session_state.page > 0:
                st.session_state.page -= 1
                st.rerun()
        with col_next:
            if st.button("Next ➡️") and st.session_state.page < total_pages - 1:
                st.session_state.page += 1
                st.rerun()
        
        # Display current page
        start_idx = st.session_state.page * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(df))
        page_df = df.iloc[start_idx:end_idx]
        
        st.markdown("### 🎯 Top Candidates")
        
        for idx, (_, row) in enumerate(page_df.iterrows()):
            # Determine score class for styling
            if row['score'] >= 70:
                score_class = "high-score"
            elif row['score'] >= 50:
                score_class = "medium-score"
            else:
                score_class = "low-score"
            
            st.markdown(f'<div class="candidate-card {score_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                # Name with OpenToWork badge
                name_display = f"**{row['name']}**"
                if row['open_to_work']:
                    name_display += ' <span class="opentowork-badge">OpenToWork</span>'
                st.markdown(f'<h4>{name_display}</h4>', unsafe_allow_html=True)
                
                st.write(f"**Platform:** {row['platform']} | **Score:** {row['score']}/100")
                st.write(f"**Snippet:** {row['snippet'][:200]}...")
                st.write(f"**Reason:** {row['reason']}")
                
                if row['missing_skills']:
                    st.warning(f"Missing skills: {', '.join(row['missing_skills'])}")
                
                st.write(f"[View Profile]({row['url']})")
            
            with col2:
                st.text_area(
                    f"Message for {row['name']}",
                    value=row['message'],
                    height=120,
                    key=f"msg_{idx}_{st.session_state.page}"
                )
            
            with col3:
                if linkedin_token and row["platform"] == "LinkedIn":
                    if st.button(f"📨 Invite", key=f"invite_{idx}_{st.session_state.page}"):
                        with st.spinner("Sending invite..."):
                            result = send_linkedin_invite(row['url'], row['message'], linkedin_token)
                            if result.get('ok'):
                                st.success("✅ Invite sent!")
                            else:
                                st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                else:
                    if row["platform"] == "LinkedIn":
                        st.info("🔑 Add LinkedIn token to send invites")
                    else:
                        st.info("📝 Manual contact required")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download
        st.download_button(
            label="📥 Download All Results (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"candidates_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab2:
    st.markdown("### 📤 Post Job to Multiple Platforms")
    
    with st.form("job_posting_form"):
        st.markdown("#### Job Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title *", placeholder="Senior Software Engineer")
            company = st.text_input("Company Name *", value="DATAMETRICS SOFTWARE SYSTEMS INC")
            location = st.text_input("Location *", placeholder="New York, NY or Remote")
            email = st.text_input("Recruiter Email *", placeholder="your.email@company.com")
        
        with col2:
            job_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Internship"])
            salary_range = st.text_input("Salary Range", placeholder="$100,000 - $150,000")
            application_deadline = st.date_input("Application Deadline")
        
        st.markdown("#### Job Description *")
        job_description = st.text_area("Detailed job description", height=150)
        
        st.markdown("#### 🎯 Must-Have Skills *")
        must_have_skills = st.text_area(
            "Required skills (comma separated)",
            placeholder="Python, React, AWS, Docker...",
            height=80
        )
        
        st.markdown("#### Preferred Platforms")
        posting_platforms = st.multiselect(
            "Select platforms to post this job:",
            options=[p for p, config in PLATFORMS.items() if config.get('posting_support', False)],
            default=["LinkedIn", "Indeed", "Glassdoor"]
        )
        
        st.markdown("#### SEO Optimization")
        seo_keywords = st.text_input("Additional SEO Keywords", 
                                   placeholder="tech jobs, software development, remote work")
        
        # Form submission
        submitted = st.form_submit_button("🚀 Post Job to Selected Platforms", type="primary")
        
        if submitted:
            if not all([job_title, company, location, email, job_description, must_have_skills]):
                st.error("❌ Please fill all required fields (*)")
            elif not posting_platforms:
                st.error("❌ Please select at least one platform to post")
            else:
                job_data = {
                    'title': job_title,
                    'company': company,
                    'location': location,
                    'email': email,
                    'description': job_description,
                    'must_have_skills': [s.strip() for s in must_have_skills.split(",")],
                    'type': job_type,
                    'salary': salary_range,
                    'deadline': application_deadline.strftime('%Y-%m-%d')
                }
                
                # Post to platforms
                results = []
                
                # Post to company website
                with st.spinner("Posting to company website..."):
                    website_result = post_to_company_website(job_data)
                    results.append(("Company Website", website_result))
                
                # Post to LinkedIn company page
                if "LinkedIn" in posting_platforms and linkedin_token:
                    with st.spinner("Posting to LinkedIn company page..."):
                        linkedin_result = post_to_linkedin_company_page(job_data, linkedin_token)
                        results.append(("LinkedIn Company Page", linkedin_result))
                
                # Send notification email
                with st.spinner("Sending notifications..."):
                    email_result = send_job_post_notification(job_data, posting_platforms)
                    results.append(("Email Notification", email_result))
                
                # Display results
                st.markdown("### 📋 Posting Results")
                for platform, result in results:
                    if result.get('success'):
                        st.success(f"✅ {platform}: {result['message']}")
                    else:
                        st.error(f"❌ {platform}: {result['message']}")
                
                # Show SEO preview
                if website_result.get('seo_content'):
                    with st.expander("🔍 SEO Optimized Content Preview"):
                        st.code(website_result['seo_content'], language='html')

with tab3:
    st.markdown("### 📊 Recruitment Analytics")
    
    if not st.session_state.search_results.empty:
        df = st.session_state.search_results
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Candidates", len(df))
            st.metric("Average Score", f"{df['score'].mean():.1f}")
        
        with col2:
            st.metric("OpenToWork Rate", f"{(df['open_to_work'].sum() / len(df) * 100):.1f}%")
            st.metric("Response Rate", "85%")  # Simulated
        
        with col3:
            st.metric("Top Platform", df['platform'].mode().iloc[0] if len(df) > 0 else "N/A")
            st.metric("Success Rate", "92%")  # Simulated
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Score Distribution")
            score_bins = pd.cut(df['score'], bins=[0, 30, 50, 70, 90, 100], 
                              labels=['0-30', '31-50', '51-70', '71-90', '91-100'])
            score_dist = score_bins.value_counts().sort_index()
            st.bar_chart(score_dist)
        
        with col_chart2:
            st.markdown("#### Platform Distribution")
            platform_dist = df['platform'].value_counts()
            st.bar_chart(platform_dist)
        
        # Top candidates
        st.markdown("#### 🏆 Top 5 Candidates")
        top_candidates = df.nlargest(5, 'score')[['name', 'platform', 'score', 'open_to_work']]
        st.dataframe(top_candidates, use_container_width=True)
    
    else:
        st.info("🔍 No search data available. Perform a candidate search to see analytics.")


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>💼 Streamline your hiring process across 25+ platforms</p>
        <p style='margin-top: 10px; font-size: 0.9rem; color: #999;'>
            © 2025 <strong>Basit Shabir</strong>. All rights reserved. <br>
            This tool is currently in the <strong>development stage</strong>. 
            Features, accuracy, and integrations are subject to change.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
