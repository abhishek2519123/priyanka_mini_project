import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SPAM_KEYWORDS = [
    "wire transfer",
    "western union",
    "money transfer",
    "bitcoin",
    "cryptocurrency",
    "unlimited earning",
    "guaranteed",
    "no experience needed",
    "work from home",
    "start earning immediately",
    "processing fee",
    "registration fee",
    "starter kit",
    "commission",
    "make $",
    "earn $",
    "send $",
    "paypal",
    "mystery shopper",
    "paid surveys",
    "envelope stuffing",
    "upfront payment",
    "activate your account"
]

def contains_spam_keywords(text):

    text_lower = text.lower()

    return any(keyword in text_lower for keyword in SPAM_KEYWORDS)

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text

@st.cache_resource
def train_job_detector():

    df = pd.read_excel("FakeJobPostings (2).xlsx")

    df = df[['title', 'description', 'fraudulent']]

    df['title'] = df['title'].fillna("")

    df['description'] = df['description'].fillna("")

    df = df.dropna(subset=['fraudulent'])

    df['fraudulent'] = df['fraudulent'].astype(int)

    df['text'] = df['title'] + " " + df['description']

    df['cleaned'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df['cleaned'])

    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = MultinomialNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    return model, vectorizer, metrics

st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background-color: #050816;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #1e1e2f;
}

.main-header {
    color: white;
    text-align: center;
    padding: 20px;
}

.fake-job {
    background-color: #2b0b0b;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #ff4b4b;
    color: white;
}

.real-job {
    background-color: #0f2b16;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #00c853;
    color: white;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    border: none;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

textarea {
    background-color: #202436 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.header("🔧 Settings")

    with st.expander("📊 Model Information", expanded=True):

        st.markdown("### Model Type: Multinomial Naive Bayes")

        st.markdown("### Feature Extraction: TF-IDF Vectorizer")

        st.markdown("### Training Data: Job Postings Dataset")

    st.markdown("---")

    st.subheader("📋 Sample Jobs to Test")

    sample_jobs = {

       "✅ Real Job 1 - Senior Software Engineer": """
Senior Software Engineer - Python

Join our innovative tech company as a Senior Software Engineer. We're looking for experienced Python developers to build scalable applications. Responsibilities include: designing software solutions, writing clean code, code reviews, and mentoring junior developers.

Requirements: 5+ years Python experience, BS in Computer Science or equivalent, experience with Django/FastAPI. We offer competitive salary ($120-150K), health insurance, 401k matching, remote work options, and professional development budget.
        """,
        "✅ Real Job 2 - Marketing Manager": """
Head of Content - Online Media Company

Manage the English-speaking editorial team and build a team of best-in-class editors. Set up content creation schedules and ensure deadlines are adhered to. Research and write about the latest tech topics and news in relation to industry trends.

Requirements: Journalism/Media studies degree preferred, professional editorial experience, strong connections in key industries, leadership experience. Located in Berlin. Salary 20,000-28,000 EUR. Benefits include flat hierarchies, creative freedom, team events, and professional development.
        """,
        "✅ Real Job 3 - Accountant": """
Accounting Clerk - Environmental Consulting Firm

Apex Environmental Consulting seeks a self-motivated Accounts Payable Clerk. Process high volume of invoices and work in a fast-paced environment. Key in and verify various types of invoices to General Ledger accounts.

Requirements: High school diploma + 2-5 years accounting experience, knowledge of accounting software, advanced Excel, attention to detail, professionalism. Competitive benefits including health, dental, vision insurance, 401k, professional development.
        """,
        "✅ Real Job 4 - Healthcare Professional": """
HAAD/DHA Licensed Doctors Opening in UAE

Leading healthcare group in Abu Dhabi seeking specialist doctors. Positions available for: Endocrinologists, Gastroenterologists, Cardiologists, Neurologists, and other specialties.

Requirements: DHA/HAAD License required, board certification in specialty. Reputed healthcare group offering good standard of living and assured career growth. Competitive compensation, benefits, and opportunities for advancement.
        """,
        "✅ Real Job 5 - Customer Service Manager": """
Customer Service Team Lead - Novitex Solutions

Novitex Enterprise Solutions seeks a Customer Service Team Lead for our Dover, NH location. Manage customer communications, data entry operations, mail center activities, and quality standards.

Requirements: High school diploma + 1 year customer service experience, MS Word/Excel proficiency, ability to communicate effectively, handle 55 lbs, attention to detail, good attendance. Full benefits, salary commensurate with experience.
        """,
        "✅ Real Job 6 - Project Manager": """
Project Manager - Oil & Gas Industry

Valor Services seeks experienced Project Manager for offshore projects. Manage projects for major oil and gas exploration companies. Establish and maintain client relationships. Ensure projects delivered on time, within budget, to highest quality standards.

Requirements: BSc/MSc in Civil, Mechanical, or Petroleum Engineering, 10+ years offshore installation experience, 3+ years project management, multi-disciplinary team experience. Self-motivated with proven commercial success track record.
        """,
        
        # FRAUDULENT JOBS (40-100% SPAM)
        "⚠️ FAKE Job 1 - Wire Transfer Scam": """
EARN $5000 WEEKLY - WIRE TRANSFER PROCESSING!!!

URGENT HIRING!!! Work from anywhere! Make incredible money fast! No experience needed. Join our growing team of successful money processors!

WORK FROM HOME AND EARN BIG CASH FAST!!! We are looking for money transfer agents. Your job is simple: Receive funds via bank transfer. Process and redirect funds. Keep commission for each transfer. $5000+ PER WEEK POSSIBLE! Start immediately! Limited spots available. Act NOW before positions fill up!

Requirements: Must be 18+. Have active bank account. Can process wire transfers. Must be willing to work urgently. Unlimited earning potential. Weekly PayPal payments. Flexible schedule. Minimal commitment.
        """,
        "⚠️ FAKE Job 2 - Payment Processing": """
QUICK CASH JOBS - WORK FROM HOME!!!

Amazing opportunity for anyone looking to make fast cash!! We need reliable people to process payments and manage funds from home. NO EXPERIENCE NEEDED! Start earning immediately!!

Make $3000+ every week!! This is a REAL job opportunity! Simply process payment transactions from home. We will send you funds via bank transfer to manage and forward to our clients. You keep a commission from each transaction. NO EXPERIENCE NEEDED.

This is 100% LEGAL. We wire you funds. You forward them and keep 10-15% commission. Its that simple! Start Today! Limited positions available. Must have valid bank account.
        """,
        "⚠️ FAKE Job 3 - Data Entry Upfront Fee": """
Data Entry - WORK FROM HOME - $50/Hour Guaranteed

Exciting opportunity! Make up to $50 per hour doing simple data entry work from home. No experience necessary. We'll train you completely. To activate your account, please send a one-time payment of $150 via wire transfer or money order for processing fee.

URGENT positions available. Apply now and start earning tomorrow! Process customer records and make BIG MONEY. Limited spots - act fast! This is a legitimate opportunity for hardworking individuals. Start immediately with flexible hours.

Send $150 Western Union to activate. Email when done.
        """,
        "⚠️ FAKE Job 4 - Assembly Work Scam": """
Assembly Work From Home - Make $$$

Make money assembling products at home! Easy assembly work, no experience needed. We provide all materials. Earn $2500+ per month assembling simple items.

HOW IT WORKS: We send you materials. You assemble and send back. We pay you $10 per item. Simple! Start TODAY!

INITIAL INVESTMENT: $199 startup fee for materials kit. Send wire transfer or money order. Only 50 kits available! Get yours before they're gone!

Start earning immediately. Work your own hours. Unlimited earning potential. Contact us NOW!
        """,
        "⚠️ FAKE Job 5 - Email Forwarding": """
Email Processing Job - WORK FROM HOME

Process emails for businesses from your home computer. EASY MONEY for minimal work! Make $1500-2000 per month just forwarding emails and uploading documents.

NO EXPERIENCE REQUIRED. We train you 100%. You set your own hours. Work whenever you want. Start TODAY!

To qualify for this position, send $79.95 for training materials and access to our secure email forwarding platform via Western Union or wire transfer. Positions limited! Act now!

Once payment received, instant access to start earning immediately. Passive income opportunity. Join thousands of happy workers!
        """,
        "⚠️ FAKE Job 6 - Cryptocurrency/Bitcoin": """
Bitcoin Processor - EARN $8000 MONTHLY

Make HUGE profits processing cryptocurrency transactions! Simple work from home! No experience needed - we teach you everything!

CRYPTO is the future! Get in NOW and start earning! Process Bitcoin transactions. Earn 20% commission on each one. $8000+ per month GUARANTEED!

LIMITED OPPORTUNITY: To join our exclusive team, investment of $299 required. One time payment via Bitcoin/wire transfer. Spaces filling FAST!

We will provide you crypto wallet access and show you exactly how to make money. Start TODAY! Unlimited earning potential! This is a legitimate money-making opportunity!
        """,
        "⚠️ FAKE Job 7 - Mystery Shopper Scam": """
Mystery Shopper - Get PAID to Shop!

Get paid $50-150 per shopping visit! Have FUN while making EASY MONEY! No experience necessary!

HOW IT WORKS: We send you shopping assignments. You shop, take photos, complete survey. Get paid $50-150 per visit! Work flexible hours from home.

LIMITED POSITIONS: 100 spots available nationwide. To be considered, send $69.95 registration fee via wire transfer for background check and to access job assignments.

Your fee waived for first 10 applicants! Send money order to get started TODAY! Make money doing what you love!
        """,
        "⚠️ FAKE Job 8 - Envelope Stuffing": """
ENVELOPE STUFFING - Make $1500/Week!

Make $1500 per week stuffing envelopes at home! No experience necessary! This is legitimate work!

EASY MONEY: Stuff envelopes with promotional materials. We mail you the materials. You stuff them. We pick up. EARN $0.50-$1.00 per envelope!

GET STARTED TODAY: Send $99.95 processing fee via Western Union for starter kit and materials. Fee covers first month's supplies.

You will receive: 500 envelopes, materials, and access to daily job assignments. Start earning IMMEDIATELY after payment received! This is REAL income!

Popular work - positions limited. Send payment now!
        """,
        "⚠️ FAKE Job 9 - Paid Surveys": """
Online Surveys - Make Money From Home!

Get PAID $50-$200 per survey! Work from home! Make EASY MONEY in your spare time! NO EXPERIENCE NEEDED!

We need people to complete online surveys. Major companies pay us to get your opinions. We share the profits with you!

GUARANTEED: Minimum $50 per survey. Complete 3-4 surveys daily = $150-$800 daily income!

EXCLUSIVE ACCESS: Limited to 50 people. Join our program TODAY. $49 membership fee grants you lifetime access to surveys.

Send payment via wire transfer. Receive login details within 1 hour. Start making money immediately! This is legitimate! Join thousands earning monthly!
        """,
        "⚠️ FAKE Job 10 - Immediate Payment Request": """
Remote Opportunity - Earn Fast Cash

Work from home and earn money by helping us process small payments. No experience needed. Start today.

To join, pay a one-time $120 activation fee via Western Union. After payment, you will receive a login and begin earning immediately.

This is a limited-time position. Apply now before it closes.
        """,
        "⚠️ FAKE Job 11 - Investment / Crypto Pitch": """
Crypto Processing Specialist

We are hiring people to handle cryptocurrency payments and earn commissions. Simple remote work, flexible hours, no previous experience needed.

We promise fast cash and huge returns. Investment of $199 required to get started. Accept Bitcoin and wire transfer only.

Join our team today and start earning immediately.
        """,
        "⚠️ FAKE Job 12 - Paid To Forward Emails": """
Email Forwarding Assistant

Earn extra cash from home by forwarding emails and checking accounts. This role requires no training and offers weekly payouts.

A $79.95 membership fee is required to access the platform. Payment can be made by PayPal, wire transfer, or money order.

Start now and get paid daily.
        """
    }

    selected_sample = st.selectbox(
        "Select a sample:",
        list(sample_jobs.keys())
    )

    if st.button("📌 Load Sample"):

        st.session_state.sample_text = sample_jobs[selected_sample]

st.markdown("# 🛡️ Job Scam Detector")

st.markdown(
    "### Analyze job postings to determine if they are legitimate or fraudulent using AI."
)

st.markdown("---")

model, vectorizer, metrics = train_job_detector()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")

with col2:
    st.metric("Precision", f"{metrics['precision']:.2%}")

with col3:
    st.metric("Recall", f"{metrics['recall']:.2%}")

with col4:
    st.metric("F1 Score", f"{metrics['f1']:.2%}")

st.markdown("---")

if "sample_text" not in st.session_state:

    st.session_state.sample_text = ""

col_input, col_clear = st.columns([0.9, 0.1])

with col_input:

    job_text = st.text_area(
        "📝 Paste Job Title and Description:",
        value=st.session_state.sample_text,
        height=250,
        placeholder="Enter job posting text here..."
    )

with col_clear:

    st.write("")

    st.write("")

    if st.button("🗑️ Clear"):

        st.session_state.sample_text = ""

        st.rerun()

col_btn1, col_btn2, col_btn3 = st.columns([1,1,2])

with col_btn1:

    analyze_button = st.button(
        "🔍 Analyze Job",
        use_container_width=True
    )

if analyze_button:

    if not job_text.strip():

        st.warning("⚠️ Please enter job posting text.")

    else:

        cleaned_input = clean_text(job_text)

        vectorized_input = vectorizer.transform([cleaned_input])

        prediction = model.predict(vectorized_input)[0]

        confidence = model.predict_proba(vectorized_input).max()

        confidence_percent = round(confidence * 100, 2)

        spam_keywords_found = contains_spam_keywords(job_text)

        is_spam = prediction == 1 or spam_keywords_found

        st.markdown("---")

        st.subheader("📊 Analysis Results")

        col_result1, col_result2 = st.columns(2)

        with col_result1:

            if is_spam:

                st.markdown(f"""
                <div class="fake-job">
                    <h2>🚨 SPAM JOB ALERT</h2>
                    <h3>Confidence: {confidence_percent}%</h3>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="real-job">
                    <h2>✅ THIS PLACE IS NOT SPAM</h2>
                    <h3>Confidence: {confidence_percent}%</h3>
                </div>
                """, unsafe_allow_html=True)

        with col_result2:

            st.subheader("🔑 Keywords Analysis")

            scam_keywords = {
                "High Risk": [
                    "fee",
                    "pay",
                    "wire transfer",
                    "western union"
                ],

                "Medium Risk": [
                    "urgent",
                    "earn",
                    "easy",
                    "work from home"
                ]
            }

            detected_keywords = {}

            for risk_level, keywords in scam_keywords.items():

                found = [
                    word for word in keywords
                    if word in cleaned_input
                ]

                if found:

                    detected_keywords[risk_level] = found

            if detected_keywords:

                for risk_level, words in detected_keywords.items():

                    if risk_level == "High Risk":

                        st.error(
                            f"**{risk_level}:** {', '.join(words)}"
                        )

                    else:

                        st.warning(
                            f"**{risk_level}:** {', '.join(words)}"
                        )

            else:

                st.success("✅ No suspicious keywords detected")

        st.markdown("---")

        st.subheader("📈 Detailed Metrics")

        prob = model.predict_proba(vectorized_input)[0]

        d1, d2 = st.columns(2)

        with d1:

            st.metric(
                "Legitimate Job Probability",
                f"{prob[0]:.2%}"
            )

        with d2:

            st.metric(
                "Fraudulent Job Probability",
                f"{prob[1]:.2%}"
            )

        st.subheader("📝 Text Statistics")

        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "Original Length",
                f"{len(job_text)} characters"
            )

        with s2:

            st.metric(
                "Word Count",
                f"{len(job_text.split())} words"
            )

        with s3:

            st.metric(
                "Unique Words",
                f"{len(set(cleaned_input.split()))} unique"
            )

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:gray;padding:20px;">
🛡️ Job Scam Detector | Powered by Machine Learning
<br><br>
⚠️ Disclaimer: This tool provides analysis assistance.
Always conduct your own verification.
</div>
""", unsafe_allow_html=True)
