import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Page configuration
st.set_page_config(
    page_title="CineSense AI - Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling with larger text
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 2rem;
        color: #6B7280;
        text-align: center;
        font-weight: 400;
        margin-bottom: 3rem;
        line-height: 1.5;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    
    .positive-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 176, 155, 0.3);
    }
    
    .negative-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
    }
    
    .example-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        color: white;
        border: none;
        font-size: 1.1rem;
    }
    
    .example-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
    }
    
    .confidence-meter {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #00b09b 100%);
        height: 20px;
        border-radius: 12px;
        margin: 2.5rem 0;
        position: relative;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .confidence-indicator {
        position: absolute;
        top: -8px;
        width: 8px;
        height: 36px;
        background: white;
        border-radius: 4px;
        transform: translateX(-50%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .stButton button {
        width: 100%;
        border-radius: 16px;
        height: 4rem;
        font-weight: 600;
        font-size: 1.4rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 2rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
    }
    
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(0, 176, 155, 0.3);
    }
    
    .tutor-card {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
        text-align: center;
        margin: 3rem 0;
    }
    
    .section-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 1.5rem;
    }
    
    .section-subtitle {
        font-size: 1.3rem;
        color: #6B7280;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    .feature-title {
        font-size: 1.5rem;
        color: #1F2937;
        margin-bottom: 1rem;
    }
    
    .feature-text {
        font-size: 1.2rem;
        color: #6B7280;
        line-height: 1.6;
    }
    
    .percentage-text {
        color: #1F2937 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# App header with gradient background
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 5rem 2rem; border-radius: 0 0 40px 40px; margin-bottom: 3rem;">
    <div style="max-width: 1200px; margin: 0 auto;">
        <h1 class="main-header">🎬 CineSense AI</h1>
        <p class="sub-header" style="color: white; opacity: 0.9;">
            Professional Sentiment Analysis for Movie Reviews<br>
            Powered by Pre-trained AI Model
        </p>
        <div style="text-align: center; margin-top: 2rem;">
            <span class="status-badge">✅ Model Connected & Ready</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div style="max-width: 1200px; margin: 0 auto;">', unsafe_allow_html=True)

# Load model with caching - UPDATED TO USE PRE-TRAINED MODEL
@st.cache_resource
def load_model():
    try:
        # Using a pre-trained sentiment analysis model
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        st.info("🔄 Loading pre-trained sentiment analysis model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

if tokenizer and model:
    # Main layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Write Your Review")
        st.markdown('<p class="section-subtitle">Share your thoughts about any movie and get instant sentiment analysis.</p>', unsafe_allow_html=True)
        
        # Quick examples in a beautiful card
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 20px; margin: 2rem 0;">
            <h4 style="color: white; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.4rem;">💡 Quick Examples</h4>
        </div>
        """, unsafe_allow_html=True)
        
        examples = {
            "😊 Positive": [
                "This movie was absolutely fantastic! The acting was superb and the story was captivating.",
                "Brilliant cinematography and outstanding performances by the entire cast.",
                "One of the best films I've seen this year. Highly recommended!"
            ],
            "😞 Negative": [
                "Terrible movie with poor acting and a boring storyline.",
                "The plot made no sense and the dialogue was awful.",
                "Weak character development and predictable twists."
            ]
        }
        
        # Display examples
        for sentiment, sentiment_examples in examples.items():
            st.markdown(f"<h4 style='font-size: 1.3rem; margin-bottom: 1rem;'>{sentiment}</h4>", unsafe_allow_html=True)
            for i, example in enumerate(sentiment_examples):
                if st.button(
                    f"{example[:70]}...", 
                    key=f"{sentiment}_{i}",
                    use_container_width=True
                ):
                    st.session_state.review_text = example
        
        # Text input
        review_text = st.text_area(
            "**Your Movie Review:**",
            height=180,
            value=st.session_state.get('review_text', ''),
            placeholder="Share your honest thoughts about the movie...\n\nExample: 'The visual effects were stunning and the acting was phenomenal, though the plot felt somewhat predictable in the second act.'",
            label_visibility="collapsed"
        )
        
        # Analyze button
        analyze_clicked = st.button(
            "**🚀 Analyze Sentiment**", 
            type="primary", 
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Results")
        st.markdown('<p class="section-subtitle">Real-time sentiment analysis powered by pre-trained AI model.</p>', unsafe_allow_html=True)
        
        if analyze_clicked and review_text.strip():
            with st.spinner("""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
                <h3 style="color: #667eea; font-size: 1.8rem;">Analyzing Your Review</h3>
                <p style="font-size: 1.2rem;">Processing with AI model...</p>
            </div>
            """):
                # Process the review
                inputs = tokenizer(review_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = predictions[0][predicted_class].item()
                    
                    # For this model: 0 = NEGATIVE, 1 = POSITIVE
                    positive_prob = predictions[0][1].item()
                    negative_prob = predictions[0][0].item()
                
                # Determine sentiment
                sentiment = "Positive" if predicted_class == 1 else "Negative"
                
                # Display sentiment result
                if sentiment == "Positive":
                    st.markdown("""
                    <div class="positive-card">
                        <div style="display: flex; align-items: center; gap: 1.5rem;">
                            <div style="font-size: 5rem;">🎉</div>
                            <div>
                                <h2 style="margin: 0; font-size: 2.5rem; font-weight: 700;">Positive Review</h2>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.9;">Great! This review shows positive sentiment</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="negative-card">
                        <div style="display: flex; align-items: center; gap: 1.5rem;">
                            <div style="font-size: 5rem;">💔</div>
                            <div>
                                <h2 style="margin: 0; font-size: 2.5rem; font-weight: 700;">Negative Review</h2>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.9;">This review shows negative sentiment</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence section
                st.markdown("#### 📈 Confidence Level")
                
                # Visual confidence meter
                confidence_percent = int(confidence * 100)
                
                st.markdown(f"""
                <div style="position: relative; margin: 2.5rem 0;">
                    <div class="confidence-meter">
                        <div class="confidence-indicator" style="left: {confidence_percent}%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 1.2rem; color: #6B7280; margin-top: 0.5rem;">
                        <span>Low</span>
                        <span>Medium</span>
                        <span>High</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence text
                confidence_level = "Very High" if confidence > 0.9 else "High" if confidence > 0.7 else "Moderate" if confidence > 0.6 else "Low"
                confidence_emoji = "🎯" if confidence > 0.9 else "👍" if confidence > 0.7 else "🤔" if confidence > 0.6 else "⚠️"
                
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{confidence_emoji}</div>
                    <h3 style="margin: 0; color: #1F2937; font-size: 1.8rem;">{confidence_level} Confidence</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #6B7280; font-size: 1.3rem;">{confidence:.1%} certainty in this analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics
                st.markdown("#### 🔍 Detailed Analysis")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">😊</div>
                        <h2 style="margin: 0; color: #1F2937; font-size: 2.5rem; font-weight: 700;">{positive_prob:.1%}</h2>
                        <p style="margin: 0; color: #1F2937; font-size: 1.2rem; font-weight: 600;">Positive Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">😞</div>
                        <h2 style="margin: 0; color: #1F2937; font-size: 2.5rem; font-weight: 700;">{negative_prob:.1%}</h2>
                        <p style="margin: 0; color: #1F2937; font-size: 1.2rem; font-weight: 600;">Negative Score</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif analyze_clicked and not review_text.strip():
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); padding: 2.5rem; border-radius: 16px; border-left: 4px solid #f59e0b;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
                    <h3 style="color: #92400e; margin-bottom: 0.5rem; font-size: 1.8rem;">Review Needed</h3>
                    <p style="margin: 0; color: #92400e; font-size: 1.2rem;">Please enter a movie review to analyze</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); padding: 3.5rem 2rem; border-radius: 16px; border-left: 4px solid #3b82f6;">
                <div style="text-align: center;">
                    <div style="font-size: 5rem; margin-bottom: 1.5rem;">📊</div>
                    <h3 style="color: #1e40af; margin-bottom: 1rem; font-size: 1.8rem;">Ready for Analysis</h3>
                    <p style="margin: 0; color: #1e40af; font-size: 1.2rem;">Enter a movie review and click "Analyze Sentiment" to see detailed results from our AI model.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Features section
    st.markdown("---")
    st.markdown('<h2 class="section-title">✨ Why Choose CineSense AI</h2>', unsafe_allow_html=True)

    features_col1, features_col2, features_col3 = st.columns(3)

    with features_col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <div class="feature-icon">🤖</div>
            <h4 class="feature-title">Pre-trained AI Model</h4>
            <p class="feature-text">Powered by DistilBERT fine-tuned on sentiment analysis for accurate movie review insights</p>
        </div>
        """, unsafe_allow_html=True)

    with features_col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <div class="feature-icon">⚡</div>
            <h4 class="feature-title">Instant Results</h4>
            <p class="feature-text">Get comprehensive sentiment analysis in seconds with detailed confidence metrics</p>
        </div>
        """, unsafe_allow_html=True)

    with features_col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
            <div class="feature-icon">🎯</div>
            <h4 class="feature-title">Professional Grade</h4>
            <p class="feature-text">Enterprise-level accuracy with beautiful, intuitive interface for seamless user experience</p>
        </div>
        """, unsafe_allow_html=True)

    # Acknowledgment section
    st.markdown("---")
    st.markdown("""
    <div class="tutor-card">
        <div style="font-size: 4rem; margin-bottom: 2rem;">👨‍🏫</div>
        <h2 style="margin: 0 0 2rem 0; font-size: 2.5rem;">Special Acknowledgments</h2>
        <p style="font-size: 1.4rem; margin: 1rem 0; opacity: 0.9;">
            <strong>Project Developed by:</strong><br>
            Uwuilekhue Blessed
        </p>
        <p style="font-size: 1.4rem; margin: 1rem 0; opacity: 0.9;">
            <strong>Under the Guidance of:</strong><br>
            Emmanuel Ani
        </p>
        <p style="font-size: 1.2rem; margin: 2rem 0 0 0; opacity: 0.8;">
            This sentiment analysis project represents the culmination of dedicated learning and development in machine learning and AI deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 3rem 0 1rem 0;">
        <p style="margin: 0; font-size: 1.2rem; font-weight: 600;">🎬 CineSense AI - Professional Sentiment Analysis</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.8;">
            Developed by Uwuilekhue Blessed • Guided by Emmanuel Ani • Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("""
    ❌ Failed to load the model. Please check:
    - You have a stable internet connection for downloading the model
    - The required dependencies are installed
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""