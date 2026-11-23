"""
Streamlit code for RLHF integration
"""

import streamlit as st
from modules.rlhf_utils import start_rlhf, get_training_history, analyze_rating_patterns
import plotly.express as px
import pandas as pd
import torch
import os
import gc
    
    # Set CUDA environment variables for better multi-GPU performance
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use first two GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def show_rlhf_training_page():
    """RLHF training page"""
        
    st.title("RLHF Training with PPO")
    if 'model' in st.session_state:
        model_path = st.session_state.model
    
    if "generated_output" not in st.session_state:
        st.warning("Please generate policies first in RAG Generation tab.")
        return
    
    output = st.session_state.generated_output
    titles = st.session_state.generated_titles
    print(titles)
    
    # Display current outputs for reference
    st.subheader("Generated Policy Sections")
    
    for i, title in enumerate(titles):
        with st.expander(f"📋 {title}", expanded=False):
                content = output.get(title.rstrip(), output.get(title.rstrip(), "Content not found"))
                st.write(content[:500] + "..." if len(content) > 500 else content)
    
    st.divider()
    
    # Rating interface
    st.subheader("Rate Generated Sections")
    st.write("Rate each section from 0.0 (poor) to 1.0 (excellent):")
    
    with st.form("ratings_form"):
        ratings = dict()
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        for i, title in enumerate(titles):
            with col1 if i % 2 == 0 else col2:
                rating = st.slider(
                        f"📊 {title}",
                        0.0, 1.0, 0.5, 
                        step=0.1,
                        key=f"rating_{title}",
                        help=f"Rate the quality of: {title}"
                    )
                ratings[title] = rating
        sec_head = st.text_input("Additional section headings (separated by commas):") 
        sec_context = st.text_input("Additional context to be considered:")
        # Training options
        st.subheader("Training Options")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            reward_epochs = st.selectbox("Reward Model Epochs", [3, 5, 10], index=0)
        with col2:
            num_samples = st.selectbox("Samples per Section", [1, 2, 3], index=0)
        with col3:
            save_data = st.checkbox("Save Training Data", value=True)
        
        submit = st.form_submit_button("🚀 Start RLHF Training", type="primary")
    
    # Show current ratings summary
    if ratings:
        st.subheader("Current Ratings Summary")
        
        # Create a simple visualization
        rating_df = pd.DataFrame([
            {"Section": title, "Rating": rating} 
            for title, rating in ratings.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                rating_df, 
                x="Section", 
                y="Rating",
                title="Section Ratings",
                color="Rating",
                color_continuous_scale="RdYlGn"
            )
            #fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, width=True)
        
        with col2:
            avg_rating = sum(ratings.values()) / len(ratings)
            st.metric("Average Rating", f"{avg_rating:.2f}")
            st.metric("Total Sections", len(ratings))
            st.metric("Highest Rated", max(ratings, key=ratings.get))
            st.metric("Lowest Rated", min(ratings, key=ratings.get))
    
    # Training execution
    if submit:
        if len(sec_head)>0:
            for w in sec_head.split(','):
                st.session_state.generated_titles.append(w)
        st.session_state.sec_context = sec_context
        with st.spinner("Running RLHF Training... This may take a few minutes."):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing training data...")
            progress_bar.progress(20)
            
            # Run training function
            try:
                result = start_rlhf(ratings, output, model_path, reward_epochs, num_samples)
                
                status_text.text("Training reward model...")
                progress_bar.progress(60)
                
                status_text.text("Updating policy...")
                progress_bar.progress(80)
                
                status_text.text("Finalizing...")
                progress_bar.progress(100)
                
                st.success("✅ RLHF Training Completed!")
                st.success(result)
                
                # Show training results
                st.balloons()
                
                # Store results in session state
                if 'training_history' not in st.session_state:
                    st.session_state.training_history = []
                
                st.session_state.training_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'ratings': ratings,
                    'result': result,
                    'avg_rating': sum(ratings.values()) / len(ratings)
                })
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.error("Please check model path and try again.")
    
    # Training history section
    st.divider()
    st.subheader("Training History")
    
    # Load historical data
    history = get_training_history()
    
    if history or 'training_history' in st.session_state:
        tab1, tab2 = st.tabs(["Recent Sessions", "All Time Analytics"])
        
        with tab1:
            if 'training_history' in st.session_state and st.session_state.training_history:
                st.write("Recent training sessions from this browser session:")
                
                for i, session in enumerate(reversed(st.session_state.training_history[-5:])):
                    with st.expander(f"Session {len(st.session_state.training_history)-i} - Avg Rating: {session['avg_rating']:.2f}"):
                        st.write(f"**Time:** {session['timestamp']}")
                        st.write(f"**Result:** {session['result']}")
                        
                        # Show ratings
                        session_df = pd.DataFrame([
                            {"Section": k, "Rating": v} 
                            for k, v in session['ratings'].items()
                        ])
                        st.dataframe(session_df, use_container_width=True)
            else:
                st.info("No recent training sessions in this browser session.")
        
        with tab2:
            if history:
                st.write("Historical training data analysis:")
                analysis = analyze_rating_patterns(history)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Ratings", analysis.get('total_ratings', 0))
                with col2:
                    st.metric("Average Rating", f"{analysis.get('average_rating', 0):.3f}")
                with col3:
                    st.metric("Min Rating", f"{analysis.get('min_rating', 0):.3f}")
                with col4:
                    st.metric("Max Rating", f"{analysis.get('max_rating', 0):.3f}")
            else:
                st.info("No historical training data found.")
    else:
        st.info("No training history available. Complete a training session to see history.")

# Additional helper functions for Streamlit app

def show_model_comparison():
    """Optional: Compare performance before/after RLHF"""
    st.subheader("Model Performance Comparison")
    
    if 'training_history' in st.session_state and len(st.session_state.training_history) > 1:
        # Create comparison chart
        history_df = pd.DataFrame([
            {
                'Session': i+1,
                'Average Rating': session['avg_rating'],
                'Timestamp': session['timestamp']
            }
            for i, session in enumerate(st.session_state.training_history)
        ])
        
        fig = px.line(
            history_df,
            x='Session',
            y='Average Rating',
            title='Rating Improvement Over Training Sessions',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 training sessions to show comparison.")

def export_training_data():
    """Export training data for external analysis"""
    if 'training_history' in st.session_state:
        df = pd.DataFrame([
            {
                'timestamp': session['timestamp'],
                'section': section,
                'rating': rating,
                'session_id': i
            }
            for i, session in enumerate(st.session_state.training_history)
            for section, rating in session['ratings'].items()
        ])
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training Data as CSV",
            data=csv,
            file_name=f"rlhf_training_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def flush():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    for key in ["rag", "llm_model"]:
        if key in st.session_state:
            del st.session_state[key]
    flush()
    show_rlhf_training_page()
    