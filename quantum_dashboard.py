import streamlit as st
import matplotlib.pyplot as plt
from training.train import train_model
from evaluation.evaluate import evaluate_model

# App config
st.set_page_config(page_title="Quantum Transformer Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Dashboard", "Info"])

if page == "Dashboard":
    st.title("🧠 Quantum Transformer Training Dashboard")

    st.markdown("""
    This dashboard provides a visual interface for training and evaluating a hybrid quantum-classical transformer model.
    Use the controls on the sidebar to train the model or run evaluation.
    """)

    # Train model button
    if st.button("🚀 Train Model"):
        with st.spinner("Training the model..."):
            metrics = train_model()
            if metrics:
                st.success("✅ Training Complete!")
            else:
                st.error("Training failed or metrics not returned.")

        if metrics is not None:
            st.subheader("📊 Training Metrics")
            col1, col2 = st.columns(2)

            with col1:
                st.line_chart({"Loss": metrics["loss"]})
                st.line_chart({"Fidelity": metrics["fidelity"]})
            with col2:
                st.line_chart({"Gate Depth": metrics["gate_depth"]})
                st.line_chart({"Inference Time": metrics["inference_time"]})
                st.line_chart({"Epoch Time": metrics["epoch_time"]})
        else:
            st.warning("⚠️ Metrics not available. Please make sure training was successful.")

    # Evaluate model button
    if st.button("🔍 Evaluate Model"):
        with st.spinner("Evaluating the model..."):
            predicted_tokens, decoded = evaluate_model()
        st.success("✅ Evaluation Complete!")
        st.subheader("🧾 Evaluation Output")
        st.write("**Predicted Tokens:**", predicted_tokens)
        st.write("**Decoded Tokens:**", decoded)

elif page == "Info":
    st.title("📘 Framework Info")

    st.markdown("""
    ### 🔬 Quantum Transformer Project Overview

    This project implements a hybrid **quantum-classical transformer model** to explore how quantum circuits can enhance learning in sequence-based NLP tasks like character-level prediction.

    #### ⚙️ Key Frameworks Used:

    | Framework       | Purpose                                                                 |
    |-----------------|-------------------------------------------------------------------------|
    | **PyTorch**     | For building and training the transformer model                         |
    | **PennyLane**   | For quantum circuit simulation and integration with PyTorch             |
    | **Streamlit**   | For building this interactive dashboard                                 |
    | **Matplotlib**  | For plotting any local graphs or additional custom visualizations       |
    | **TorchMetrics**| (Optional) For enhanced metric tracking if included in future upgrades  |

    #### 🧪 Quantum Circuit Role:
    - Each token is passed through a **quantum circuit layer** with multiple qubits and rotations.
    - The quantum layer acts like a non-linear encoder inside the transformer block.

    #### 💡 Why This Is Undiscovered:
    - Quantum + Transformer hybrids are still **rare in applied NLP**, especially in **char-level** tasks.
    - This model paves the way for testing **quantum advantages in text modeling**.
    - It can scale to real quantum backends in the future via PennyLane's device support.

    #### 🧠 Possible Applications:
    - Secure sequence generation
    - Advanced language modeling for quantum cryptography
    - Quantum-enhanced NLP token encoders

    ---
    Made with ❤️ by Quantum Enthusiasts.
    """)

