import os
import streamlit as st
import requests

# ----------------------------
# Config
# ----------------------------
DEFAULT_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Settings")
url = st.sidebar.text_input(
    "FastAPI endpoint URL",
    value=os.getenv("API_URL", DEFAULT_URL),
)
st.sidebar.caption("Make sure your FastAPI server is running before predicting.")

# ----------------------------
# Main UI
# ----------------------------
st.title("📧 Email Spam Detector")
st.write("Enter an email or message below and check whether it's spam or not.")

message = st.text_area(
    "Enter the email message here",
    height=180,
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim..."
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

# ----------------------------
# Prediction logic
# ----------------------------
if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    elif not url.strip():
        st.warning("Please provide the FastAPI endpoint URL in the sidebar.")
    else:
        try:
            with st.spinner("Predicting..."):
                response = requests.post(url, json={"message": message}, timeout=10)
            response.raise_for_status()
            result = response.json()

            prediction = result.get("response")

            st.markdown("---")
            if prediction is None:
                st.warning("API response didn't contain a 'response' field. Raw response below:")
                st.json(result)
            else:
                label = str(prediction).lower()
                if "spam" in label and "ham" not in label:
                    st.error(f"🚨 **This looks like SPAM**")
                else:
                    st.success(f"✅ **This looks like HAM (Not Spam)**")

                with st.expander("Raw API response"):
                    st.json(result)

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Couldn't connect to the API at `{url}`. "
                "Make sure your FastAPI server is running."
            )
        except requests.exceptions.Timeout:
            st.error("❌ The request to the API timed out. Please try again.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API returned an error: {e}")
        except ValueError:
            st.error("❌ The API response wasn't valid JSON.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

st.markdown("---")
st.caption("Frontend built with Streamlit. Backend powered by FastAPI.")