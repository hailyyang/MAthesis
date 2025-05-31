import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import re
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

# Streamlit app title
st.title("Stakes vs Rewards Log-Log Plot Generator")

# Define system prompt
SYSTEM_PROMPT = '''Generate a Python script for data visualization. Ensure:
- Only executable Python code is returned, with no explanations, comments, or plt.show() calls.
- It uses the variables raw_csv_path and processed_csv_path for file paths.
- It reads two CSV files:
    1. The raw data file at raw_csv_path.
    2. The processed CSV file at processed_csv_path.
    2a. The processed data has two columns, total_stakes and median_total_rewards. The data format for total_stakes is a RANGE, like "(a, b]", meaning from an amount of stakes from a to b, and median_total_rewards is the median total rewards for this bin of stakes. Preprocess this file by splitting the content of the cell into two floats, as such: lower, upper = range_str.strip("()[]").split(", ").
- Only keep data points whose total stakes is larger than 10**13.
- It creates a scatter plot of 'total_stakes' vs. 'total_rewards' from the raw data (blue, size=1, alpha=0.3, labeled 'Raw Data').
- It overlays a scatter plot of the median rewards per bin from the processed CSV (orange, labeled 'Median per bin').
- For the orange data points, only keep data points whose bin value (lower bound) is larger than 10**13.
- For the median total rewards per bin in logarithm scale, calculate the logarithm of the x and y values, fit a linear regression line (green, labeled 'Linear Regression') to these values. Then, exponentiate the logarithmic values to recalculate the real values of the linear regression line for display.
- Display the x and y axes in logarithm scale using plt.xscale('log') and plt.yscale('log').
- Handle 0 and negative values before applying np.log() by filtering them out.
- Handle missing values and drop any remaining NaNs as a final safeguard.
- Ensure correct column names are used.
- Save the plot as a PNG at plot_output_path with plt.savefig() and do NOT call plt.show().
- Convert relevant columns to numeric.
- Set the plot title to 'Stakes vs Rewards' and include legends.
- Use pandas for data loading, matplotlib.pyplot for plotting, numpy for numerical operations, and scipy.stats for linear regression.'''

# Sidebar for inputs
st.sidebar.header("Input Parameters")
raw_csv_file = st.sidebar.file_uploader("Upload Raw Data CSV", type=["csv"])
processed_csv_file = st.sidebar.file_uploader("Upload Processed Data CSV", type=["csv"])
user_prompt = st.sidebar.text_area(
    "Enter Plotting Prompt (leave blank to use default)",
    value=SYSTEM_PROMPT,
    height=300
)
plot_filename = st.sidebar.text_input("Output Plot Filename", value="stakes_vs_rewards_plot.png")

# Set Groq API key
load_dotenv()
# os.environ["GROQ_API_KEY"]

# Function to load files
def load_files(raw_csv_path, processed_csv_path):
    try:
        raw_df = pd.read_csv(raw_csv_path)
        processed_df = pd.read_csv(processed_csv_path)
        plot_output_path = raw_csv_path.replace("query_result.csv", "stakes_vs_rewards_plot.png")
        return (
            raw_df,
            processed_df,
            plot_output_path,
            f"Loaded files successfully. Plot will be saved as PNG at {plot_output_path}"
        )
    except Exception as e:
        return None, None, None, f"Error loading files: {str(e)}"

# Function to generate plot script
def generate_plot_script(raw_csv_path, processed_csv_path, plot_output_path, user_prompt):
    # Use user prompt if provided, otherwise fall back to system prompt
    plot_script_prompt = user_prompt if user_prompt.strip() else SYSTEM_PROMPT
    chat_model = ChatGroq(model_name="llama3-70b-8192", temperature=0.0)
    response = chat_model.invoke(plot_script_prompt)
    script_code = response.content.strip()
    
    # Extract Python code and clean up
    match = re.search(r'```python\n(.*?)```', script_code, re.DOTALL)
    script_code = match.group(1).strip() if match else script_code
    script_code = re.sub(r'^[^\n]*?\n', '', script_code, 1).strip()
    script_code = script_code.replace("```", "").strip()
    
    if not script_code:
        return None, "Error: Plot script generation failed."
    
    try:
        compile(script_code, '<string>', 'exec')
    except SyntaxError as e:
        return None, f"Syntax Error in generated script: {str(e)}\n\nScript:\n{script_code}"
    
    return script_code, script_code

# Function to execute plot script
def execute_plot_script(script_code, raw_csv_path, processed_csv_path, plot_output_path):
    try:
        # Verify all input files exist
        for path in [raw_csv_path, processed_csv_path]:
            if not os.path.exists(path):
                return f"Error: Input file not found: {path}", None
        
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_script:
            temp_script.write(script_code)
            temp_script_path = temp_script.name
        
        # Pass all file paths to exec_globals
        exec_globals = {
            "__file__": temp_script_path,
            "raw_csv_path": raw_csv_path,
            "processed_csv_path": processed_csv_path,
            "plot_output_path": plot_output_path
        }
        exec(open(temp_script_path).read(), exec_globals)
        os.remove(temp_script_path)
        
        if not os.path.exists(plot_output_path):
            return "Error: Plot file was not created.", None
        
        return f"Plot generated successfully and saved as PNG at {plot_output_path}.", plot_output_path
    except Exception as e:
        return f"Error executing plot script: {str(e)}\n\nScript:\n{script_code}", None

# Main app logic
if all([raw_csv_file, processed_csv_file]):
    # Save uploaded files to temporary locations
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_raw:
        temp_raw.write(raw_csv_file.getvalue())
        raw_csv_path = temp_raw.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_processed:
        temp_processed.write(processed_csv_file.getvalue())
        processed_csv_path = temp_processed.name
    
    # Load files
    raw_df, processed_df, plot_output_path, load_message = load_files(
        raw_csv_path, processed_csv_path
    )
    st.write(load_message)
    
    if raw_df is not None:
        # Generate plot script
        script_code, script_message = generate_plot_script(raw_csv_path, processed_csv_path, plot_output_path, user_prompt)
        st.write("Generated Python Script:")
        st.code(script_message, language="python")
        
        # Execute plot script
        exec_message, plot_path = execute_plot_script(script_code, raw_csv_path, processed_csv_path, plot_output_path)
        st.write(exec_message)
        
        if plot_path and os.path.exists(plot_path):
            # Display plot
            st.write("Generated Plot:")
            st.image(plot_path)
            
            # Download plot
            with open(plot_path, "rb") as f:
                st.download_button(
                    label="Download Plot",
                    data=f,
                    file_name=plot_filename,
                    mime="image/png"
                )
    
    # Clean up temporary files
    for path in [raw_csv_path, processed_csv_path]:
        if os.path.exists(path):
            os.unlink(path)
    if plot_path and os.path.exists(plot_path):
        os.unlink(plot_path)

else:
    st.warning("Please upload both required CSV files to proceed.")