import os
import pandas as pd
import re

def process_csv_files(data_folder):
    # Regex to match the filename pattern and extract the subject ID
    # Pattern: DS2_Sub_XXX_Face_SDDefault.csv -> Sub_XXX
    pattern = re.compile(r"DS2_(Sub_\d+)_Face_SDDefault\.csv")

    for filename in os.listdir(data_folder):
        match = pattern.match(filename)
        if match:
            sub_id = match.group(1)
            input_path = os.path.join(data_folder, filename)
            output_filename = f"{sub_id}.csv"
            output_path = os.path.join(data_folder, output_filename)

            print(f"Processing {filename} -> {output_filename}")

            try:
                # Read the CSV file
                df = pd.read_csv(input_path)
                
                # Clean column names (strip whitespace) to ensure we find the columns
                df.columns = df.columns.str.strip()

                # Check if required columns exist
                if 'Timestamp' in df.columns and 'rPPG_POS_Raw' in df.columns:
                    # Select columns: Timestamp and rPPG_POS_Raw
                    # We copy rPPG_POS_Raw to the new dataframe
                    new_df = df[['Timestamp', 'rPPG_POS_Raw']].copy()
                    
                    # Rename rPPG_POS_Raw to rPPG_signal as requested
                    new_df.rename(columns={'rPPG_POS_Raw': 'rPPG_signal'}, inplace=True)
                    
                    # Save to new CSV
                    new_df.to_csv(output_path, index=False)
                    print(f"Saved {output_path} with columns: {list(new_df.columns)}")
                else:
                    print(f"Skipping {filename}: Missing required columns. Found: {list(df.columns)}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Assuming the data folder is 'data' relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    if os.path.exists(data_dir):
        process_csv_files(data_dir)
    else:
        print(f"Data directory not found: {data_dir}")
