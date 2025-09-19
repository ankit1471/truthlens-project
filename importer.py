# importer.py - FINAL SUPER-SMART VERSION

from app import app, db, NewsArticle
import pandas as pd
import os

# Function ab file ka naam aur column ka naam, dono lega
def import_data_from_file(filename, column_name, label):
    
    # Check karo ki file .csv hai ya nahi, aur poora naam banao
    filepath = filename if filename.endswith('.csv') else filename + '.csv'
    
    print(f"--- Processing file: {filepath} (Using Column: '{column_name}', Label: '{label}') ---")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found -> {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
        print(f"Found {len(df)} articles in the file.")
    except Exception as e:
        print(f"ERROR: Could not read the file. Error: {e}")
        return

    if column_name not in df.columns:
        print(f"ERROR: Column '{column_name}' not found in {filepath}.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # Counter to keep track of added articles for this file
    articles_added_count = 0
    
    for index, row in df.iterrows():
        article_content = row[column_name]
        
        if pd.isna(article_content):
            continue

        new_article = NewsArticle(
            content=str(article_content),
            model_prediction=label, # Hum yahan label set karenge (REAL/FAKE)
            user_feedback=label,    # Hum is data ko 100% sahi maan rahe hain
        )
        db.session.add(new_article)
        articles_added_count += 1
    
    # Ek saath saare articles database mein save karo
    db.session.commit()
    print(f"SUCCESS: Imported {articles_added_count} articles from {filepath}!")


# Main execution block
if __name__ == '__main__':
    with app.app_context():
        
        # === YAHAN HAI ASLI JAADU ===
        # Humne ek list banayi hai jo batayegi ki kis file ke liye kaunsa column aur label use karna hai
        
        files_to_process = [
            # Real News Files
            {'filename': 'business_data', 'column': 'content', 'label': 'REAL'},
            {'filename': 'education_data', 'column': 'content', 'label': 'REAL'},
            {'filename': 'entertainment_data', 'column': 'content', 'label': 'REAL'},
            {'filename': 'sports_data', 'column': 'content', 'label': 'REAL'},
            {'filename': 'technology_data', 'column': 'content', 'label': 'REAL'},
            {'filename': 'True.csv', 'column': 'text', 'label': 'REAL'}, # True.csv ko bhi daal dete hain
            
            # Fake News File
            {'filename': 'Fake.csv', 'column': 'text', 'label': 'FAKE'}
        ]

        # Ab ek hi loop mein saari files process ho jayengi
        for file_info in files_to_process:
            import_data_from_file(
                filename=file_info['filename'], 
                column_name=file_info['column'], 
                label=file_info['label']
            )
            
    print("\n--- ALL FILES PROCESSED. DATABASE HAS BEEN UPDATED. ---")