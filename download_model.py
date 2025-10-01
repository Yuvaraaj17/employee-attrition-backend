import os
from google.cloud import storage
import pathlib
# Remove 'from dotenv import load_dotenv' and 'load_dotenv(".env")' for Render

# --- Environment Variables (Read directly from the environment) ---
BUCKET_NAME = os.environ.get("BUCKET_NAME") 
SOURCE_BLOB_NAME = os.environ.get("SOURCE_BLOB_NAME")
DESTINATION_FILE_NAME = os.environ.get("DESTINATION_FILE_NAME")
# Note: No need to explicitly read the GCS key, as the Google library 
# handles authentication via GOOGLE_APPLICATION_CREDENTIALS automatically.
# --- End of Environment Variables ---

def download_model_from_gcs():
    """
    Authenticates automatically via the GOOGLE_APPLICATION_CREDENTIALS path 
    (set by Render's Secret File feature) and downloads the model.
    """
    
    # 1. Basic check for required variables
    if not all([BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME]):
        print("ERROR: One or more required environment variables (BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME) are missing.")
        return False
        
    try:
        # 2. Authenticate the client (automatic via GOOGLE_APPLICATION_CREDENTIALS)
        storage_client = storage.Client() 
        # The library looks at the env var GOOGLE_APPLICATION_CREDENTIALS
        # which points to the mounted secret file.

        # 3. Perform the download
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(SOURCE_BLOB_NAME)
        
        # Ensure the destination directory exists
        local_path = pathlib.Path(DESTINATION_FILE_NAME)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading gs://{BUCKET_NAME}/{SOURCE_BLOB_NAME} to {DESTINATION_FILE_NAME}...")
        
        # Download the file
        blob.download_to_filename(DESTINATION_FILE_NAME)
        
        print(f"✅ Download complete. Model available at: {DESTINATION_FILE_NAME}")
        return True

    except Exception as e:
        print(f"❌ An error occurred during model download from GCS: {e}")
        print("HINT: Check if the Service Account has the Storage Object Viewer role on the bucket.")
        return False

# --- Main Application Entry Point ---

if __name__ == "__main__":
    if download_model_from_gcs():
        # ... (Your main application logic here)
        print("\nStarting main application service...")
        pass