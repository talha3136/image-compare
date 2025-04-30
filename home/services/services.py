import os
import shutil
from tempfile import NamedTemporaryFile
import requests
from django.core.files import File
from django.conf import settings
from django.core.files.storage import default_storage
class FileRelatedService:
    
    @staticmethod
    def convert_url_to_file(url):
        if url:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                # Make the request
                response = requests.get(url, headers=headers, stream=True)

                if response.status_code != 200:
                    print(f"Error: Unable to retrieve file. HTTP Status code: {response.status_code}")
                    return None

                # Create a temporary file
                temp_file = NamedTemporaryFile(delete=False)
                for chunk in response.iter_content(chunk_size=128):
                    temp_file.write(chunk)
                temp_file.close()

                # Get the file extension from the URL or the response header
                file_name = os.path.basename(url)
                file_extension = file_name.split('.')[-1]

                # Construct the path where the file will be saved under the `uploads` directory
                upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)
                
                # Move the temporary file to the upload folder
                shutil.move(temp_file.name, upload_path)

                # Return the path of the saved file instead of a file object
                return upload_path
            
            except requests.exceptions.RequestException as e:
                print(f"Request Exception: Error retrieving file from URL: {url}. Error: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error occurred while retrieving file from {url}: {e}")
                return None
        else:
            print("Error: URL is None or empty")
            return None
