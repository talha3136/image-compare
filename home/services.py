import os
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve
from django.core.files import File
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import requests
from tempfile import NamedTemporaryFile
from django.core.files import File

class FileRelatedService:
    
    @staticmethod
    def convert_url_to_file(url):
        if url:
            try:

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }


                response = requests.get(url, headers=headers, stream=True)

                if response.status_code != 200:
                    print(f"Error: Unable to retrieve file. HTTP Status code: {response.status_code}")
                    return None


                temp_file = NamedTemporaryFile(delete=True)
                for chunk in response.iter_content(chunk_size=128):
                    temp_file.write(chunk)
                temp_file.seek(0)
                
                return File(temp_file)
            
            except requests.exceptions.RequestException as e:
                print(f"Request Exception: Error retrieving file from URL: {url}. Error: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error occurred while retrieving file from {url}: {e}")
                return None
        else:
            print("Error: URL is None or empty")
            return None
