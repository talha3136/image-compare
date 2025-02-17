import os
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve
from django.core.files import File
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
class FileRelatedService:
    
    @staticmethod
    def convert_url_to_file(url):
        if url:
            try:
                # Verify the URL is reachable before attempting to download
                with urlopen(url) as response:
                    if response.status != 200:
                        return None
                
                temp_file = NamedTemporaryFile(delete=True)
                urlretrieve(url, temp_file.name)
                temp_file.seek(0)
                return File(temp_file)
            except (URLError, HTTPError) as e:
                print(f"Error retrieving file from URL: {url}. Error: {e}")
                return None
        else:
            return None