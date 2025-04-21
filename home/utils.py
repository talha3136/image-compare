from datetime import datetime
from django.conf import settings
from rest_framework.pagination import PageNumberPagination
from django.http import HttpResponse
from django.shortcuts import redirect
from django.utils import timezone
from functools import wraps

def delete_file(file_field):
    if file_field:
        file_field.delete(save=False)





def get_file_object_info(file):

    if not file:
        return None
    file_size_mb = round(file.size / (1024 * 1024), 2)  # Convert size to MB
    file_name = file.name.split('/')[-1]  # Extract the file name without the path
    return {
        'url': file.url,
        'name': file_name,
        'type': file_name.split('.')[-1],
        'size': file_size_mb,  
    }



def append_datetime_to_filepath(func):
    @wraps(func)
    def wrapper(instance, filename):
        # Get current datetime in the specified format
        current_datetime = timezone.now().strftime('%Y%m%d%H%M%S')
        
        # Get the original filepath
        filepath = func(instance, filename)
        
        # Insert the datetime as a subfolder before the filename
        directory, filename = filepath.rsplit('/', 1)
        updated_filepath = f'{directory}/{current_datetime}/{filename}'
        
        return updated_filepath
    
    return wrapper

@append_datetime_to_filepath
def get_uniform_checker_image(instance, filename):
    filepath = f'uniform_checker_images/{filename}'
    return filepath


class DefaultPagination(PageNumberPagination):
    ordering = 'id'
    page_size_query_param = 'page_size'
    page_size = 10
    max_page_size = 50

class SiteDefaultPagination(PageNumberPagination):
    ordering = 'id'
    page_size_query_param = 'page_size'
    page_size = 10
    max_page_size = 5000




def server_running(request):
    content = """
    <html>
    <head>
        <style>
            body {
                  background: linear-gradient(
                    to right,
                    #6f7280,
                    #2a2c3c,
                    #181827,
                    #833ab4,
                    #fd1d1d,
                    #fcb045
                );
                background-size: 400% 400%;
                animation: body 10s infinite ease-in-out;

            }
            h1 {
                font-family: system-ui;
                color: black;
                font-size: 30px;
                text-align: center;
                margin-top: 50px;
                margin-top: 50px;
            }
            p{
                text-align: center;
                font-size: 2rem;
            }
            div{
                margin-top:100px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            h5{
                text-align: center;
                color: darkolivegreen;
            }
            
            @keyframes body {
            0% {
                background-position: 0 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0 50%;
            }
            }

        </style>
    </head>
    <body>
    <div>
    	<img src="https://vms.arez.io/images/logos/favicon_512x512.png">
        </div>
        <h1>VMS-AREZ server is online and operational.</h1> 
        <h5 class="text-center">V 3.10</h5>
    </body>
    </html>

    """
    return HttpResponse(content)


def redirect_to_arez(request):
    return redirect({settings.AREZ_LINK})