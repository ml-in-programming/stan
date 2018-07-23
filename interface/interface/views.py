import sys
import os
sys.path.append(os.getcwd() + '/backend/')

from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from project_predictor import run_backend

import git
import json

# from .forms import NameForm


def home(request):
    return HttpResponse('Welcome to the Tinyapp\'s Homepage!', content_type='text/plain')


def about(request):
    title = 'Tinyapp'
    author = 'Egor'
    html = render_to_string('request_page.html', {'title': title, 'author': author})
    return HttpResponse(html)


@csrf_exempt
def get_name(request):
    if request.method == 'GET':
        address = request.GET.get('address', '')
        # if address != '':
        #     run_backend(address)
    else:
        address = ''

    return render(request, 'request_page.html', {'address': address})


def load_content(request):
    result = 'Invalid request'
    if request.method == 'POST':
        address = request.POST.get('address')
        counts = request.POST.get('count')
        if address != '':
            try:
                result = run_backend(address, int(counts))
            except git.GitCommandError:
                result = json.dumps({'error': 'Invalid repository address'})

    return HttpResponse(result)
